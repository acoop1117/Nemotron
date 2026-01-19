# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Xenna pipeline runner for Nemotron data prep."""

from __future__ import annotations

from dataclasses import asdict
import gc
import json
import threading
import time
from typing import TYPE_CHECKING

import cosmos_xenna.pipelines.v1 as pipelines_v1


def _log_memory_status(label: str) -> None:
    """Log memory usage for debugging OOM issues."""
    try:
        import psutil
        process = psutil.Process()
        rss_gb = process.memory_info().rss / (1024**3)
        print(f"[Memory] {label}: RSS={rss_gb:.2f} GB")
    except ImportError:
        print(f"[Memory] {label}: psutil not available")

    try:
        import ray
        if ray.is_initialized():
            resources = ray.available_resources()
            obj_store = resources.get("object_store_memory", 0) / (1024**3)
            print(f"[Ray] {label}: object_store_available={obj_store:.2f} GB")
    except Exception as e:
        print(f"[Ray] {label}: error getting status - {e}")

if TYPE_CHECKING:
    from cosmos_xenna.pipelines.private.monitoring_types import PipelineStats

from nemotron.data_prep.xenna.stages import (
    ChatSftCentralPackStage,
    ChatSftSpoolStage,
    HfPredownloadStage,
    JsonlShardStage,
    PretrainShardStage,
)
from nemotron.data_prep.xenna.work_items import (
    ChatSftShardWorkItem,
    JsonlShardWorkItem,
    ShardWorkItem,
)


def run_xenna_pipeline(
    *,
    execution_plans: list,
    output_config,
    output_root: str,
    fs,
    live_status,
    results: dict,
    max_concurrent_downloads: int = 64,
    wandb_log_downloads: bool = False,
    wandb_log_pipeline_stats: bool = False,
    wandb_download_log_interval_sec: int = 30,
    hf_download_timeout_sec: int = 300,
    hf_download_max_retries: int = 3,
) -> None:
    """Run shard processing via Xenna pipeline."""
    if not execution_plans:
        return

    resolved_tokenizer = execution_plans[0].plan.resolved_tokenizer
    for ep in execution_plans[1:]:
        if ep.plan.resolved_tokenizer != resolved_tokenizer:
            raise ValueError(
                f"Tokenizer mismatch: dataset '{ep.name}' uses different tokenizer. "
                "Xenna executor requires uniform tokenizer across datasets in v1."
            )

    tasks: list[ShardWorkItem] = []
    for ep in execution_plans:
        live_status.start_dataset(ep.name)
        live_status.report_phase(ep.name, "processing", "xenna")

        assignment_dicts = {}
        for a in ep.plan.file_assignments:
            assignment_dicts[a.shard_index] = {
                "shard_index": a.shard_index,
                "files": [asdict(f) for f in a.files],
                "total_bytes": a.total_bytes,
            }

        for shard_idx in ep.pending_indices:
            tasks.append(
                ShardWorkItem(
                    dataset_name=ep.name,
                    plan_hash=ep.plan.plan_hash,
                    shard_index=shard_idx,
                    assignment=assignment_dicts[shard_idx],
                    output_dir=ep.dataset_dir,
                    receipts_dir=ep.receipts_dir,
                    text_field=ep.config.text_field,
                    dtype=output_config.dtype,
                    min_doc_chars=output_config.min_doc_chars,
                    max_doc_tokens=output_config.max_doc_tokens,
                    max_rows=output_config.max_rows,
                )
            )

    if not tasks:
        return

    print(f"[Xenna] Launching pipeline for {len(tasks)} shard(s)")

    pipeline_spec = pipelines_v1.PipelineSpec(
        input_data=tasks,
        stages=[
            pipelines_v1.StageSpec(
                HfPredownloadStage(
                    max_concurrent_downloads=max_concurrent_downloads,
                    output_root=output_root,
                    download_timeout_sec=hf_download_timeout_sec,
                    max_retries=hf_download_max_retries,
                ),
                num_workers_per_node=1,
            ),
            pipelines_v1.StageSpec(
                PretrainShardStage(
                    resolved_tokenizer=resolved_tokenizer,
                    output_root=output_root,
                ),
            )
        ],
        config=pipelines_v1.PipelineConfig(
            execution_mode=pipelines_v1.ExecutionMode.STREAMING,
            return_last_stage_outputs=False,
        ),
    )

    dataset_pending_counts = {ep.name: len(ep.pending_indices) for ep in execution_plans}

    stop_event = threading.Event()

    def _poll_receipts() -> None:
        last_counts: dict[str, int] = {ep.name: 0 for ep in execution_plans}
        seen_receipts: dict[str, set[str]] = {ep.name: set() for ep in execution_plans}
        tokens_by_dataset: dict[str, int] = {ep.name: 0 for ep in execution_plans}
        while not stop_event.is_set():
            for ep in execution_plans:
                try:
                    if not fs.exists(ep.receipts_dir):
                        continue
                    entries = [p for p in fs.ls(ep.receipts_dir, detail=False) if str(p).endswith(".json")]
                    count = len(entries)
                except Exception:
                    continue

                last = last_counts.get(ep.name, 0)
                if count > last:
                    new_entries = []
                    for p in entries:
                        if p not in seen_receipts[ep.name]:
                            seen_receipts[ep.name].add(p)
                            new_entries.append(p)

                    for receipt_path in new_entries:
                        try:
                            with fs.open(receipt_path, "r") as f:
                                receipt = json.load(f)
                            tokens_by_dataset[ep.name] += _extract_tokens(receipt)
                        except Exception:
                            pass

                    for _ in range(count - last):
                        live_status.advance_dataset(ep.name)
                    last_counts[ep.name] = count
                    live_status.report_tokens(ep.name, tokens_by_dataset[ep.name])
            stop_event.wait(10.0)

    poll_thread = threading.Thread(target=_poll_receipts, daemon=True)
    poll_thread.start()

    wandb_thread = None
    if wandb_log_downloads:
        wandb_thread = threading.Thread(
            target=_poll_download_stats,
            args=(fs, output_root, stop_event, wandb_download_log_interval_sec),
            daemon=True,
        )
        wandb_thread.start()

    try:
        pipelines_v1.run_pipeline(pipeline_spec)
    finally:
        stop_event.set()
        poll_thread.join(timeout=2.0)
        if wandb_thread is not None:
            wandb_thread.join(timeout=2.0)

    for ep in execution_plans:
        results[ep.name] = _aggregate_stats_from_receipts(ep.receipts_dir, ep.plan, fs)
        live_status.report_metrics(
            ep.name,
            rows=results[ep.name].get("total_sequences", 0),
            tokens=results[ep.name].get("total_tokens", 0),
        )
        live_status.complete_dataset(ep.name)


def _aggregate_stats_from_receipts(receipts_dir: str, plan, fs) -> dict:
    """Import-free wrapper; actual implementation lives in pipeline.py."""
    from nemotron.data_prep.pipeline import _aggregate_stats_from_receipts as _agg

    return _agg(receipts_dir, plan, fs)


def run_xenna_jsonl_pipeline(
    *,
    tasks: list[JsonlShardWorkItem],
    dataset_infos: list[dict],
    output_root: str,
    fs,
    live_status,
    results: dict,
    text_field: str,
    transform,
    compression: str,
    max_rows: int | None,
    resolve_hf_placeholders: bool,
    max_concurrent_downloads: int = 64,
    wandb_log_downloads: bool = False,
    wandb_log_pipeline_stats: bool = False,
    wandb_download_log_interval_sec: int = 30,
    hf_download_timeout_sec: int = 300,
    hf_download_max_retries: int = 3,
) -> None:
    if not tasks:
        return

    pipeline_spec = pipelines_v1.PipelineSpec(
        input_data=tasks,
        stages=[
            pipelines_v1.StageSpec(
                HfPredownloadStage(
                    max_concurrent_downloads=max_concurrent_downloads,
                    output_root=output_root,
                    download_timeout_sec=hf_download_timeout_sec,
                    max_retries=hf_download_max_retries,
                ),
                num_workers_per_node=1,
            ),
            pipelines_v1.StageSpec(
                JsonlShardStage(
                    output_root=output_root,
                    text_field=text_field,
                    transform=transform,
                    compression=compression,
                    max_rows=max_rows,
                    resolve_hf_placeholders=resolve_hf_placeholders,
                ),
                # Limit workers to prevent OOM on large datasets
                num_workers=4,
            ),
        ],
        config=pipelines_v1.PipelineConfig(
            execution_mode=pipelines_v1.ExecutionMode.STREAMING,
            return_last_stage_outputs=False,
        ),
    )

    stop_event = threading.Event()

    def _poll_receipts() -> None:
        last_counts: dict[str, int] = {info["name"]: 0 for info in dataset_infos}
        seen_receipts: dict[str, set[str]] = {info["name"]: set() for info in dataset_infos}
        tokens_by_dataset: dict[str, int] = {info["name"]: 0 for info in dataset_infos}
        while not stop_event.is_set():
            for info in dataset_infos:
                name = info["name"]
                receipts_dir = info["receipts_dir"]
                try:
                    if not fs.exists(receipts_dir):
                        continue
                    entries = [p for p in fs.ls(receipts_dir, detail=False) if str(p).endswith(".json")]
                    count = len(entries)
                except Exception:
                    continue

                last = last_counts.get(name, 0)
                if count > last:
                    new_entries = []
                    for p in entries:
                        if p not in seen_receipts[name]:
                            seen_receipts[name].add(p)
                            new_entries.append(p)

                    for receipt_path in new_entries:
                        try:
                            with fs.open(receipt_path, "r") as f:
                                receipt = json.load(f)
                            tokens_by_dataset[name] += _extract_tokens(receipt)
                        except Exception:
                            pass

                    for _ in range(count - last):
                        live_status.advance_dataset(name)
                    last_counts[name] = count
                    live_status.report_tokens(name, tokens_by_dataset[name])
            stop_event.wait(10.0)

    poll_thread = threading.Thread(target=_poll_receipts, daemon=True)
    poll_thread.start()

    wandb_thread = None
    if wandb_log_downloads:
        wandb_thread = threading.Thread(
            target=_poll_download_stats,
            args=(fs, output_root, stop_event, wandb_download_log_interval_sec),
            daemon=True,
        )
        wandb_thread.start()

    try:
        pipelines_v1.run_pipeline(pipeline_spec)
    finally:
        stop_event.set()
        poll_thread.join(timeout=2.0)
        if wandb_thread is not None:
            wandb_thread.join(timeout=2.0)

    for info in dataset_infos:
        name = info["name"]
        stats = _aggregate_jsonl_stats_from_receipts(
            dataset_dir=info["dataset_dir"],
            num_shards=info["num_shards"],
            fs=fs,
        )
        results[name] = stats
        live_status.report_metrics(
            name,
            rows=stats.get("num_records", 0),
            tokens=stats.get("total_tokens", 0),
        )
        live_status.complete_dataset(name)


def run_xenna_chat_sft_pipeline(
    *,
    tasks: list[ChatSftShardWorkItem],
    dataset_infos: list[dict],
    output_root: str,
    fs,
    live_status,
    results: dict,
    resolved_tokenizer: dict,
    messages_field: str,
    tools_field: str,
    pack_size: int,
    algorithm: str,
    dtype: str,
    chat_template: str | None,
    max_doc_tokens: int | None,
    max_rows: int | None,
    seed: int | None,
    used_in_filter: str | None,
    used_in_field: str,
    max_concurrent_downloads: int = 64,
    wandb_log_downloads: bool = False,
    wandb_log_pipeline_stats: bool = False,
    wandb_download_log_interval_sec: int = 30,
    hf_download_timeout_sec: int = 300,
    hf_download_max_retries: int = 3,
) -> None:
    if not tasks:
        return

    stages: list[pipelines_v1.StageSpec] = [
        pipelines_v1.StageSpec(
            HfPredownloadStage(
                max_concurrent_downloads=max_concurrent_downloads,
                output_root=output_root,
                download_timeout_sec=hf_download_timeout_sec,
                max_retries=hf_download_max_retries,
            ),
            num_workers_per_node=1,
        ),
        pipelines_v1.StageSpec(
            ChatSftSpoolStage(
                resolved_tokenizer=resolved_tokenizer,
                output_root=output_root,
                messages_field=messages_field,
                tools_field=tools_field,
                pack_size=pack_size,
                algorithm=algorithm,
                dtype=dtype,
                chat_template=chat_template,
                max_doc_tokens=max_doc_tokens,
                max_rows=max_rows,
                seed=seed,
                used_in_filter=used_in_filter,
                used_in_field=used_in_field,
            ),
            # Let Xenna auto-scale - spool stage is memory-efficient
        ),
        pipelines_v1.StageSpec(
            ChatSftCentralPackStage(
                output_root=output_root,
                pack_size=pack_size,
                algorithm=algorithm,
                dtype=dtype,
                seed=seed,
            ),
            num_workers=1,  # Must be single-worker for centralized packing
        ),
    ]

    pipeline_spec = pipelines_v1.PipelineSpec(
        input_data=tasks,
        stages=stages,
        config=pipelines_v1.PipelineConfig(
            execution_mode=pipelines_v1.ExecutionMode.STREAMING,
            return_last_stage_outputs=False,
        ),
    )

    stop_event = threading.Event()

    def _poll_receipts() -> None:
        last_counts: dict[str, int] = {info["name"]: 0 for info in dataset_infos}
        seen_receipts: dict[str, set[str]] = {info["name"]: set() for info in dataset_infos}
        tokens_by_dataset: dict[str, int] = {info["name"]: 0 for info in dataset_infos}
        while not stop_event.is_set():
            for info in dataset_infos:
                name = info["name"]
                receipts_dir = info["receipts_dir"]
                try:
                    if not fs.exists(receipts_dir):
                        continue
                    entries = [p for p in fs.ls(receipts_dir, detail=False) if str(p).endswith(".json")]
                    count = len(entries)
                except Exception:
                    continue

                last = last_counts.get(name, 0)
                if count > last:
                    new_entries = []
                    for p in entries:
                        if p not in seen_receipts[name]:
                            seen_receipts[name].add(p)
                            new_entries.append(p)

                    for receipt_path in new_entries:
                        try:
                            with fs.open(receipt_path, "r") as f:
                                receipt = json.load(f)
                            tokens_by_dataset[name] += _extract_tokens(receipt)
                        except Exception:
                            pass

                    for _ in range(count - last):
                        live_status.advance_dataset(name)
                    last_counts[name] = count
                    live_status.report_tokens(name, tokens_by_dataset[name])
            stop_event.wait(10.0)

    poll_thread = threading.Thread(target=_poll_receipts, daemon=True)
    poll_thread.start()

    wandb_thread = None
    if wandb_log_downloads:
        wandb_thread = threading.Thread(
            target=_poll_download_stats,
            args=(fs, output_root, stop_event, wandb_download_log_interval_sec),
            daemon=True,
        )
        wandb_thread.start()

    _log_memory_status("Before run_pipeline")
    try:
        pipelines_v1.run_pipeline(pipeline_spec)
    finally:
        _log_memory_status("After run_pipeline (in finally)")
        stop_event.set()
        poll_thread.join(timeout=2.0)
        if wandb_thread is not None:
            wandb_thread.join(timeout=2.0)

    _log_memory_status("After thread cleanup")

    # Force garbage collection to release memory from pipeline
    gc.collect()
    _log_memory_status("After gc.collect()")

    for info in dataset_infos:
        name = info["name"]
        _log_memory_status(f"Before aggregating {name}")
        stats = _aggregate_packed_stats_from_receipts(
            dataset_dir=info["dataset_dir"],
            receipts_dir=info["receipts_dir"],
            fs=fs,
        )
        results[name] = stats
        live_status.report_metrics(
            name,
            rows=stats.get("num_sequences", 0),
            tokens=stats.get("total_tokens", 0),
        )
        live_status.complete_dataset(name)

    _log_memory_status("After all aggregation - pipeline complete")


def _aggregate_jsonl_stats_from_receipts(*, dataset_dir: str, num_shards: int, fs) -> dict:
    from nemotron.data_prep.pipeline import _aggregate_jsonl_stats as _agg

    return _agg(dataset_dir, num_shards, fs)


def _aggregate_packed_stats_from_receipts(*, dataset_dir: str, receipts_dir: str, fs) -> dict:
    from nemotron.data_prep.pipeline import _aggregate_packed_stats as _agg

    return _agg(dataset_dir, receipts_dir, fs)


def _extract_tokens(receipt: dict) -> int:
    return int(receipt.get("stats", {}).get("total_tokens", 0))


def _make_wandb_stats_callback():
    """Create a callback function for logging pipeline stats to wandb.

    Returns a callback if wandb is active, None otherwise.
    """
    try:
        import wandb
    except ImportError:
        return None

    if wandb.run is None:
        return None

    def _log_stats(stats: "PipelineStats") -> None:
        """Log PipelineStats to wandb."""
        metrics = {
            # Overall pipeline progress
            "data_prep/pipeline_duration_min": stats.pipeline_duration_s / 60,
            "data_prep/inputs_initial": stats.num_initial_input_tasks,
            "data_prep/inputs_remaining": stats.num_input_tasks_remaining,
            "data_prep/outputs_total": stats.num_outputs,
            "data_prep/main_loop_rate_hz": stats.main_loop_rate_hz,
            # Cluster resources
            "data_prep/cluster_cpus_total": stats.cluster.total.num_cpus,
            "data_prep/cluster_cpus_available": stats.cluster.available.num_cpus,
            "data_prep/cluster_gpus_total": stats.cluster.total.num_gpus,
            "data_prep/cluster_gpus_available": stats.cluster.available.num_gpus,
            "data_prep/cluster_memory_total_gb": stats.cluster.total.memory / 1e9,
            "data_prep/cluster_memory_available_gb": stats.cluster.available.memory / 1e9,
        }

        # Progress percentage
        if stats.num_initial_input_tasks > 0:
            progress = 1.0 - (stats.num_input_tasks_remaining / stats.num_initial_input_tasks)
            metrics["data_prep/pipeline_progress"] = progress

        # Per-stage resource usage
        for stage_name, usage in stats.resource_usage_per_stage.items():
            safe_name = stage_name.replace(" ", "_").replace("-", "_")
            metrics[f"data_prep/stage_{safe_name}_cpu_pct"] = usage.cpu_utilization
            metrics[f"data_prep/stage_{safe_name}_memory_gb"] = usage.memory_usage / 1e9
            metrics[f"data_prep/stage_{safe_name}_actor_count"] = usage.actor_count

        # Per-stage state from actor pools
        for pool_stats in stats.actor_pools:
            safe_name = pool_stats.name.replace(" ", "_").replace("-", "_")
            # Actor counts
            metrics[f"data_prep/stage_{safe_name}_actors_target"] = pool_stats.actor_stats.target
            metrics[f"data_prep/stage_{safe_name}_actors_ready"] = pool_stats.actor_stats.ready
            metrics[f"data_prep/stage_{safe_name}_actors_running"] = pool_stats.actor_stats.running
            metrics[f"data_prep/stage_{safe_name}_actors_idle"] = pool_stats.actor_stats.idle
            # Task stats
            metrics[f"data_prep/stage_{safe_name}_tasks_completed"] = pool_stats.task_stats.total_completed
            metrics[f"data_prep/stage_{safe_name}_input_queue_size"] = pool_stats.task_stats.input_queue_size
            metrics[f"data_prep/stage_{safe_name}_output_queue_size"] = pool_stats.task_stats.output_queue_size
            # Slot stats
            metrics[f"data_prep/stage_{safe_name}_slots_used"] = pool_stats.slot_stats.num_used
            metrics[f"data_prep/stage_{safe_name}_slots_empty"] = pool_stats.slot_stats.num_empty
            # Speed
            if pool_stats.processing_speed_tasks_per_second is not None:
                metrics[f"data_prep/stage_{safe_name}_speed_tasks_per_sec"] = pool_stats.processing_speed_tasks_per_second

        wandb.log(metrics)

    return _log_stats


def _poll_download_stats(fs, output_root: str, stop_event: threading.Event, interval_sec: int) -> None:
    try:
        import wandb
    except ImportError:
        return

    if wandb.run is None:
        return

    progress_dir = f"{output_root.rstrip('/')}/.xenna/downloads"
    last_logged = 0.0

    while not stop_event.is_set():
        now = time.time()
        if now - last_logged < interval_sec:
            stop_event.wait(1.0)
            continue
        last_logged = now

        try:
            if not fs.exists(progress_dir):
                continue
            entries = fs.ls(progress_dir, detail=False)
        except Exception:
            continue

        total_completed = 0
        total_files = 0
        max_elapsed = 0.0
        max_rate = 0.0

        for path in entries:
            if not str(path).endswith(".json"):
                continue
            try:
                with fs.open(path, "r") as f:
                    data = json.load(f)
            except Exception:
                continue
            total_completed += int(data.get("completed", 0))
            total_files += int(data.get("total", 0))
            max_elapsed = max(max_elapsed, float(data.get("elapsed_sec", 0.0)))
            max_rate = max(max_rate, float(data.get("rate", 0.0)))

        if total_files == 0:
            continue

        wandb.log(
            {
                "data_prep/hf_download_completed": total_completed,
                "data_prep/hf_download_total": total_files,
                "data_prep/hf_download_rate": max_rate,
                "data_prep/hf_download_elapsed_sec": max_elapsed,
            }
        )
