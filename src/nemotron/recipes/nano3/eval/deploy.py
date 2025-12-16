#!/usr/bin/env python3

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

"""Deploy script for nano3 model evaluation.

This script deploys a model using Ray Serve and either:
- Waits for external done signal (--run mode, CLI orchestrates evaluation)
- Orchestrates evaluation itself (--batch mode with --orchestrate flag)

Usage:
    # --run mode: deploy only, wait for external done signal
    python deploy.py --config /path/to/config.yaml

    # --batch mode: deploy + orchestrate evaluation
    python deploy.py --config /path/to/config.yaml --orchestrate
"""

from __future__ import annotations

import argparse
import json
import logging
import socket
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

LOGGER = logging.getLogger("NeMo")


@dataclass
class DeployConfig:
    """Configuration for model deployment."""

    checkpoint_path: str
    host: str = "0.0.0.0"
    port: int = 8000
    num_gpus: int = 8
    tensor_model_parallel_size: int = 4
    expert_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    context_parallel_size: int = 1
    model_id: str = "nano3-eval"
    model_type: str = "gpt"
    model_format: str = "megatron"
    max_batch_size: int = 32
    num_replicas: int = 1
    num_cpus_per_replica: float = 8


@dataclass
class EvalConfig:
    """Configuration for model evaluation."""

    tasks: list[dict] = field(default_factory=list)
    parallelism: int = 32
    max_retries: int = 5
    request_timeout: int = 360
    max_new_tokens: int = 8192
    limit_samples: int | None = None
    output_dir: str = "/nemo_run/eval_results"
    extra: dict = field(default_factory=dict)
    slurm: dict = field(default_factory=dict)


@dataclass
class CommsConfig:
    """Configuration for coordination files."""

    base_dir: str = "/nemo_run/eval_comms"
    endpoint_file: str = "endpoint.json"
    completion_file: str = "done"


@dataclass
class Config:
    """Combined configuration for deploy + eval."""

    deploy: DeployConfig
    eval: EvalConfig
    comms: CommsConfig


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Deploy model for evaluation")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--orchestrate",
        action="store_true",
        help="Orchestrate evaluation (--batch mode). Without this flag, waits for external done signal.",
    )
    return parser.parse_args()


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file."""
    cfg = OmegaConf.load(config_path)

    # Extract deploy config
    deploy_cfg = cfg.get("deploy", {})
    model_cfg = cfg.get("model", {})

    deploy = DeployConfig(
        checkpoint_path=model_cfg.get("checkpoint_path"),
        host=deploy_cfg.get("host", "0.0.0.0"),
        port=deploy_cfg.get("port", 8000),
        num_gpus=deploy_cfg.get("num_gpus", 8),
        tensor_model_parallel_size=deploy_cfg.get("tensor_model_parallel_size", 4),
        expert_model_parallel_size=deploy_cfg.get("expert_model_parallel_size", 1),
        pipeline_model_parallel_size=deploy_cfg.get("pipeline_model_parallel_size", 1),
        context_parallel_size=deploy_cfg.get("context_parallel_size", 1),
        model_id=deploy_cfg.get("model_id", "nano3-eval"),
        model_type=model_cfg.get("model_type", "gpt"),
        model_format=deploy_cfg.get("model_format", "megatron"),
        max_batch_size=deploy_cfg.get("max_batch_size", 32),
        num_replicas=deploy_cfg.get("num_replicas", 1),
        num_cpus_per_replica=deploy_cfg.get("num_cpus_per_replica", 8),
    )

    # Extract eval config
    eval_cfg = cfg.get("eval", {})

    # Convert tasks to list of dicts
    tasks = []
    raw_tasks = eval_cfg.get("tasks", [])
    for task in raw_tasks:
        if isinstance(task, str):
            tasks.append({"name": task})
        else:
            tasks.append(OmegaConf.to_container(task, resolve=True))

    eval_config = EvalConfig(
        tasks=tasks,
        parallelism=eval_cfg.get("parallelism", 32),
        max_retries=eval_cfg.get("max_retries", 5),
        request_timeout=eval_cfg.get("request_timeout", 360),
        max_new_tokens=eval_cfg.get("max_new_tokens", 8192),
        limit_samples=eval_cfg.get("limit_samples"),
        output_dir=eval_cfg.get("output_dir", "/nemo_run/eval_results"),
        extra=OmegaConf.to_container(eval_cfg.get("extra", {}), resolve=True),
        slurm=OmegaConf.to_container(eval_cfg.get("slurm", {}), resolve=True),
    )

    # Extract comms config
    comms_cfg = cfg.get("comms", {})
    comms = CommsConfig(
        base_dir=comms_cfg.get("base_dir", "/nemo_run/eval_comms"),
        endpoint_file=comms_cfg.get("endpoint_file", "endpoint.json"),
        completion_file=comms_cfg.get("completion_file", "done"),
    )

    return Config(deploy=deploy, eval=eval_config, comms=comms)


def deploy_model(config: DeployConfig) -> tuple[Any, Any]:
    """Deploy model using Ray Serve.

    Returns:
        Tuple of (ray module, serve module) for cleanup.
    """
    try:
        import ray
        from ray import serve

        from nemo_deploy.llm.megatronllm_deployable_ray import MegatronRayDeployable
    except ImportError as e:
        print(f"Error: Required packages not available: {e}", file=sys.stderr)
        print("Ensure ray, ray[serve], and nemo-deploy are installed", file=sys.stderr)
        sys.exit(1)

    # Validate checkpoint path
    checkpoint_path = Path(config.checkpoint_path)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint path does not exist: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    # Calculate GPUs per replica
    gpus_per_replica = config.num_gpus // config.num_replicas

    print(f"Initializing Ray with {config.num_gpus} GPUs...")
    print(f"Configuration: {config.num_replicas} replicas, {gpus_per_replica} GPUs per replica")

    # Initialize Ray - connect to existing cluster started by RayJob
    ray.init(address="auto", ignore_reinit_error=True)

    # Start Ray Serve
    print(f"Starting Ray Serve on {config.host}:{config.port}...")
    serve.start(
        http_options={
            "host": config.host,
            "port": config.port,
        }
    )

    # Create the Megatron model deployment
    print(f"Deploying model from {config.checkpoint_path}...")
    app = MegatronRayDeployable.options(
        num_replicas=config.num_replicas,
        ray_actor_options={"num_cpus": config.num_cpus_per_replica},
    ).bind(
        nemo_checkpoint_filepath=None,  # Not using .nemo format
        megatron_checkpoint_filepath=config.checkpoint_path,
        num_gpus=gpus_per_replica,
        tensor_model_parallel_size=config.tensor_model_parallel_size,
        pipeline_model_parallel_size=config.pipeline_model_parallel_size,
        expert_model_parallel_size=config.expert_model_parallel_size,
        context_parallel_size=config.context_parallel_size,
        model_id=config.model_id,
        model_type=config.model_type,
        model_format=config.model_format,
        max_batch_size=config.max_batch_size,
    )

    # Deploy the model - this blocks until ready!
    serve.run(app, name=config.model_id)

    print(f"Model deployed successfully at http://{config.host}:{config.port}")
    print(f"Model ID: {config.model_id}")

    return ray, serve


def shutdown_deployment(ray_module: Any, serve_module: Any) -> None:
    """Gracefully shutdown Ray Serve and Ray."""
    print("Shutting down deployment...")

    try:
        print("Shutting down Ray Serve...")
        serve_module.shutdown()
    except Exception as e:
        print(f"Warning: Error during serve.shutdown(): {e}")

    try:
        print("Shutting down Ray...")
        ray_module.shutdown()
    except Exception as e:
        print(f"Warning: Error during ray.shutdown(): {e}")

    print("Shutdown complete")


def write_endpoint_file(comms: CommsConfig, endpoint_url: str, model_id: str) -> Path:
    """Write endpoint.json to comms directory for CLI/eval to discover."""
    comms_dir = Path(comms.base_dir)
    comms_dir.mkdir(parents=True, exist_ok=True)

    endpoint_path = comms_dir / comms.endpoint_file
    endpoint_data = {
        "url": endpoint_url,
        "model_id": model_id,
        "ready_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
    }

    endpoint_path.write_text(json.dumps(endpoint_data, indent=2))
    print(f"Endpoint written to: {endpoint_path}")
    return endpoint_path


def wait_for_done_signal(comms: CommsConfig, poll_interval: int = 5) -> None:
    """Wait for external done signal (--run mode)."""
    done_path = Path(comms.base_dir) / comms.completion_file
    print(f"Waiting for done signal at: {done_path}")

    while not done_path.exists():
        time.sleep(poll_interval)

    print("Done signal received")


def build_evaluator_config(
    eval_config: EvalConfig,
    endpoint_url: str,
    model_id: str,
) -> DictConfig:
    """Build nemo-evaluator-launcher compatible config.

    For --batch mode (orchestrate), uses local executor since we're already on the cluster.
    """
    evaluator_config = {
        "defaults": [
            {"execution": "local"},
            {"deployment": "none"},
            "_self_",
        ],
        "execution": {
            "output_dir": eval_config.output_dir,
        },
        "target": {
            "api_endpoint": {
                "url": endpoint_url,
                "model_id": model_id,
            }
        },
        "evaluation": {
            "nemo_evaluator_config": {
                "config": {
                    "params": {
                        "parallelism": eval_config.parallelism,
                        "max_retries": eval_config.max_retries,
                        "request_timeout": eval_config.request_timeout,
                        "max_new_tokens": eval_config.max_new_tokens,
                        "extra": eval_config.extra,
                    }
                }
            },
            "tasks": eval_config.tasks,
        },
    }

    # Add limit_samples if specified
    if eval_config.limit_samples is not None:
        evaluator_config["evaluation"]["nemo_evaluator_config"]["config"]["params"][
            "limit_samples"
        ] = eval_config.limit_samples

    return OmegaConf.create(evaluator_config)


def run_evaluation_orchestrated(
    eval_config: EvalConfig,
    endpoint_url: str,
    model_id: str,
) -> int:
    """Run evaluation in orchestrate mode (--batch).

    Calls nemo-evaluator-launcher with local executor and waits for completion.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    # Build evaluator config
    evaluator_cfg = build_evaluator_config(eval_config, endpoint_url, model_id)

    # Create output directory
    output_dir = Path(eval_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save evaluator config for debugging
    config_path = output_dir / "evaluator_config.yaml"
    OmegaConf.save(evaluator_cfg, config_path)
    print(f"Evaluator config saved to {config_path}")

    # Try to import and run nemo-evaluator-launcher
    try:
        from nemo_evaluator_launcher.api.functional import get_status, run_eval
        from nemo_evaluator_launcher.api.types import RunConfig

        # Convert to RunConfig type expected by run_eval
        run_config = RunConfig(evaluator_cfg)

        print("Starting evaluation (orchestrate mode)...")
        print(f"Tasks: {[t.get('name', t) for t in eval_config.tasks]}")
        print(f"Endpoint: {endpoint_url}")

        invocation_id = run_eval(run_config, dry_run=False)

        if invocation_id:
            print(f"Evaluation started. Invocation ID: {invocation_id}")

            # Poll for completion
            return _wait_for_eval_completion(invocation_id, get_status)
        else:
            # Local executor may return None but still complete synchronously
            print("Evaluation completed (synchronous execution)")
            return 0

    except ImportError as e:
        print(f"nemo-evaluator-launcher not available, falling back to CLI: {e}")
        return _run_evaluation_cli(evaluator_cfg, eval_config)


def _wait_for_eval_completion(
    invocation_id: str,
    get_status_fn: Any,
    poll_interval: int = 30,
    timeout: int = 14400,  # 4 hours
) -> int:
    """Poll for evaluation completion.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    start_time = time.time()
    terminal_states = {"success", "failed", "killed", "error"}

    while True:
        elapsed = time.time() - start_time
        if elapsed > timeout:
            print(f"Evaluation timeout after {timeout}s")
            return 1

        try:
            status_list = get_status_fn([invocation_id])
            if not status_list:
                print(f"No status found for invocation {invocation_id}")
                time.sleep(poll_interval)
                continue

            # Check if all jobs are complete
            all_complete = True
            any_failed = False

            for status in status_list:
                job_status = status.get("status", "unknown").lower()
                job_id = status.get("job_id", "unknown")
                progress = status.get("progress", {})

                print(f"  Job {job_id}: {job_status} (progress: {progress})")

                if job_status not in terminal_states:
                    all_complete = False
                elif job_status in {"failed", "killed", "error"}:
                    any_failed = True

            if all_complete:
                if any_failed:
                    print("Evaluation completed with failures")
                    return 1
                else:
                    print("Evaluation completed successfully")
                    return 0

        except Exception as e:
            print(f"Error checking status: {e}")

        time.sleep(poll_interval)


def _run_evaluation_cli(evaluator_config: DictConfig, eval_config: EvalConfig) -> int:
    """Run evaluation via CLI subprocess.

    Fallback if nemo-evaluator-launcher is not importable.
    """
    import subprocess
    import tempfile

    # Save config to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        OmegaConf.save(evaluator_config, f)
        temp_config_path = f.name

    try:
        # Run nemo-evaluator-launcher CLI
        cmd = [
            "nemo-evaluator-launcher",
            "run",
            "--config",
            temp_config_path,
        ]

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=False, text=True)
        return result.returncode

    finally:
        # Clean up temp file
        Path(temp_config_path).unlink(missing_ok=True)


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Load configuration
    print(f"Loading config from {args.config}")
    config = load_config(args.config)

    # Deploy model
    ray_module, serve_module = deploy_model(config.deploy)

    # Build endpoint URL using compute node hostname (accessible from eval job)
    hostname = socket.gethostname()
    endpoint_url = f"http://{hostname}:{config.deploy.port}/v1"

    # Write endpoint file for discovery
    write_endpoint_file(config.comms, endpoint_url, config.deploy.model_id)

    exit_code = 0
    try:
        if args.orchestrate:
            # --batch mode: orchestrate evaluation ourselves
            print("Running in orchestrate mode (--batch)")
            exit_code = run_evaluation_orchestrated(
                config.eval,
                endpoint_url,
                config.deploy.model_id,
            )
        else:
            # --run mode: wait for external done signal
            print("Running in wait mode (--run)")
            wait_for_done_signal(config.comms)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        exit_code = 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        exit_code = 1
    finally:
        # Always shutdown deployment
        shutdown_deployment(ray_module, serve_module)

    print(f"Deploy exiting with code {exit_code}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
