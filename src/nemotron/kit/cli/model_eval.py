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

"""@model_eval decorator for defining model evaluation CLI commands.

The decorator standardizes the deploy+eval pattern for model evaluation,
handling config loading, executor setup, and RayJob orchestration via nemo-run.

Supports two execution modes:
- --run (attached): CLI orchestrates evaluation, polls for endpoint, runs
  nemo-evaluator-launcher, writes done signal, follows logs
- --batch (detached): RayJob orchestrates everything, CLI returns immediately

Example:
    @model_eval(
        name="nano3/eval",
        config_dir="src/nemotron/recipes/nano3/eval/config",
        script="src/nemotron/recipes/nano3/eval/deploy.py",
    )
    def eval_cmd(ctx: typer.Context, model_path: str) -> None:
        '''Evaluate nano3 model.'''
        ...
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any

import typer
from omegaconf import OmegaConf
from rich.console import Console

from nemotron.kit.cli.config import apply_dotlist_overrides, find_config_file, load_config
from nemotron.kit.cli.display import display_job_config
from nemotron.kit.cli.globals import GlobalContext, split_unknown_args

console = Console()

# Polling intervals (seconds)
ENDPOINT_POLL_INTERVAL = 5
EVAL_STATUS_POLL_INTERVAL = 30


@dataclass
class ModelEvalMetadata:
    """Metadata attached to a model_eval command function.

    Attributes:
        name: Eval identifier (e.g., "nano3/eval")
        config_dir: Path to config directory relative to repo root
        script: Path to deploy script relative to repo root
        default_config: Default config name (stem)
        model_arg: Name of the model path argument
    """

    name: str
    config_dir: str
    script: str
    default_config: str = "default"
    model_arg: str = "model_path"


def model_eval(
    name: str,
    config_dir: str,
    script: str,
    default_config: str = "default",
    model_arg: str = "model_path",
) -> Callable:
    """Decorator marking a function as a model evaluation command.

    The decorated function becomes a typer command that:
    1. Loads and merges configuration
    2. Resolves env profile (if --run/--batch)
    3. Saves eval config
    4. Executes deploy+eval via RayJob (like data_prep stages)

    Args:
        name: Eval identifier (e.g., "nano3/eval")
        config_dir: Path to config directory relative to repo root
        script: Path to combined deploy+eval script relative to repo root
        default_config: Default config name (default: "default")
        model_arg: Name of the model path argument in decorated function (default: "model_path")

    Example:
        @model_eval(
            name="nano3/eval",
            config_dir="src/nemotron/recipes/nano3/eval/config",
            script="src/nemotron/recipes/nano3/eval/deploy_eval.py",
        )
        def eval_cmd(ctx: typer.Context, model_path: str) -> None:
            '''Evaluate model performance.'''
            ...
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(ctx: typer.Context) -> None:
            # Get global context
            global_ctx: GlobalContext = ctx.obj
            if global_ctx is None:
                global_ctx = GlobalContext()

            # Split unknown args into dotlist, passthrough, and extract model_path
            # Model path is the first non-option, non-dotlist argument
            args = ctx.args or []
            model_path, remaining_args = _extract_model_path(args)
            dotlist, passthrough, global_ctx = split_unknown_args(remaining_args, global_ctx)
            global_ctx.dotlist = dotlist
            global_ctx.passthrough = passthrough

            # Validate options
            if global_ctx.run and global_ctx.batch:
                typer.echo("Error: --run and --batch cannot both be set", err=True)
                raise typer.Exit(1)

            # Require --run or --batch (model eval only works on SLURM)
            if global_ctx.mode == "local" and not global_ctx.dry_run:
                typer.echo(
                    "Error: Model evaluation requires SLURM execution.\n"
                    "Use --run <profile> or --batch <profile> to specify a cluster.\n"
                    "Use --dry-run to preview the configuration without execution.",
                    err=True,
                )
                raise typer.Exit(1)

            # Find repo root
            repo_root = _find_repo_root()
            if repo_root is None:
                typer.echo(
                    "Error: Could not find repository root (looking for pyproject.toml)",
                    err=True,
                )
                raise typer.Exit(1)

            # Load config
            config_dir_path = repo_root / config_dir
            config_name = global_ctx.config or default_config

            try:
                config_path = find_config_file(config_name, config_dir_path)
            except FileNotFoundError as e:
                typer.echo(f"Error: {e}", err=True)
                raise typer.Exit(1)

            console.print(f"[dim]Loading config: {config_path}[/dim]")
            config = load_config(config_path)

            # Apply dotlist overrides
            config = apply_dotlist_overrides(config, dotlist)

            # Determine the model reference (from CLI argument or config)
            # Config may have run.model set (e.g., stage-specific configs like sft.yaml)
            from nemotron.kit.resolvers import _is_artifact_reference

            config_has_model = (
                "run" in config
                and "model" in config.run
                and config.run.model is not None
                and str(config.run.model) != "???"
            )

            if model_path:
                # CLI argument takes precedence
                if "run" not in config:
                    config["run"] = {}
                config.run.model = model_path
                effective_model = model_path
            elif config_has_model:
                # Use model from config (e.g., ModelArtifact-sft:latest)
                effective_model = str(config.run.model)
            else:
                typer.echo(
                    "Error: No model specified. Either:\n"
                    "  - Provide a model path: nemotron nano3 model eval /path/to/checkpoint --run <profile>\n"
                    "  - Use a stage config: nemotron nano3 model eval -c sft --run <profile>\n"
                    "  - Set via dotlist: nemotron nano3 model eval --run <profile> run.model=sft:latest",
                    err=True,
                )
                raise typer.Exit(1)

            # If it's a direct path (not an artifact reference), validate it exists
            # Artifact references (e.g., "sft:latest") are resolved at runtime
            if not _is_artifact_reference(effective_model):
                checkpoint = Path(effective_model)
                if not checkpoint.exists():
                    typer.echo(f"Error: Checkpoint path does not exist: {effective_model}", err=True)
                    raise typer.Exit(1)
                # For direct paths, also set model.checkpoint_path directly
                # This provides backwards compatibility and works without artifact resolution
                if "model" not in config:
                    config["model"] = {}
                config.model.checkpoint_path = effective_model

            # Build job config with execution metadata
            job_config = _build_job_config(config, global_ctx, name)

            # Display config
            display_job_config(job_config, for_remote=(global_ctx.mode in ("run", "batch")))

            # Handle dry-run mode
            if global_ctx.dry_run:
                return

            # Save config to job directory
            job_dir = _create_job_dir(name)
            saved_config_path = job_dir / "eval_config.yaml"
            OmegaConf.save(job_config, saved_config_path)
            console.print(f"[dim]Config saved to: {saved_config_path}[/dim]")

            # Execute via RayJob (like data_prep stages)
            _execute_ray_job(
                config_path=saved_config_path,
                job_config=job_config,
                script=script,
                name=name,
                attached=(global_ctx.mode == "run"),
            )

        # Attach metadata to function for introspection
        wrapper._model_eval_metadata = ModelEvalMetadata(
            name=name,
            config_dir=config_dir,
            script=script,
            default_config=default_config,
            model_arg=model_arg,
        )

        return wrapper

    return decorator


def _extract_model_path(args: list[str]) -> tuple[str | None, list[str]]:
    """Extract model path from args list.

    Model path is the first argument that:
    - Doesn't start with '-' (not an option)
    - Doesn't contain '=' (not a dotlist override)
    - Looks like a path or artifact reference

    Args:
        args: List of arguments from ctx.args

    Returns:
        Tuple of (model_path or None, remaining_args)
    """
    # Known options that take a value (skip these and their values)
    options_with_values = {"-c", "--config", "-r", "--run", "-b", "--batch"}

    remaining = []
    model_path = None
    skip_next = False

    for i, arg in enumerate(args):
        if skip_next:
            remaining.append(arg)
            skip_next = False
            continue

        if arg in options_with_values:
            remaining.append(arg)
            skip_next = True
            continue

        if arg.startswith("-"):
            # Option flag (like --dry-run)
            remaining.append(arg)
            continue

        if "=" in arg:
            # Dotlist override
            remaining.append(arg)
            continue

        # First non-option, non-dotlist argument is the model path
        if model_path is None:
            model_path = arg
        else:
            remaining.append(arg)

    return model_path, remaining


def _find_repo_root() -> Path | None:
    """Find the repository root by looking for pyproject.toml."""
    current = Path.cwd()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return None


def _create_job_dir(name: str) -> Path:
    """Create a unique job directory for this evaluation run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = name.replace("/", "-")
    job_dir = Path.cwd() / ".jobs" / f"{safe_name}_{timestamp}"
    job_dir.mkdir(parents=True, exist_ok=True)
    return job_dir


def _build_job_config(
    config: OmegaConf,
    ctx: GlobalContext,
    name: str,
) -> OmegaConf:
    """Build the full job config with execution metadata."""
    from nemotron.kit.cli.env import load_env_profile

    # Start with the eval config
    job_config = OmegaConf.create(OmegaConf.to_container(config, resolve=False))

    # Build run section
    run_updates = {
        "mode": ctx.mode,
        "profile": ctx.profile,
        "env": {},
        "cli": {
            "argv": sys.argv,
            "dotlist": ctx.dotlist,
            "config": ctx.config,
        },
        "recipe": {
            "name": name,
        },
    }

    # Get existing run.env from config (if any)
    existing_env = {}
    if "run" in job_config and "env" in job_config.run:
        existing_env = OmegaConf.to_container(job_config.run.env, resolve=False)

    # Merge env profile if we have one
    if ctx.profile:
        env_config = load_env_profile(ctx.profile)
        profile_env = OmegaConf.to_container(env_config, resolve=True)
        run_updates["env"] = {**existing_env, **profile_env}
    elif existing_env:
        run_updates["env"] = existing_env

    # Ensure run section exists and merge updates
    if "run" not in job_config:
        job_config["run"] = {}
    job_config.run = OmegaConf.merge(job_config.run, OmegaConf.create(run_updates))

    return job_config


def _execute_ray_job(
    config_path: Path,
    job_config: OmegaConf,
    script: str,
    name: str,
    attached: bool,
) -> None:
    """Execute evaluation via RayJob.

    Architecture:
    - RayJob (via nemo-run) submits a SLURM job that runs the deploy script
    - Deploy script runs inside a container on compute nodes (Ray Serve)
    - Evaluation is a SEPARATE SLURM job submitted via nemo-evaluator-launcher
    - nemo-evaluator-launcher runs locally/login-node and SSHs to submit eval jobs

    In attached mode (--run): CLI orchestrates evaluation from local machine:
    1. Start RayJob (deploy only, waits for done signal)
    2. Poll for endpoint.json via SSH (written by deploy to shared filesystem)
    3. Run evaluation via uvx nemo-evaluator-launcher (submits separate SLURM job)
    4. Poll evaluation status via uvx nemo-evaluator-launcher status
    5. Write done signal via SSH (tells deploy to shutdown)
    6. Follow RayJob logs until completion

    In detached mode (--batch): NOT YET IMPLEMENTED
    Would require orchestration from login node (not compute nodes) since
    nemo-evaluator-launcher needs SSH access to submit SLURM jobs.
    """
    try:
        from nemo_run.run.ray.job import RayJob
    except ImportError:
        typer.echo("Error: nemo-run is required for --run/--batch execution", err=True)
        typer.echo("Install with: pip install nemo-run", err=True)
        raise typer.Exit(1)

    # Extract env config for building executor
    env_config = {}
    if hasattr(job_config, "run") and hasattr(job_config.run, "env"):
        env_config = OmegaConf.to_container(job_config.run.env, resolve=True)

    # Build environment variables
    env_vars = _build_env_vars(job_config, env_config)

    # Build executor (GPU job for deploy+eval)
    executor = _build_executor(env_config, env_vars)

    # Generate unique job name to prevent directory collisions
    job_name = f"{name.replace('/', '-')}_{int(time.time())}"
    ray_job = RayJob(name=job_name, executor=executor)

    # Get comms directory from config (for coordination files)
    # Comms is now under run.comms
    remote_job_dir = env_config.get("remote_job_dir", "/nemo_run")
    run_comms = job_config.get("run", {}).get("comms", {})
    comms_base = run_comms.get("base_dir") if run_comms else None
    if not comms_base:
        # Default comms dir based on job name
        comms_base = f"{remote_job_dir}/eval_comms/{job_name}"

    # Update config with comms directory
    if "comms" not in job_config.run:
        job_config.run["comms"] = {}
    job_config.run.comms.base_dir = comms_base

    # Copy config to repo root so it gets rsynced to the remote
    repo_config = Path.cwd() / "config.yaml"
    OmegaConf.save(job_config, repo_config)

    # Setup commands to prepare the environment before running
    setup_commands = [
        "find . -type d -name __pycache__ -delete 2>/dev/null || true",
        "uv sync --reinstall-package nemotron",
    ]

    # Build the command to run
    if attached:
        # --run mode: deploy only, wait for done signal
        cmd = f"uv run python {script} --config config.yaml"
    else:
        # --batch mode: deploy + orchestrate eval
        cmd = f"uv run python {script} --config config.yaml --orchestrate"

    # Build runtime_env with environment variables for Ray workers
    import yaml

    runtime_env: dict = {"env_vars": dict(env_vars)}

    # Auto-detect HuggingFace token for Ray workers
    try:
        from huggingface_hub import HfFolder

        hf_token = HfFolder.get_token()
        if hf_token:
            runtime_env["env_vars"]["HF_TOKEN"] = hf_token
    except Exception:
        pass

    # Auto-detect Weights & Biases API key for Ray workers
    try:
        import wandb

        wandb_api_key = wandb.api.api_key
        if wandb_api_key:
            runtime_env["env_vars"]["WANDB_API_KEY"] = wandb_api_key
    except Exception:
        pass

    # Create temporary runtime_env YAML file
    runtime_env_yaml = None
    if runtime_env["env_vars"]:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(runtime_env, f)
            runtime_env_yaml = f.name

    console.print(f"[bold]Starting RayJob evaluation: {job_name}[/bold]")

    ray_job.start(
        command=cmd,
        # Pass empty workdir to use CodePackager (respects .gitignore)
        workdir="",
        pre_ray_start_commands=setup_commands,
        runtime_env_yaml=runtime_env_yaml,
    )

    # Workaround for nemo-run bug: when reusing an existing cluster,
    # SlurmRayCluster.create() returns None instead of the job_id.
    if ray_job.backend.job_id is None:
        status = ray_job.backend.status(display=False)
        if status and status.get("job_id"):
            ray_job.backend.job_id = status["job_id"]
            typer.echo(f"[info] Recovered job_id {status['job_id']} from cluster status")

    if attached:
        # --run mode: CLI orchestrates evaluation
        _run_attached_mode(ray_job, job_config, env_config, comms_base)
    else:
        # --batch mode: return immediately
        console.print(f"[bold green]Job submitted: {job_name}[/bold green]")
        console.print(f"Check SLURM job status: squeue -u $USER")
        output_dir = job_config.get("eval", {}).get("output_dir", "/nemo_run/eval_results")
        console.print(f"Results will be saved to: {output_dir}")


def _build_env_vars(job_config: OmegaConf, env_config: dict) -> dict[str, str]:
    """Build environment variables for execution."""
    env_vars = {}

    # Add wandb config if available
    if hasattr(job_config, "run") and hasattr(job_config.run, "wandb"):
        wandb_config = job_config.run.wandb
        if hasattr(wandb_config, "project"):
            env_vars["WANDB_PROJECT"] = wandb_config.project
        if hasattr(wandb_config, "entity"):
            env_vars["WANDB_ENTITY"] = wandb_config.entity

    # Pass through common env vars
    for key in ["WANDB_API_KEY", "HF_TOKEN", "HF_HOME", "NGC_API_KEY"]:
        if key in os.environ:
            env_vars[key] = os.environ[key]

    return env_vars


def _build_executor(env_config: dict, env_vars: dict[str, str]) -> Any:
    """Build nemo-run executor from env config."""
    import nemo_run as run

    executor_type = env_config.get("executor", "local")

    if executor_type == "local":
        return run.LocalExecutor(
            ntasks_per_node=1,
            env_vars=env_vars,
        )

    elif executor_type == "slurm":
        # Build tunnel if configured
        tunnel = None
        remote_job_dir = env_config.get("remote_job_dir")
        if env_config.get("tunnel") == "ssh":
            tunnel = run.SSHTunnel(
                host=env_config.get("host", "localhost"),
                user=env_config.get("user"),
                job_dir=remote_job_dir,
            )

        # Container image
        container_image = env_config.get("container_image") or env_config.get("container")

        # Partition (use run_partition if available for interactive)
        partition = env_config.get("run_partition") or env_config.get("partition", "interactive")

        # GPU job for deploy+eval
        return run.SlurmExecutor(
            account=env_config.get("account"),
            partition=partition,
            nodes=env_config.get("nodes", 1),
            ntasks_per_node=env_config.get("ntasks_per_node", 1),
            gpus_per_node=env_config.get("gpus_per_node", 8),
            time=env_config.get("time", "04:00:00"),
            container_image=container_image,
            tunnel=tunnel,
            env_vars=env_vars,
        )

    else:
        typer.echo(f"Error: Unsupported executor type: {executor_type}", err=True)
        raise typer.Exit(1)


def _run_attached_mode(
    ray_job: Any,
    job_config: OmegaConf,
    env_config: dict,
    comms_base: str,
) -> None:
    """Run evaluation in attached mode (--run).

    CLI orchestrates the evaluation:
    1. Poll for endpoint.json via SSH
    2. Run evaluation via nemo-evaluator-launcher
    3. Write done signal via SSH
    4. Follow RayJob logs
    """
    try:
        # Get SSH connection details
        host = env_config.get("host")
        user = env_config.get("user")

        if not host or not user:
            typer.echo("Error: SSH host and user required for --run mode", err=True)
            raise typer.Exit(1)

        # 1. Wait for endpoint.json
        console.print("[dim]Waiting for model deployment to be ready...[/dim]")
        endpoint_file = f"{comms_base}/endpoint.json"
        endpoint_data = _wait_for_endpoint(host, user, endpoint_file, timeout=600)

        endpoint_url = endpoint_data["url"]
        model_id = endpoint_data.get("model_id", "nano3-eval")
        console.print(f"[green]Model deployed at: {endpoint_url}[/green]")

        # 2. Run evaluation via nemo-evaluator-launcher
        console.print("[dim]Starting evaluation via nemo-evaluator-launcher...[/dim]")
        invocation_id = _run_evaluation(job_config, endpoint_url, model_id, env_config)

        if invocation_id:
            console.print(f"[dim]Evaluation invocation: {invocation_id}[/dim]")
            # 3. Wait for evaluation completion
            _wait_for_eval_completion(invocation_id)
        else:
            console.print("[dim]Evaluation completed (synchronous)[/dim]")

        # 4. Write done signal
        done_file = f"{comms_base}/done"
        _write_done_signal(host, user, done_file)
        console.print("[dim]Done signal sent to deployment[/dim]")

        # 5. Follow RayJob logs until complete
        console.print("[dim]Following deployment shutdown...[/dim]")
        ray_job.logs(follow=True, timeout=120)

    except KeyboardInterrupt:
        typer.echo("\n[info] Ctrl-C detected, cleaning up...")
        try:
            # Write done signal to trigger deployment shutdown
            done_file = f"{comms_base}/done"
            _write_done_signal(
                env_config.get("host", ""),
                env_config.get("user", ""),
                done_file,
            )
        except Exception:
            pass

        try:
            ray_job.stop()
            typer.echo("[info] Ray cluster stopped")
        except Exception as e:
            typer.echo(f"[warning] Failed to stop Ray cluster: {e}")

        raise typer.Exit(130)


def _wait_for_endpoint(
    host: str,
    user: str,
    endpoint_file: str,
    timeout: int = 600,
) -> dict:
    """Poll for endpoint.json via SSH.

    Returns:
        Endpoint data dict with url, model_id, etc.
    """
    import subprocess

    start_time = time.time()

    while True:
        elapsed = time.time() - start_time
        if elapsed > timeout:
            typer.echo(f"Error: Timeout waiting for endpoint after {timeout}s", err=True)
            raise typer.Exit(1)

        # Check if file exists via SSH
        cmd = ["ssh", f"{user}@{host}", f"cat {endpoint_file} 2>/dev/null"]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0 and result.stdout.strip():
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError:
                pass  # File exists but not valid JSON yet

        time.sleep(ENDPOINT_POLL_INTERVAL)


def _run_evaluation(
    job_config: OmegaConf,
    endpoint_url: str,
    model_id: str,
    env_config: dict,
) -> str | None:
    """Run evaluation via nemo-evaluator-launcher.

    Uses the nemo-evaluator-launcher package to submit SLURM evaluation jobs.
    The launcher's SLURM executor submits eval jobs to the cluster via SSH.

    Configuration is read from the eval section which uses OmegaConf interpolation:
    - eval.execution: SLURM settings (references ${run.env.*})
    - eval.export.wandb: W&B export settings (references ${run.wandb.*})
    - eval.tasks, eval.parallelism, etc.: Evaluation parameters

    Returns:
        Invocation ID if async, None if synchronous.
    """
    eval_cfg = job_config.get("eval", {})

    # Get execution config from eval.execution (already resolved via OmegaConf interpolation)
    execution_cfg = eval_cfg.get("execution", {})
    execution_config = OmegaConf.to_container(execution_cfg, resolve=True) if execution_cfg else {}

    # Ensure required fields are present (fallback to env_config for backwards compatibility)
    if not execution_config.get("hostname"):
        execution_config["hostname"] = env_config.get("host")
    if not execution_config.get("username"):
        execution_config["username"] = env_config.get("user")
    if not execution_config.get("account"):
        execution_config["account"] = env_config.get("account")
    if not execution_config.get("partition"):
        execution_config["partition"] = env_config.get("partition", "batch")
    if not execution_config.get("output_dir"):
        remote_job_dir = env_config.get("remote_job_dir", "/nemo_run")
        execution_config["output_dir"] = f"{remote_job_dir}/eval_results"

    # Build evaluator config
    evaluator_config = {
        "defaults": [
            {"execution": "slurm/default"},
            {"deployment": "none"},
            "_self_",
        ],
        "execution": execution_config,
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
                        "parallelism": eval_cfg.get("parallelism", 32),
                        "max_retries": eval_cfg.get("max_retries", 5),
                        "request_timeout": eval_cfg.get("request_timeout", 360),
                        "max_new_tokens": eval_cfg.get("max_new_tokens", 8192),
                    }
                }
            },
            "tasks": OmegaConf.to_container(eval_cfg.get("tasks", []), resolve=True),
        },
    }

    # Get wandb export config from eval.export.wandb (already resolved via OmegaConf interpolation)
    export_cfg = eval_cfg.get("export", {})
    wandb_export_cfg = export_cfg.get("wandb", {}) if export_cfg else {}

    if wandb_export_cfg:
        wandb_config = OmegaConf.to_container(wandb_export_cfg, resolve=True)

        if wandb_config.get("project"):
            # Enable auto_export to wandb
            evaluator_config["execution"]["auto_export"] = {"destinations": ["wandb"]}

            # Add WANDB_API_KEY to env_vars for export
            try:
                import wandb

                wandb_api_key = wandb.api.api_key
                if wandb_api_key:
                    if "env_vars" not in evaluator_config["execution"]:
                        evaluator_config["execution"]["env_vars"] = {}
                    evaluator_config["execution"]["env_vars"]["export"] = {
                        "WANDB_API_KEY": "WANDB_API_KEY"  # Uses env var passthrough
                    }
            except Exception:
                pass

            # Configure wandb export settings
            export_config = {
                "entity": wandb_config.get("entity"),
                "project": wandb_config.get("project"),
                "log_mode": "multi_task",  # Log all tasks in single run
                "log_artifacts": True,
                "log_logs": True,
            }

            # Add optional tags from config
            tags = wandb_config.get("tags", [])
            if tags:
                export_config["tags"] = list(tags) if not isinstance(tags, list) else tags

            # Add eval-specific tags
            eval_tags = ["eval", "nemotron"]
            if "tags" in export_config:
                export_config["tags"].extend(eval_tags)
            else:
                export_config["tags"] = eval_tags

            evaluator_config["export"] = {"wandb": export_config}
            console.print(f"[dim]W&B export enabled: {wandb_config.get('project')}[/dim]")

    # Add limit_samples if specified
    limit_samples = eval_cfg.get("limit_samples")
    if limit_samples is not None:
        evaluator_config["evaluation"]["nemo_evaluator_config"]["config"]["params"][
            "limit_samples"
        ] = limit_samples

    # Add extra params if specified
    extra = eval_cfg.get("extra", {})
    if extra:
        evaluator_config["evaluation"]["nemo_evaluator_config"]["config"]["params"][
            "extra"
        ] = OmegaConf.to_container(extra, resolve=True)

    task_names = [t.get("name", t) if isinstance(t, dict) else t for t in eval_cfg.get("tasks", [])]
    console.print(f"[dim]Running tasks: {task_names}[/dim]")

    try:
        from nemo_evaluator_launcher.api.functional import run_eval
        from nemo_evaluator_launcher.api.types import RunConfig
    except ImportError:
        typer.echo(
            "Error: nemo-evaluator-launcher not installed.\n"
            "Install with: pip install nemotron[eval]",
            err=True,
        )
        raise typer.Exit(1)

    try:
        # Create RunConfig from our evaluator config dict
        run_config = RunConfig(OmegaConf.create(evaluator_config))

        console.print("[dim]Submitting evaluation via nemo-evaluator-launcher...[/dim]")
        invocation_id = run_eval(run_config, dry_run=False)

        if invocation_id:
            console.print(f"[dim]Evaluation submitted: {invocation_id}[/dim]")

        return invocation_id

    except Exception as e:
        typer.echo(f"Error running nemo-evaluator-launcher: {e}", err=True)
        raise typer.Exit(1)


def _wait_for_eval_completion(invocation_id: str | None, timeout: int = 14400) -> None:
    """Poll for evaluation completion using nemo-evaluator-launcher API.

    Args:
        invocation_id: Invocation ID from run_eval (if None, skips polling)
        timeout: Maximum time to wait in seconds (default 4 hours)
    """
    if not invocation_id:
        console.print("[dim]No invocation ID, skipping status polling[/dim]")
        return

    try:
        from nemo_evaluator_launcher.api.functional import get_status
    except ImportError:
        console.print("[dim]nemo-evaluator-launcher not available for status polling[/dim]")
        return

    console.print(f"[dim]Polling evaluation status for: {invocation_id}[/dim]")

    terminal_states = {"success", "completed", "failed", "killed", "error", "cancelled"}
    start_time = time.time()

    while True:
        elapsed = time.time() - start_time
        if elapsed > timeout:
            typer.echo(f"Warning: Evaluation timeout after {timeout}s", err=True)
            return

        try:
            status_list = get_status([invocation_id])
            if not status_list:
                time.sleep(EVAL_STATUS_POLL_INTERVAL)
                continue

            all_complete = True
            any_failed = False

            for status in status_list:
                job_status = str(status.get("status", "unknown")).lower()
                job_id = status.get("job_id", "unknown")
                progress = status.get("progress", {})

                console.print(f"[dim]  Job {job_id}: {job_status} ({progress})[/dim]")

                if job_status not in terminal_states:
                    all_complete = False
                elif job_status in {"failed", "killed", "error", "cancelled"}:
                    any_failed = True

            if all_complete:
                if any_failed:
                    console.print("[yellow]Evaluation completed with failures[/yellow]")
                else:
                    console.print("[green]Evaluation completed successfully[/green]")
                return

        except Exception as e:
            console.print(f"[dim]Error checking status: {e}[/dim]")

        time.sleep(EVAL_STATUS_POLL_INTERVAL)


def _write_done_signal(host: str, user: str, done_file: str) -> None:
    """Write done signal via SSH."""
    import subprocess

    cmd = ["ssh", f"{user}@{host}", f"touch {done_file}"]
    subprocess.run(cmd, capture_output=True)
