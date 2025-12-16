# Model Evaluation

Evaluate trained models against standard benchmarks.

## Powered By

This evaluation pipeline leverages two NVIDIA NeMo ecosystem projects:

| Project | Purpose | Repository |
|---------|---------|------------|
| **Export-Deploy** | High-performance model deployment with Ray Serve and vLLM | [NVIDIA-NeMo/Export-Deploy](https://github.com/NVIDIA-NeMo/Export-Deploy) |
| **Evaluator** | Benchmark orchestration with nemo-evaluator-launcher | [NVIDIA-NeMo/Evaluator](https://github.com/NVIDIA-NeMo/Evaluator) |

Export-Deploy handles loading Megatron checkpoints and serving them via an OpenAI-compatible API.
Evaluator runs standard benchmarks (ARC, Winogrande, HellaSwag, etc.) against the deployed endpoint.

## Overview

This command orchestrates model deployment and evaluation as a single RayJob:

1. **Deploy**: Starts a Ray Serve inference server using [Export-Deploy](https://github.com/NVIDIA-NeMo/Export-Deploy)
2. **Eval**: Runs benchmarks using [nemo-evaluator-launcher](https://github.com/NVIDIA-NeMo/Evaluator)
3. **Shutdown**: Gracefully shuts down the deployment

| Component | Description |
|-----------|-------------|
| `deploy.py` | Deployment script using Export-Deploy's Ray Serve integration |
| `config/` | Configuration files for deployment and evaluation settings |

## Requirements

- SLURM cluster with GPU nodes
- Model checkpoint in Megatron format
- Configured env profile (e.g., in `env.toml`)

## Quick Start

### Using nemotron CLI

Each stage config (pretrain, sft, rl) has the model artifact pre-configured:

```bash
# Evaluate pretrained checkpoint (uses ModelArtifact-pretrain:latest)
nemotron nano3 model eval -c pretrain --run YOUR-CLUSTER

# Evaluate SFT checkpoint (uses ModelArtifact-sft:latest)
nemotron nano3 model eval -c sft --run YOUR-CLUSTER

# Evaluate RL-aligned checkpoint (uses ModelArtifact-rl:latest)
nemotron nano3 model eval -c rl --run YOUR-CLUSTER

# Quick test with tiny config (single benchmark, limited samples)
nemotron nano3 model eval -c sft -c tiny --run YOUR-CLUSTER

# Preview configuration without execution
nemotron nano3 model eval -c pretrain --run YOUR-CLUSTER --dry-run

# Detached execution (returns immediately)
nemotron nano3 model eval -c sft --batch YOUR-CLUSTER
```

### Override Settings

Use dotlist overrides to customize evaluation:

```bash
# Override model artifact version
nemotron nano3 model eval -c sft --run YOUR-CLUSTER \
    run.model=ModelArtifact-sft:v5

# Increase parallelism
nemotron nano3 model eval -c pretrain --run YOUR-CLUSTER \
    eval.parallelism=64

# Change deployment settings
nemotron nano3 model eval -c sft --run YOUR-CLUSTER \
    deploy.num_gpus=4 \
    deploy.tensor_model_parallel_size=2

# Limit samples for quick testing
nemotron nano3 model eval -c pretrain --run YOUR-CLUSTER \
    eval.limit_samples=100
```

## Configuration

### Config Files

| File | Description |
|------|-------------|
| `config/pretrain.yaml` | Pretrain stage evaluation (ModelArtifact-pretrain:latest) |
| `config/sft.yaml` | SFT stage evaluation (ModelArtifact-sft:latest) |
| `config/rl.yaml` | RL stage evaluation (ModelArtifact-rl:latest) |
| `config/default.yaml` | Base config (requires model artifact override) |
| `config/tiny.yaml` | Quick test with single benchmark, 100 samples |

### Configuration Structure

```yaml
# Runtime settings (merged with env.toml profile)
run:
  model: ModelArtifact-sft:latest  # Model artifact to evaluate
  env:
    container: nvcr.io/nvidian/nemo:25.11-nano-v3.rc2
  comms:
    endpoint_file: endpoint.json
    completion_file: done

# Model loading configuration
model:
  checkpoint_path: ${art:model,path}  # Resolved from run.model artifact
  model_type: gpt

# Export-Deploy configuration (Ray Serve)
deploy:
  host: "0.0.0.0"
  port: 8000
  num_gpus: 8              # Total GPUs for deployment
  tensor_model_parallel_size: 4
  expert_model_parallel_size: 1
  pipeline_model_parallel_size: 1

# Evaluator configuration (nemo-evaluator-launcher)
eval:
  execution:               # SLURM settings (from env.toml profile)
    hostname: ${run.env.host}
    username: ${run.env.user}
    account: ${run.env.account}
    partition: ${run.env.partition}
    output_dir: ${run.env.remote_job_dir}/eval_results
  export:                  # Results export (W&B)
    wandb:
      entity: ${run.wandb.entity}
      project: ${run.wandb.project}
  tasks:                   # Benchmarks to run
    - name: adlr_arc_challenge_llama_25_shot
    - name: adlr_winogrande_5_shot
    - name: hellaswag
    - name: piqa
  parallelism: 32          # Concurrent evaluation requests
  max_retries: 5
  request_timeout: 360
```

### Available Benchmarks

The following benchmarks are available via nemo-evaluator-launcher:

| Benchmark | Description |
|-----------|-------------|
| `adlr_arc_challenge_llama_25_shot` | ARC Challenge with 25-shot |
| `adlr_winogrande_5_shot` | Winogrande with 5-shot |
| `hellaswag` | HellaSwag commonsense reasoning |
| `piqa` | Physical Intuition QA |
| `adlr_mmlu_pro_5_shot_base` | MMLU-Pro with 5-shot |
| `adlr_humaneval_greedy` | HumanEval code generation |

See nemo-evaluator-launcher documentation for the full list of available benchmarks.

## Architecture

### Execution Flow

```
User: nemotron nano3 model eval /path/to/ckpt --run cluster

1. CLI parses args, loads config
2. Creates RayJob with GPU executor
3. RayJob submitted to SLURM

On Ray Cluster:
├─ ray.init() - Connect to Ray cluster
├─ serve.start() - Start Ray Serve HTTP server
├─ MegatronRayDeployable.bind() - Create deployment
├─ serve.run() - Deploy model (blocks until ready)
├─ Run evaluation against localhost:8000
├─ Save results to output_dir
├─ serve.shutdown() - Graceful shutdown
└─ ray.shutdown()

4. Job completes
5. Results in eval.output_dir
```

### Benefits of RayJob Architecture

- **Single cluster**: Deploy and eval run in the same Ray cluster
- **No polling**: `serve.run()` blocks until deployment is ready
- **Simpler orchestration**: One job instead of two coordinated jobs
- **Consistent with data_prep**: Uses same RayJob pattern

## Troubleshooting

### Evaluation Timeout

If individual benchmark requests timeout:

```yaml
eval:
  request_timeout: 600  # Increase from default 360 seconds
  max_retries: 10       # Increase retry attempts
```

### GPU Configuration

Ensure `deploy.num_gpus` matches your SLURM allocation and parallelism settings:

```yaml
deploy:
  num_gpus: 8
  tensor_model_parallel_size: 4  # Must divide num_gpus evenly
  expert_model_parallel_size: 1
```

## Results

Evaluation results are saved to `eval.output_dir` (default: `/nemo_run/eval_results/`).

The nemo-evaluator-launcher generates:
- Per-benchmark scores in JSON format
- Aggregated metrics across all benchmarks
- Detailed logs for debugging

## See Also

### Upstream Projects

- [Export-Deploy](https://github.com/NVIDIA-NeMo/Export-Deploy) - Model deployment with Ray Serve and vLLM
- [Evaluator](https://github.com/NVIDIA-NeMo/Evaluator) - Benchmark orchestration with nemo-evaluator-launcher

### Related Documentation

- [SFT Training](sft.md) - Train instruction-following models
- [RL Training](rl.md) - Alignment training with GRPO
- [nemo-run Documentation](../../nemo-run.md) - Cluster execution details
