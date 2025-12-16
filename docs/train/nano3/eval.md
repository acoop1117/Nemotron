# Model Evaluation

Evaluate trained models against standard benchmarks using Export-Deploy and nemo-evaluator-launcher.

## Overview

This command orchestrates model deployment and evaluation as a single RayJob:

1. **Deploy**: Starts a Ray Serve inference server using direct Ray integration
2. **Eval**: Runs benchmarks against the deployed model using nemo-evaluator-launcher
3. **Shutdown**: Gracefully shuts down the deployment

| Component | Description |
|-----------|-------------|
| `deploy_eval.py` | Combined deploy + evaluation script using Ray Serve |
| `config/` | Configuration files for deployment and evaluation settings |

## Requirements

- SLURM cluster with GPU nodes
- Model checkpoint in Megatron format
- Configured env profile (e.g., in `env.toml`)

## Quick Start

### Using nemotron CLI

```bash
# Evaluate a model checkpoint
nemotron nano3 model eval /path/to/checkpoint --run YOUR-CLUSTER

# Quick test with tiny config (single benchmark, limited samples)
nemotron nano3 model eval /path/to/checkpoint -c tiny --run YOUR-CLUSTER

# Preview configuration without execution
nemotron nano3 model eval /path/to/checkpoint --dry-run

# Detached execution (returns immediately)
nemotron nano3 model eval /path/to/checkpoint --batch YOUR-CLUSTER
```

### Override Settings

Use dotlist overrides to customize evaluation:

```bash
# Increase parallelism
nemotron nano3 model eval /path/to/checkpoint --run YOUR-CLUSTER \
    eval.parallelism=64

# Change deployment settings
nemotron nano3 model eval /path/to/checkpoint --run YOUR-CLUSTER \
    deploy.num_gpus=4 \
    deploy.tensor_model_parallel_size=2

# Limit samples for quick testing
nemotron nano3 model eval /path/to/checkpoint --run YOUR-CLUSTER \
    eval.limit_samples=100
```

## Configuration

### Config Files

| File | Description |
|------|-------------|
| `config/default.yaml` | Full evaluation with multiple benchmarks |
| `config/tiny.yaml` | Quick test with single benchmark, 100 samples |

### Configuration Structure

```yaml
model:
  checkpoint_path: ???     # Set via CLI argument
  model_type: gpt

deploy:
  host: "0.0.0.0"
  port: 8000
  num_gpus: 8              # Total GPUs for deployment
  tensor_model_parallel_size: 4
  expert_model_parallel_size: 1
  pipeline_model_parallel_size: 1

eval:
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

- [SFT Training](sft.md) - Train instruction-following models
- [RL Training](rl.md) - Alignment training with GRPO
- [nemo-run Documentation](../../nemo-run.md) - Cluster execution details
