# Nano3 Model Evaluation Recipe

This recipe orchestrates model deployment and evaluation for Nano3 models using:
- **RayJob**: Deploys model via Ray Serve (Export-Deploy container)
- **nemo-evaluator-launcher**: Runs benchmarks via SLURM (evaluator container)

## Requirements

- SLURM cluster with GPU nodes
- Container: `nvcr.io/nvidian/nemo:25.09.rc11.mamba-conv-fix` (or configured in env profile)
- Model checkpoint in Megatron format

## Usage

```bash
# Attached mode (CLI orchestrates, interactive logs)
nemotron nano3 model eval /path/to/checkpoint --run <profile>

# Detached mode (cluster orchestrates, submit and return)
nemotron nano3 model eval /path/to/checkpoint --batch <profile>

# Preview configuration without execution
nemotron nano3 model eval /path/to/checkpoint --dry-run

# Use tiny config for quick testing
nemotron nano3 model eval /path/to/checkpoint -c tiny --run <profile>

# Override evaluation settings
nemotron nano3 model eval /path/to/checkpoint --run <profile> \
    eval.parallelism=64 \
    deploy.port=9000
```

## Configuration

### Config Files

- `config/default.yaml` - Full evaluation with multiple benchmarks
- `config/tiny.yaml` - Quick test with single benchmark and limited samples

### Key Settings

```yaml
model:
  checkpoint_path: ???  # Set via CLI argument
  model_type: gpt

deploy:
  num_gpus: 8
  tensor_model_parallel_size: 4
  expert_model_parallel_size: 1
  port: 8000

eval:
  tasks:
    - name: adlr_arc_challenge_llama_25_shot
    - name: hellaswag
  parallelism: 32

comms:
  endpoint_file: endpoint.json
  completion_file: done
```

## Architecture

### --run Mode (CLI Orchestrates)

```
CLI (local/login node)
├─ Start RayJob (deploy only)
├─ Poll endpoint.json via SSH
├─ Call nemo-evaluator-launcher run_eval()
├─ Poll launcher status via get_status()
├─ Write done signal via SSH
└─ Follow RayJob logs

RayJob (GPU cluster)
├─ Deploy model via Ray Serve
├─ Write endpoint.json
├─ Wait for done signal
└─ Shutdown
```

### --batch Mode (Cluster Orchestrates)

```
CLI (local/login node)
├─ Start RayJob (deploy + orchestrate)
└─ Return immediately

RayJob (GPU cluster)
├─ Deploy model via Ray Serve
├─ Write endpoint.json
├─ Call nemo-evaluator-launcher run_eval() directly
│   └─ SLURM executor submits eval sbatch job
├─ Poll launcher status until complete
├─ Save results
└─ Shutdown (no external done signal needed)
```

### Benefits

- **Correct containers**: Deploy uses Export-Deploy container, eval uses evaluator container
- **Two modes**: --run for interactive, --batch for detached execution
- **Leverage nemo-evaluator-launcher**: Uses its SLURM executor with auto-resume
- **RayJob for deploy**: Direct Ray Serve integration, no subprocess/polling

## Files

- `deploy.py` - Deploy script with --orchestrate flag for batch mode
- `config/` - YAML configuration files
