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

"""Model evaluation command.

Orchestrates model deployment and evaluation via:
1. Deploy: Export-Deploy's Ray-based inference server (RayJob)
2. Eval: nemo-evaluator-launcher evaluation (SLURM sbatch)

Supports two execution modes:
- --run: CLI orchestrates evaluation (attached, interactive logs)
- --batch: RayJob orchestrates everything (detached, submit and return)

Usage:
    nemotron nano3 model eval /path/to/checkpoint --run <profile>
    nemotron nano3 model eval -c sft --run <profile>
    nemotron nano3 model eval -c pretrain --run <profile>
"""

from __future__ import annotations

import typer

from nemotron.kit.cli.model_eval import model_eval

# Paths relative to repository root
CONFIG_DIR = "src/nemotron/recipes/nano3/eval/config"
SCRIPT = "src/nemotron/recipes/nano3/eval/deploy.py"


@model_eval(
    name="nano3/eval",
    config_dir=CONFIG_DIR,
    script=SCRIPT,
    default_config="default",
)
def eval_cmd(ctx: typer.Context) -> None:
    """Evaluate model performance via deploy + eval pipeline.

    Deploys the model using Export-Deploy's Ray inference server and runs
    evaluation using nemo-evaluator-launcher against the deployed endpoint.

    Two execution modes:
    - --run <profile>: CLI orchestrates (attached, interactive logs)
    - --batch <profile>: Cluster orchestrates (detached, submit and return)

    Examples:
        # With explicit checkpoint path
        nemotron nano3 model eval /path/to/checkpoint --run YOUR-CLUSTER

        # Using stage-specific configs (model artifact pre-configured)
        nemotron nano3 model eval -c sft --run YOUR-CLUSTER
        nemotron nano3 model eval -c pretrain --run YOUR-CLUSTER
        nemotron nano3 model eval -c rl --run YOUR-CLUSTER

        # Override model in stage config
        nemotron nano3 model eval -c sft --run YOUR-CLUSTER run.model=ModelArtifact-sft:v5
    """
    ...
