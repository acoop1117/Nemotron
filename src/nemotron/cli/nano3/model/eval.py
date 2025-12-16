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

Usage:
    nemotron nano3 model eval -c sft --run <profile>
    nemotron nano3 model eval -c pretrain --run <profile>
    nemotron nano3 model eval -c rl --run <profile>
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

    The model to evaluate is specified in the config file (run.model).
    Use stage-specific configs (-c sft, -c pretrain, -c rl) which have
    the model artifact pre-configured.

    Examples:
        # Evaluate SFT checkpoint (uses ModelArtifact-sft:latest)
        nemotron nano3 model eval -c sft --run YOUR-CLUSTER

        # Evaluate pretrained checkpoint
        nemotron nano3 model eval -c pretrain --run YOUR-CLUSTER

        # Evaluate RL-aligned checkpoint
        nemotron nano3 model eval -c rl --run YOUR-CLUSTER

        # Quick test with limited samples
        nemotron nano3 model eval -c sft -c tiny --run YOUR-CLUSTER

        # Override model artifact version
        nemotron nano3 model eval -c sft --run YOUR-CLUSTER run.model=ModelArtifact-sft:v5
    """
    ...
