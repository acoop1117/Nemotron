# NVIDIA Nemotron Developer Repository

Developer companion repo for working with NVIDIA's Nemotron models: inference, fine-tuning, agents, visual reasoning, deployment.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)

---

## 📂 Repo Layout

```
nemotron/
│
├── usage-cookbook/        Usage cookbooks (how to deploy, and simple model usage guides)
│
│
└── use-case-examples/     Examples of leveraging Nemotron Models in Agentic Workflows and more 
```

---

## What is Nemotron?

[NVIDIA Nemotron™](https://developer.nvidia.com/nemotron) is a family of open, high-efficiency models with fully transparent training data, weights, and recipes.

Nemotron models are designed for **agentic AI workflows** — they excel at coding, math, scientific reasoning, tool calling, instruction following, and visual reasoning (for the VL models).

They are optimized for deployment across a spectrum of compute tiers (edge, single GPU, data center) and support frameworks like NeMo and TensorRT-LLM, vLLM, and SGLang, with NIM microservice options for scalable serving.

---

### More Resources

- **[Usage Cookbook](usage-cookbook/)** - Practical deployment and simple model usage guides for Nemotron models
- **[Use Case Examples](use-case-examples/)** - Practical use-case examples and apps
- **[Nemotron Developer Page](https://developer.nvidia.com/nemotron)** - Developer resources for the Nemotron family of models
- **[Nemotron Research Hub](https://research.nvidia.com/labs/nemotron/)** - Research affiliated with the Nemotron effort
- **[Nemotron Datasets](https://huggingface.co/nvidia/datasets?search=nemotron)** - Datasets part of various Nemotron collections, from pre-training to post-training

---

## 💡 Feature Requests & Ideas

Have an idea for improving Nemotron models? Create an issue and tag it `idea`!

Your feedback helps shape the future of Nemotron models!

---

## Training Recipes (Coming Soon)

Full, reproducible training pipelines will be included in the `nemotron` package at `src/nemotron/recipes/`.

### Each Recipe Includes
- 🎨 **Synthetic Data Generation** - Scripts to generate synthetic datasets using [NVIDIA-NeMo/DataDesigner](https://github.com/NVIDIA-NeMo/DataDesigner)
- 🗂️ **Data Curation** - Scripts to prepare training data using [NVIDIA-NeMo/Curator](https://github.com/NVIDIA-NeMo/Curator)
- 🔁 **Training** - Complete training loops with hyperparameters using:
  - [NVIDIA-NeMo/Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main) for Megatron models
  - [NVIDIA-NeMo/Automodel](https://github.com/NVIDIA-NeMo/Automodel) for HuggingFace models
  - [NVIDIA-NeMo/NeMo-RL](https://github.com/NVIDIA-NeMo/RL/tree/main) when RL is needed
- 📊 **Evaluation** - Benchmark evaluation on standard suites using [NVIDIA-NeMo/Evaluator](https://github.com/NVIDIA-NeMo/Evaluator)
- 📖 **Documentation** - Detailed explanations of each stage

---

## Model Specific Usage Cookbooks

Learn how to deploy and use the models through an API.

| Model | Best For | Key Features | Trade-offs | Resources |
|-------|----------|--------------|------------|-----------|
| [**NVIDIA-Nemotron-3-Nano**](https://huggingface.co/collections/nvidia/nvidia-nemotron-v3) | High-throughput agentic workflows, reasoning, tool-use, chat | • 31.6B total / 3.6B active (MoE)<br>• Hybrid Mamba-Transformer MoE<br>• 1M-token context window<br>• Reasoning ON/OFF + thinking budget | Sparse MoE trades total params for efficiency | [📁 Cookbooks](./usage-cookbook/Nemotron-3-Nano/) |
| [**Llama-3.3-Nemotron-Super-49B-v1.5**](https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1_5) | Production deployments needing strong reasoning with efficiency | • 128K context<br>• Single H200 GPU<br>• RAG & tool calling<br>• Optimized via NAS | Balances accuracy & throughput | [📁 Cookbooks](./usage-cookbook/Llama-Nemotron-Super-49B-v1.5/) |
| [**NVIDIA-Nemotron-Nano-9B-v2**](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2) | Resource-constrained environments needing flexible reasoning | • 9B params<br>• Hybrid Mamba-2 architecture<br>• Controllable reasoning traces<br>• Unified reasoning/non-reasoning | Smaller model with configurable reasoning | [📁 Cookbooks](./usage-cookbook/Nemotron-Nano-9B-v2/) |
| [**NVIDIA-Nemotron-Nano-12B-v2-VL**](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL) | Document intelligence and video understanding | • 12B VLM<br>• Video & multi-image reasoning<br>• Controllable reasoning (/think mode)<br>• Efficient Video Sampling (EVS) | Vision-language with configurable reasoning | [📁 Cookbooks](./usage-cookbook/Nemotron-Nano2-VL/) |
| [**Llama-3.1-Nemotron-Safety-Guard-8B-v3**](https://huggingface.co/nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3) | Multilingual content moderation with cultural nuance | • 9 languages<br>• 23 safety categories<br>• Cultural sensitivity<br>• NeMo Guardrails integration | Focused on safety/moderation tasks | [📁 Cookbooks](./usage-cookbook/Llama-3.1-Nemotron-Safety-Guard-V3/) |
| **Nemotron-Parse** (link coming soon!) | Document parsing for RAG and AI agents | • VLM for document parsing<br>• Table extraction (LaTeX)<br>• Semantic segmentation<br>• Spatial grounding (bbox) | Specialized for document structure | [📁 Cookbooks](./usage-cookbook/Nemotron-Parse-v1.1/) |


## Nemotron Use Case Examples

Below is an outline of the end-to-end use case examples provided in the [`use-case-examples`](./use-case-examples/) directory. These scenarios demonstrate practical applications that go beyond basic model inference.

### What You'll Find

- **Agentic Workflows**  
  Orchestration of multi-step AI agents, integrating planning, context management, and external tools/APIs.

- **Retrieval-Augmented Generation (RAG) Systems**  
  Building pipelines that combine retrieval components (vector databases, search APIs) with Nemotron models for grounded, accurate outputs.

- **Integration with External Tools & APIs**  
  Examples of Nemotron models powering applications with structured tool calling, function execution, or data enrichment.

- **Production-Ready Application Patterns**  
  Architectures supporting scalability, monitoring, data pipelines, and real-world deployment considerations.

> See the [`use-case-examples/`](./use-case-examples/) subfolders for in-depth, runnable examples illustrating these concepts.

## Nemotron Open Datasets

More than just weights, recipes, and libraries: Nemotron is commited to opening data across many domains, training phases, and use cases. 

<details>
<summary><strong>Nemotron Data Catalogue</strong></summary>

<details>
<summary><strong>Nemotron 3 Family (Nano, Super, Ultra)</strong></summary>

*Hybrid Mamba-Transformer MoE architecture for agentic AI*

### Pre-training Data

| Dataset | Usage | License | Description |
|---------|-------|---------|-------------|
| [Nemotron-CC-v2.1](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2.1) | Pre-training | NVIDIA Data Agreement | 2.5T tokens English web data with synthetic rephrases and translations |
| [Nemotron-CC-v2](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2) | Pre-training | NVIDIA Data Agreement | 6.6T tokens quality-filtered Common Crawl with multilingual Q&A |
| [Nemotron-CC-Code-v1](https://huggingface.co/datasets/nvidia/Nemotron-CC-Code-v1) | Pre-training | NVIDIA Data Agreement | 427.9B tokens from Common Crawl code pages using Lynx + LLM pipeline |
| [Nemotron-CC-Math-v1](https://huggingface.co/datasets/nvidia/Nemotron-CC-Math-v1) | Pre-training | NVIDIA Data Agreement | 133B-token math dataset from Common Crawl using Lynx + LLM pipeline |
| [Nemotron-Pretraining-Code-v2](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Code-v2) | Pre-training | NVIDIA Data Agreement | Updated GitHub code + synthetic QA with STEM reasoning for Nemotron 3 |
| [Nemotron-Pretraining-Dataset-sample](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Dataset-sample) | Pre-training (Sample) | NVIDIA Data Agreement | Sample subset of Nemotron pre-training corpus for experimentation |

### Post-training Data (SFT)

| Dataset | Usage | License | Description |
|---------|-------|---------|-------------|
| [Nemotron-Math-Proofs-v1](https://huggingface.co/datasets/nvidia/Nemotron-Math-Proofs-v1) | SFT | CC-BY-4.0 | Mathematical proofs dataset for Nemotron 3 post-training |
| [Nemotron-Science-v1](https://huggingface.co/datasets/nvidia/Nemotron-Science-v1) | SFT | CC-BY-4.0 | Synthetic science reasoning (MCQA + chemistry RQA) for Nemotron 3 Nano |
| [Nemotron-Instruction-Following-Chat-v1](https://huggingface.co/datasets/nvidia/Nemotron-Instruction-Following-Chat-v1) | SFT | CC-BY-4.0 | Multi-turn chat and structured output generation for Nemotron 3 Nano |
| [Nemotron-Agentic-v1](https://huggingface.co/datasets/nvidia/Nemotron-Agentic-v1) | SFT | CC-BY-4.0 | Multi-turn trajectories for conversational tool use and agentic workflows |

### Reinforcement Learning Data

| Dataset | Usage | License | Description |
|---------|-------|---------|-------------|
| [Nemotron-3-Nano-RL-Training-Blend](https://huggingface.co/datasets/nvidia/Nemotron-3-Nano-RL-Training-Blend) | RL Training | CC-BY-4.0 | Curated multi-domain blend for Nemotron-3-Nano-30B-A3B |
| [Nemotron-CrossThink](https://huggingface.co/datasets/nvidia/Nemotron-CrossThink) | RL Training | CC-BY-4.0 | Multi-domain QA with MCQ and open-ended formats for verifiable rewards |
| [Nemotron-Cascade-RL-SWE](https://huggingface.co/datasets/nvidia/Nemotron-Cascade-RL-SWE) | RL Training | CC-BY-4.0 | SWE code repair from SWE-Bench, SWE-Smith, R2E-Gym |
| [Nemotron-Cascade-RL-Instruction-Following](https://huggingface.co/datasets/nvidia/Nemotron-Cascade-RL-Instruction-Following) | RL Training | ODC-BY-1.0 | 108K samples for instruction-following RL |
| [Nemotron-RL-instruction_following](https://huggingface.co/datasets/nvidia/Nemotron-RL-instruction_following) | RL Training | ODC-BY-1.0 | Verifiable instruction adherence from WildChat-1M + Open-Instruct |
| [Nemotron-RL-instruction_following-structured_outputs](https://huggingface.co/datasets/nvidia/Nemotron-RL-instruction_following-structured_outputs) | RL Training | ODC-BY-1.0 | JSON schema-constrained output formatting tests |
| [Nemotron-RL-agent-workplace_assistant](https://huggingface.co/datasets/nvidia/Nemotron-RL-agent-workplace_assistant) | RL Training | ODC-BY-1.0 | Workplace assistant agent tasks for NeMo Gym |
| [Nemotron-RL-knowledge-web_search-mcqa](https://huggingface.co/datasets/nvidia/Nemotron-RL-knowledge-web_search-mcqa) | RL Training | ODC-BY-1.0 | Web search and multiple-choice QA tasks for NeMo Gym |

</details>

<details>
<summary><strong>Llama-Nemotron Family (Nano 8B, Super 49B, Ultra 253B)</strong></summary>

*Post-trained derivatives of Meta Llama models*

| Dataset | Usage | License | Description |
|---------|-------|---------|-------------|
| [Llama-Nemotron-Post-Training-Dataset](https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset) | SFT + RL | CC-BY-4.0 | Math, code, reasoning data for Llama-Nemotron (2.2M math, 500K code) |
| [Nemotron-Post-Training-Dataset-v1](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v1) | SFT | CC-BY-4.0 | Math, code, STEM, tool calling for Llama-3.3-Nemotron-Super-49B-v1.5 |
| [Nemotron-Post-Training-Dataset-v2](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2) | SFT + RL | CC-BY-4.0 | Multilingual extension (Spanish, French, German, Italian, Japanese) |

</details>

<details>
<summary><strong>OpenMath-Nemotron / OpenReasoning</strong></summary>

*Math and code reasoning datasets (AIMO-2 winning solution)*

| Dataset | Usage | License | Description |
|---------|-------|---------|-------------|
| [OpenMathReasoning](https://huggingface.co/datasets/nvidia/OpenMathReasoning) | SFT | CC-BY-4.0 | 5.68M samples, 306K problems from AoPS with CoT/TIR (AIMO-2 winner) |
| [Nemotron-Math-v2](https://huggingface.co/datasets/nvidia/Nemotron-Math-v2) | SFT | CC-BY-4.0 | 347K samples and 7M reasoning trajectories for Deeper Math Reasoning Supervision |
| [OpenCodeReasoning](https://huggingface.co/datasets/nvidia/OpenCodeReasoning) | SFT | CC-BY-4.0 | 735K Python samples across 28K competitive programming questions |
| [OpenCodeReasoning-2](https://huggingface.co/datasets/nvidia/OpenCodeReasoning-2) | SFT | CC-BY-4.0 | 2.5M samples (1.4M Python, 1.1M C++) with code completion and critique |
| [Nemotron-Competitive-Programming-v1](https://huggingface.co/datasets/nvidia/Nemotron-Competitive-Programming-v1) | SFT | CC-BY-4.0 | 2M+ Python and 1M+ C++ samples across 34K competitive programming questions |

</details>

<details>
<summary><strong>Cosmos-Reason1 (Physical AI / Robotics)</strong></summary>

*Vision language model for embodied reasoning*

| Dataset | Usage | License | Description |
|---------|-------|---------|-------------|
| [Cosmos-Reason1-SFT-Dataset](https://huggingface.co/datasets/nvidia/Cosmos-Reason1-SFT-Dataset) | SFT | CC-BY-4.0 | Video-text pairs for robotics, ego-centric demos, AV reasoning |
| [Cosmos-Reason1-RL-Dataset](https://huggingface.co/datasets/nvidia/Cosmos-Reason1-RL-Dataset) | RL Training | CC-BY-4.0 | RL data for physical common sense and embodied reasoning |
| [Cosmos-Reason1-Benchmark](https://huggingface.co/datasets/nvidia/Cosmos-Reason1-Benchmark) | Evaluation | CC-BY-4.0 | Benchmark for embodied reasoning (robotics, HoloAssist, AV) |

</details>

<details>
<summary><strong>Llama-Nemotron VLM (Vision-Language)</strong></summary>

*Visual question answering and document intelligence*

| Dataset | Usage | License | Description |
|---------|-------|---------|-------------|
| [Nemotron-VLM-Dataset-v2](https://huggingface.co/datasets/nvidia/Nemotron-VLM-Dataset-v2) | VLM Training | CC-BY-4.0 (some CC-BY-SA-4.0) | 8M samples for OCR, image reasoning, video QA with chain-of-thought |
| [Llama-Nemotron-VLM-Dataset-v1](https://huggingface.co/datasets/nvidia/Llama-Nemotron-VLM-Dataset-v1) | VLM Training | CC-BY-4.0 (some CC-BY-SA-4.0) | 3M samples for visual question answering and captioning |

</details>

<details>
<summary><strong>HelpSteer / Reward Modeling</strong></summary>

*Alignment and RLHF data for Nemotron reward models*

| Dataset | Usage | License | Description |
|---------|-------|---------|-------------|
| [HelpSteer3](https://huggingface.co/datasets/nvidia/HelpSteer3) | Reward Modeling | CC-BY-4.0 | 40K+ samples; top on RM-Bench/JudgeBench with preference, feedback, edit-quality |
| [HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2) | Reward Modeling | CC-BY-4.0 | 21K samples with 5 attributes for Llama-3.1-Nemotron-70B-Reward |
| [HelpSteer](https://huggingface.co/datasets/nvidia/HelpSteer) | SteerLM Training | CC-BY-4.0 | 37K samples (helpfulness, correctness, coherence, complexity, verbosity) |

</details>

<details>
<summary><strong>NemoGuard / Safety</strong></summary>

*Content safety and guardrail training*

| Dataset | Usage | License | Description |
|---------|-------|---------|-------------|
| [Aegis-AI-Content-Safety-Dataset-2.0](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0) | Content Moderation | CC-BY-4.0 | Extended safety dataset with 23 violation categories |
| [Aegis-AI-Content-Safety-Dataset-1.0](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-1.0) | Content Moderation | CC-BY-4.0 | 11K annotated interactions covering 13 risk categories |
| [Nemotron-Content-Safety-Audio-Dataset](https://huggingface.co/datasets/nvidia/Nemotron-Content-Safety-Audio-Dataset) | Audio Safety | CC-BY-4.0 | 1.9K audio files from Aegis 2.0 with accent diversity |

</details>

<details>
<summary><strong>Synthetic Personas / Data Generation</strong></summary>

| Dataset | Usage | License | Description |
|---------|-------|---------|-------------|
| [Nemotron-Personas-USA](https://huggingface.co/datasets/nvidia/Nemotron-Personas-USA) | SDG | CC-BY-4.0 | 1M US personas grounded in Census demographics |
| [Nemotron-Personas-Japan](https://huggingface.co/datasets/nvidia/Nemotron-Personas-Japan) | SDG | CC-BY-4.0 | 1M Japanese personas aligned with regional statistics |
| [Nemotron-Personas-India](https://huggingface.co/datasets/nvidia/Nemotron-Personas-India) | SDG | CC-BY-4.0 | 3M Indian personas for sovereign AI development |
| [Nemotron-Personas](https://huggingface.co/datasets/nvidia/Nemotron-Personas) | SDG | CC-BY-4.0 | 100K US personas with 22 fields aligned to Census data |

</details>

<details>
<summary><strong>Privacy / PII Detection</strong></summary>

| Dataset | Usage | License | Description |
|---------|-------|---------|-------------|
| [Nemotron-PII](https://huggingface.co/datasets/nvidia/Nemotron-PII) | NER Training | CC-BY-4.0 | 100K synthetic records with 55+ PII/PHI entity types |

</details>

<details>
<summary><strong>Autonomous Vehicles</strong></summary>

| Dataset | Usage | License | Description |
|---------|-------|---------|-------------|
| [PhysicalAI-Autonomous-Vehicles](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles) | Training | NVIDIA AV Dataset License | 1,700 hours multi-sensor data from 25 countries, 306K clips |
| [PhysicalAI-Autonomous-Vehicle-Cosmos-Drive-Dreams](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicle-Cosmos-Drive-Dreams) | SDG | CC-BY-4.0 | 81K synthetic videos with LiDAR and HD-map annotations |
| [PhysicalAI-Autonomous-Vehicle-Cosmos-Synthetic](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicle-Cosmos-Synthetic) | SDG | CC-BY-4.0 | Cosmos-generated synthetic driving scenarios |
| [PhysicalAI-Autonomous-Vehicles-NuRec](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec) | Reconstruction | NVIDIA AV Dataset License | NuScenes-based reconstruction data |

</details>

<details>
<summary><strong>Robotics</strong></summary>

| Dataset | Usage | License | Description |
|---------|-------|---------|-------------|
| [PhysicalAI-Robotics-Manipulation-Augmented](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Manipulation-Augmented) | Training | CC-BY-4.0 | 1K Franka Panda demos with Cosmos Transfer1 domain augmentation |

</details>

<details>
<summary><strong>Evaluation / Benchmarks</strong></summary>

| Dataset | Usage | License | Description |
|---------|-------|---------|-------------|
| [Scoring-Verifiers](https://huggingface.co/datasets/nvidia/Scoring-Verifiers) | Evaluation | CC-BY-4.0 | Benchmark for test case generation and code reward models |

</details>

<details>
<summary><strong>Legacy / Nemotron 2</strong></summary>

| Dataset | Usage | License | Description |
|---------|-------|---------|-------------|
| [Nemotron-Pretraining-Code-v1](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Code-v1) | Pre-training | NVIDIA Data Agreement | GitHub-sourced code corpus for Nemotron Nano 2 |

</details>

<details>
<summary><strong>General SFT / RLHF</strong></summary>

| Dataset | Usage | License | Description |
|---------|-------|---------|-------------|
| [Daring-Anteater](https://huggingface.co/datasets/nvidia/Daring-Anteater) | SFT/RLHF | CC-BY-4.0 | Instruction tuning dataset; synthetic subsets + FinQA, wikitablequestions |
| [sft_datablend_v1](https://huggingface.co/datasets/nvidia/sft_datablend_v1) | SFT | CC-BY-4.0 | SFT data blend for RLHF pipeline |

</details>

</details>

## Contributing

We welcome contributions! Whether it's examples, recipes, or other tools you'd find useful.

Please read our **[Contributing Guidelines](CONTRIBUTING.md)** before submitting pull requests.

## Documentation

- **[Contributing Guidelines](CONTRIBUTING.md)** - How to contribute to this project
- **[Changelog](CHANGELOG.md)** - Version history and changes

---

## License

Apache 2.0 License - see [LICENSE](LICENSE) file for details.

---

**NVIDIA Nemotron** - Open, transparent, and reproducible.
