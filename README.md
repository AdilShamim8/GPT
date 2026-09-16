# nano-gpt-prod

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green)
![CI](https://img.shields.io/badge/CI-Passing-brightgreen?logo=githubactions)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ed?logo=docker)

**An enterprise-grade, extensible Generative Pretrained Transformer (GPT) framework built from scratch in pure PyTorch.**

[Features](#features) • [Quickstart](#quickstart) • [Architecture](#model-architecture) • [REST API](#openai-compatible-api) • [Web Studio](#interactive-web-studio) • [LoRA Fine-Tuning](#parameter-efficient-fine-tuning-lora) • [Docs](#documentation-index)

</div>

---

## Overview

`nano-gpt-prod` transforms educational micro-GPT implementations into an end-to-end, production-grade LLM development system. It combines the minimalism and clarity of Andrej Karpathy's lecture materials with state-of-the-art transformer architecture enhancements, memory-mapped data pipelines, distributed training, LoRA adaptation, an OpenAI-compatible REST API, and an interactive Web Playground.

---

## Features

- ⚡ **Cutting-Edge Architecture**:
  - Pre-LayerNorm decoder blocks with residual scaling ($1/\sqrt{2 \times N_{\text{layer}}}$).
  - PyTorch 2.0+ **Scaled Dot-Product Attention (SDPA / FlashAttention)** kernel integration with automatic causal fallback.
  - **Grouped-Query Attention (GQA)** and **Multi-Query Attention (MQA)** for memory-efficient KV states.
  - **Rotary Position Embeddings (RoPE)** alongside classic learned positional embeddings.
  - Configurable activations (**GELU, SwiGLU, QuickGELU**) and normalizations (**LayerNorm, RMSNorm**).
  - Weight tying between token embeddings and the language model head.
- 🏎️ **Accelerated Inference & Streaming**:
  - $O(1)$ step autoregressive decoding with **Key-Value (KV) Caching**.
  - Advanced decoding controls: **Temperature, Top-k, Top-p (Nucleus), Min-p, and Repetition Penalty**.
  - Token-by-token real-time streaming to stdout and Server-Sent Events (SSE).
- 🗄️ **Scalable Data Pipelines**:
  - Zero-copy binary datasets powered by `numpy.memmap` (`uint16`) for billion-token corpora.
  - Custom Byte-Pair Encoding (BPE) tokenizer trained from raw bytes with byte fallback.
  - Full compatibility with OpenAI's official `tiktoken` (`gpt2`, `cl100k_base`).
- 🎯 **Parameter-Efficient Adaptation (LoRA)**:
  - Clean `LoRALinear` layers implemented from scratch ($W_0 + \frac{\alpha}{r} B \cdot A$).
  - Trainable parameter reduction of 95%+ with standalone adapter persistence and base weight merging.
  - Instruction tuning dataset loader with prompt loss masking (`-1`).
- 🌐 **Production Serving & OpenAI API**:
  - Built-in zero-dependency HTTP server (`serve.py`).
  - OpenAI-compatible endpoints: `/v1/models`, `/v1/completions`, and `/v1/chat/completions` (with `stream: true`).
- 🖥️ **Interactive Web Playground Studio**:
  - Modern glassmorphism UI with real-time SSE streaming, live telemetry, and visual token inspector.
- 📦 **DevOps & Portability**:
  - Multi-stage `Dockerfile`, `docker-compose.yml`, and `Makefile`.
  - ONNX and TorchScript export utilities for C++ and edge deployment.

---

## Model Architecture

```
Input Tokens (Batch, Seq_Len)
        │
        ▼
Token Embedding (wte) + Positional Embedding (wpe / RoPE)
        │
        ▼
   Dropout Layer
        │
   ┌────┴──────────────────────────────────────┐
   │  Transformer Block × N                    │
   │  ├── Pre-Norm (LayerNorm / RMSNorm)       │
   │  ├── Causal Self-Attention                │
   │  │   (MHA / GQA / MQA + Flash SDPA)       │
   │  ├── Residual Add (+)                     │
   │  ├── Pre-Norm (LayerNorm / RMSNorm)       │
   │  ├── Feed-Forward MLP (GELU / SwiGLU)     │
   │  └── Residual Add (+)                     │
   └────┬──────────────────────────────────────┘
        ▼
Final Normalization (ln_f)
        │
        ▼
Language Model Head (Linear w/ Weight Tying)
        │
        ▼
Logits (Batch, Seq_Len, Vocab_Size)
```

---

## Quickstart

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/AdilShamim8/GPT.git
cd GPT

# Install dependencies and package in editable mode
pip install -e .
```

### 2. Prepare Training Dataset

```bash
# Automated Tiny Shakespeare preparation (compiles train.bin and val.bin)
python scripts/prepare_shakespeare.py --data_dir data/shakespeare
```

### 3. Train Model

```bash
# Train using YAML configuration profile
python train.py --config configs/shakespeare_char.yaml

# Or train directly with CLI flag overrides
python train.py --preset nano_shakespeare --max_iters 3000 --device auto
```

### 4. Interactive Text Generation

```bash
# Stream generation to terminal
python generate.py \
  --checkpoint checkpoints/shakespeare/ckpt_best.pt \
  --prompt "ROMEO: What lady is that" \
  --max_tokens 250 \
  --temperature 0.8
```

### 5. Launch OpenAI-Compatible API & Web Studio

```bash
# Launch server and auto-open web playground
python serve.py --checkpoint checkpoints/shakespeare/ckpt_best.pt --open
```

Navigate to `http://localhost:8000` to interact with the model via the Web Studio.

---

## OpenAI-Compatible API

The server provides complete drop-in compatibility with standard OpenAI SDKs and HTTP clients:

### cURL Chat Completion

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nano_shakespeare",
    "messages": [
      {"role": "system", "content": "You are a Shakespearean poet."},
      {"role": "user", "content": "Shall I compare thee to a summer day?"}
    ],
    "temperature": 0.8,
    "max_tokens": 150
  }'
```

### Python OpenAI Client

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="nano_shakespeare",
    messages=[{"role": "user", "content": "Hear me speak:"}],
    stream=True,
)

for chunk in response:
    content = chunk.choices[0].delta.content or ""
    print(content, end="", flush=True)
```

---

## Parameter-Efficient Fine-Tuning (LoRA)

Fine-tune on custom instruction datasets while freezing 95%+ of model weights:

```bash
python finetune.py \
  --checkpoint checkpoints/shakespeare/ckpt_best.pt \
  --data data/instructions.jsonl \
  --output_dir checkpoints/lora_adapter \
  --rank 8 \
  --alpha 16.0 \
  --epochs 3 \
  --merge
```

---

## Benchmarking & Evaluation

Measure exact perplexity and throughput metrics (tok/s, TTFT, ITL):

```bash
python evaluate.py \
  --checkpoint checkpoints/shakespeare/ckpt_best.pt \
  --test_file more.txt \
  --output eval_report.json
```

---

## Docker Deployment

```bash
# Build container image
docker build -t nano-gpt-prod .

# Run API server via docker-compose
docker-compose up -d gpt-serve
```

---

## Documentation Index

- [Model Architecture & Attention Mathematics](docs/model_architecture.md)
- [Scalable Data Pipeline & Memmap Binaries](docs/data_pipeline.md)
- [Tokenization Engine & BPE Training](docs/tokenization.md)
- [Inference, Sampling & KV Caching](docs/inference.md)
- [LoRA Fine-Tuning Guide](docs/finetuning_guide.md)
- [Evaluation & Perplexity Methodology](docs/evaluation.md)
- [Web Playground Studio & Telemetry](docs/web_ui.md)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Developed with precision by [Adil Shamim](https://github.com/AdilShamim8). Inspired by Andrej Karpathy's `nanoGPT` educational lectures.
