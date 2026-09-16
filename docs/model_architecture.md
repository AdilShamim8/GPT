# Transformer Model Architecture in nano-gpt-prod

## Architectural Overview

`nano-gpt-prod` implements a modern, highly optimized autoregressive decoder-only Transformer based on the architecture of GPT-2 and modern advancements (LLaMA / Mistral / PaLM).

```
Input Tokens (B, T)
        │
        ▼
Token Embedding (wte) + Positional Embedding (wpe / RoPE)
        │
        ▼
   Dropout Layer
        │
   ┌────┴────────────────────────┐
   │  Transformer Block × N      │
   │  ├── Pre-Norm (LN / RMSNorm)│
   │  ├── Causal Self-Attention  │
   │  │   (MHA / GQA + SDPA)     │
   │  ├── Residual Add (+)       │
   │  ├── Pre-Norm (LN / RMSNorm)│
   │  ├── MLP (GELU / SwiGLU)    │
   │  └── Residual Add (+)       │
   └────┬────────────────────────┘
        ▼
Final Normalization (ln_f)
        │
        ▼
Language Model Head (Linear w/ Weight Tying)
        │
        ▼
Logits (B, T, Vocab Size)
```

## Key Innovations & Technical Features

1. **Memory-Efficient Attention**:
   - Uses PyTorch 2.0+ `scaled_dot_product_attention` (SDPA / FlashAttention) to avoid explicit $O(T^2)$ materialization of the attention matrix during training.
   - Built-in fallback to masked softmax attention when running on older platforms.

2. **Grouped-Query Attention (GQA) & Multi-Query Attention (MQA)**:
   - Configurable `n_kv_head` allows sharing key/value heads across query groups, reducing memory bandwidth pressure and KV-cache size during decoding.

3. **Key-Value (KV) Caching**:
   - Enables $O(1)$ latency per token generated during autoregressive decoding by caching previous keys and values.

4. **Modern Normalization & Activation**:
   - Supports both classic **LayerNorm** (with or without learnable bias) and **RMSNorm**.
   - Supports classic **GELU** and modern **SwiGLU** (Swish Gated Linear Units).

5. **Weight Tying & Scaled Initialization**:
   - In accordance with Press & Wolf (2016) and Radford et al. (GPT-2), token embeddings and LM head projection share weights.
   - Residual projections are initialized with standard deviation scaled by $\frac{0.02}{\sqrt{2 \times N_{\text{layer}}}}$.

6. **Hugging Face Weight Interoperability**:
   - Seamlessly convert and import official pretrained OpenAI weights from Hugging Face (`gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`).
