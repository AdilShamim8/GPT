# High-Performance Inference & Decoding Strategies

The `gpt.inference` module is engineered for low latency, controllable text generation with support for streaming tokens, KV caching, and modern sampling mechanics.

## Sampling Strategies

### 1. Temperature Scaling
$$P(x_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$
- **$T \to 0$ (Greedy / Argmax):** Completely deterministic, selects the highest-probability token. Ideal for coding, arithmetic, and factual tasks.
- **$T \in [0.7, 0.9]$:** Balanced creativity and coherence (default for creative writing).
- **$T > 1.2$:** High randomness, creative risk, higher entropy.

### 2. Top-k Filtering
Retains only the $k$ highest-probability tokens and sets all remaining logit values to $-\infty$.

### 3. Top-p (Nucleus) Sampling
Retains the smallest set of top tokens whose cumulative probability mass sums to at least $p$ (e.g. $p = 0.95$). This dynamic threshold avoids generating improbable tail tokens when the distribution is peaked, while allowing diverse exploration when the distribution is flat.

### 4. Repetition & Frequency Penalties
Prevents the autoregressive model from entering repetitive loops by discounting the logits of tokens that have already been generated in the recent context.

## KV Cache Acceleration

In naive generation, evaluating token $T+1$ re-computes key and value projections for all preceding $T$ tokens, incurring $O(T^2)$ total operations.

With **KV caching** enabled in `TextGenerator`:
- The previous keys and values across all transformer layers are stored in GPU/RAM memory.
- For each subsequent step, only the new single token $(B, 1)$ is projected through query, key, and value matrices.
- The new key and value are appended to the cache in-place, reducing per-token step cost to $O(1)$ and total generation cost to $O(T)$.

## CLI Generation Command

```bash
python generate.py \
  --checkpoint checkpoints/shakespeare/ckpt_best.pt \
  --prompt "Romeo: " \
  --max_tokens 300 \
  --temperature 0.8 \
  --top_p 0.9 \
  --repetition_penalty 1.15
```
