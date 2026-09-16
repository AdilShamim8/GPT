# Model Evaluation & Benchmarking

Objective evaluation of generative language models requires both statistical performance metrics (loss, perplexity) and computational latency metrics.

## Statistical Evaluation: Perplexity (PPL)

Perplexity is defined as the exponentiated negative log-likelihood of a sequence:

$$\text{PPL}(X) = \exp\left( -\frac{1}{N} \sum_{i=1}^N \log P(x_i \mid x_{<i}) \right) = \exp(\mathcal{L})$$

- **Interpretation:** Perplexity corresponds to the effective branching factor—the geometric average number of likely candidate words the model is choosing between at each step.
- Lower perplexity indicates superior predictive confidence and modeling accuracy.
- **Sliding Window:** `calculate_perplexity()` slides a context window of size `block_size` across continuous documents with configurable stride, avoiding boundary bias.

## Latency & Throughput Metrics

1. **Tokens Per Second (tok/s):** Aggregate sustained token generation rate.
2. **Time to First Token (TTFT):** Prompt processing (prefill) latency until the initial completion token is output. Critical for interactive chatbots.
3. **Inter-Token Latency (ITL):** Average time elapsed between consecutive generated tokens.

## Running Evaluation via CLI

```bash
python evaluate.py \
  --checkpoint checkpoints/shakespeare/ckpt_best.pt \
  --test_file more.txt \
  --output eval_report.json
```
