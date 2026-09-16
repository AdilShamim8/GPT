# Parameter-Efficient Fine-Tuning with LoRA

Low-Rank Adaptation (LoRA, Hu et al., 2021) freezes pretrained model weights and injects trainable rank decomposition matrices into the Transformer architecture.

## Mathematical Formulation

For a linear projection $h = W_0 x$, LoRA constrains weight updates by representing $\Delta W$ as:
$$\Delta W = \frac{\alpha}{r} B \cdot A$$
where:
- $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ is frozen.
- $A \in \mathbb{R}^{r \times d_{\text{in}}}$ is initialized with Kaiming uniform.
- $B \in \mathbb{R}^{d_{\text{out}} \times r}$ is initialized to zero.
- $r \ll \min(d_{\text{in}}, d_{\text{out}})$ is the rank (typically $r \in [4, 16]$).
- $\alpha$ is a constant scaling hyperparameter.

This reduces trainable parameters by **95% to 99%**, requiring minimal VRAM while achieving performance on par with full parameter fine-tuning.

## Instruction Tuning Dataset Format

Prepare training examples as JSONL with `instruction` and `response` keys:

```json
{"instruction": "What is the primary function of self-attention in a transformer?", "response": "Self-attention allows each token to dynamically compute affinity scores and aggregate context from all positions."}
```

Prompt tokens are automatically masked out with target ID `-1`, ensuring the model is only penalized for its generated responses.

## CLI Fine-Tuning Execution

```bash
python finetune.py \
  --checkpoint checkpoints/shakespeare/ckpt_best.pt \
  --data data/instructions.jsonl \
  --output_dir checkpoints/lora_adapter \
  --rank 8 \
  --alpha 16.0 \
  --epochs 3 \
  --lr 2e-4 \
  --merge
```
