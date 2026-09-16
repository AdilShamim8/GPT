import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from gpt.config.model_config import ModelConfig


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand key/value heads for Grouped-Query Attention (GQA)."""
    if n_rep == 1:
        return x
    bs, n_kv_heads, seq_len, head_dim = x.shape
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, seq_len, head_dim)
        .reshape(bs, n_kv_heads * n_rep, seq_len, head_dim)
    )


class CausalSelfAttention(nn.Module):
    """Multi-Head / GQA Causal Self-Attention with KV-caching and SDPA."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head if config.n_kv_head is not None else config.n_head
        self.n_rep = self.n_head // self.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout_p = config.dropout

        self.q_proj = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.has_sdpa = hasattr(F, "scaled_dot_product_attention")
        if not self.has_sdpa:
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
                persistent=False,
            )

    def forward(
        self,
        x: torch.Tensor,
        use_cache: bool = False,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = x.size()

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        # Update and reuse KV cache during generation
        if layer_past is not None:
            past_k, past_v = layer_past
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)

        present = (k, v) if use_cache else None

        # Expand KV heads for GQA
        k_rep = repeat_kv(k, self.n_rep)
        v_rep = repeat_kv(v, self.n_rep)

        total_k_len = k_rep.size(-2)

        if self.has_sdpa and T == total_k_len:
            # Fast causal SDPA path during training / prefill
            y = F.scaled_dot_product_attention(
                q,
                k_rep,
                v_rep,
                attn_mask=None,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=True,
            )
        else:
            # Scaled dot-product attention with past context
            att = (q @ k_rep.transpose(-2, -1)) * (1.0 / math.sqrt(k_rep.size(-1)))
            if T > 1:
                # Causal mask for prefix
                att = att.masked_fill(
                    self.bias[:, :, :T, :total_k_len] == 0, float("-inf")
                )
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v_rep

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y, present
