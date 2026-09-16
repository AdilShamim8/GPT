import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from gpt.config.model_config import ModelConfig
from gpt.model.base import BaseModel
from gpt.model.normalization import get_norm_layer
from gpt.model.rope import precompute_freqs_cis
from gpt.model.transformer_block import TransformerBlock


class GPT(BaseModel):
    """Production Autoregressive Decoder-Only Transformer (GPT) Architecture."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=(
                    None
                    if config.use_rope
                    else nn.Embedding(config.block_size, config.n_embd)
                ),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
                ln_f=get_norm_layer(config.norm_type, config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: tie token embedding weights with output linear projection
        if config.tie_weights:
            self.transformer.wte.weight = self.lm_head.weight

        # Precompute RoPE complex frequencies if requested
        if config.use_rope:
            head_dim = config.n_embd // config.n_head
            self.register_buffer(
                "freqs_cis",
                precompute_freqs_cis(head_dim, config.block_size),
                persistent=False,
            )

        # Initialize weights following standard GPT-2 scheme
        self.apply(self._init_weights)

        # Apply special scaled initialization to residual projections per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        return_all_logits: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        device = idx.device
        b, t = idx.size()

        if past_key_values is not None:
            past_length = past_key_values[0][0].size(-2)
            pos = torch.arange(past_length, past_length + t, dtype=torch.long, device=device)
        else:
            past_length = 0
            pos = torch.arange(0, t, dtype=torch.long, device=device)

        if past_length + t > self.config.block_size:
            raise ValueError(
                f"Cannot forward sequence of length {past_length + t}, "
                f"block size is only {self.config.block_size}"
            )

        tok_emb = self.transformer.wte(idx)

        if self.config.use_rope:
            x = self.transformer.drop(tok_emb)
        else:
            pos_emb = self.transformer.wpe(pos)
            x = self.transformer.drop(tok_emb + pos_emb)

        presents = [] if use_cache else None
        for i, block in enumerate(self.transformer.h):
            layer_past = past_key_values[i] if past_key_values is not None else None
            x, present = block(x, use_cache=use_cache, layer_past=layer_past)
            if use_cache and present is not None:
                presents.append(present)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        elif return_all_logits:
            logits = self.lm_head(x)
            loss = None
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss, presents

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        """Estimate Model Flops Utilization (MFU) in units of A100/H100 bfloat16 peak FLOPS."""
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        return flops_achieved
