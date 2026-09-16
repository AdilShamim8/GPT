import collections
import time
from typing import Generator, List, Optional
import torch
from gpt.config.inference_config import InferenceConfig
from gpt.inference.base import BaseGenerator, GenerationResult
from gpt.inference.penalties import apply_frequency_presence_penalty, apply_repetition_penalty
from gpt.inference.sampling import sample_token
from gpt.model.gpt import GPT
from gpt.tokenizer.base import BaseTokenizer


class TextGenerator(BaseGenerator):
    """High-performance autoregressive text generation engine with batch and streaming support."""

    def __init__(self, model: GPT, tokenizer: BaseTokenizer, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        top_p: float = 0.95,
    ) -> List[str]:
        """Generate completions for a list of prompts in parallel batch."""
        tokenized = [self.tokenizer.encode(p) for p in prompts]
        max_prompt_len = max(len(t) for t in tokenized)
        pad_id = self.tokenizer.pad_token_id or 0

        # Left-pad prompts for causal generation
        padded_batch = []
        for t in tokenized:
            pad_len = max_prompt_len - len(t)
            padded = [pad_id] * pad_len + t
            padded_batch.append(padded)

        idx = torch.tensor(padded_batch, dtype=torch.long, device=self.device)

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.model.config.block_size :]
            logits, _, _ = self.model(idx_cond)
            last_logits = logits[:, -1, :]
            next_tokens = sample_token(
                last_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            idx = torch.cat([idx, next_tokens], dim=1)

        results = []
        for row in idx:
            text = self.tokenizer.decode(row.tolist())
            results.append(text)
        return results

    @torch.no_grad()
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
    ) -> Generator[str, None, None]:
        """Stream generated text token by token."""
        prompt_ids = self.tokenizer.encode(prompt)
        if not prompt_ids:
            prompt_ids = [0]

        idx = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        generated_ids: List[int] = []

        logits, _, past_kv = self.model(idx, use_cache=True)

        for _ in range(max_new_tokens):
            last_logits = logits[:, -1, :].clone()

            if repetition_penalty != 1.0 and generated_ids:
                last_logits = apply_repetition_penalty(
                    last_logits, generated_ids, penalty=repetition_penalty
                )

            next_token = sample_token(
                last_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            token_id = next_token.item()
            generated_ids.append(token_id)

            if self.tokenizer.eos_token_id is not None and token_id == self.tokenizer.eos_token_id:
                break

            token_str = self.tokenizer.decode([token_id])
            yield token_str

            logits, _, past_kv = self.model(
                next_token, use_cache=True, past_key_values=past_kv
            )

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
    ) -> GenerationResult:
        """Run complete generation and return full result with telemetry."""
        t0 = time.time()
        chunks = []
        for piece in self.generate_stream(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        ):
            chunks.append(piece)

        full_text = prompt + "".join(chunks)
        elapsed = time.time() - t0
        return GenerationResult(
            text=full_text,
            token_ids=self.tokenizer.encode(full_text),
            num_generated=len(chunks),
            elapsed_seconds=elapsed,
        )
