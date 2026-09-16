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
    """High-performance autoregressive text generation engine."""

    def __init__(self, model: GPT, tokenizer: BaseTokenizer, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(self.device)
        self.model.eval()

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

        # Prefill prompt with KV cache
        logits, _, past_kv = self.model(idx, use_cache=True)

        for _ in range(max_new_tokens):
            last_logits = logits[:, -1, :].clone()

            # Apply repetition penalty
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

            # Check for EOS token
            if self.tokenizer.eos_token_id is not None and token_id == self.tokenizer.eos_token_id:
                break

            token_str = self.tokenizer.decode([token_id])
            yield token_str

            # Fast incremental step using single token and cached KV
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
        tokens = []
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
