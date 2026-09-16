import time
from dataclasses import dataclass
from typing import Dict, List
import torch
from gpt.inference.generator import TextGenerator
from gpt.model.gpt import GPT
from gpt.model.utils import estimate_model_memory_mb
from gpt.tokenizer.base import BaseTokenizer


@dataclass
class BenchmarkReport:
    tokens_per_second: float
    time_to_first_token_ms: float
    inter_token_latency_ms: float
    total_time_seconds: float
    total_tokens_generated: int
    model_memory_mb: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "tokens_per_second": round(self.tokens_per_second, 2),
            "time_to_first_token_ms": round(self.time_to_first_token_ms, 2),
            "inter_token_latency_ms": round(self.inter_token_latency_ms, 2),
            "total_time_seconds": round(self.total_time_seconds, 3),
            "total_tokens_generated": self.total_tokens_generated,
            "model_memory_mb": round(self.model_memory_mb, 2),
        }


def benchmark_generation_speed(
    model: GPT,
    tokenizer: BaseTokenizer,
    prompt: str = "Once upon a time in a ancient kingdom",
    num_tokens: int = 100,
    device: str = "cpu",
) -> BenchmarkReport:
    """Benchmark inference throughput, TTFT, and inter-token latency."""
    generator = TextGenerator(model=model, tokenizer=tokenizer, device=device)

    # Warmup pass
    for _ in generator.generate_stream(prompt, max_new_tokens=5):
        pass

    # Timed pass
    t_start = time.perf_counter()
    token_timestamps = []

    for _ in generator.generate_stream(prompt, max_new_tokens=num_tokens):
        token_timestamps.append(time.perf_counter())

    t_end = time.perf_counter()
    total_time = t_end - t_start

    n_tokens = len(token_timestamps)
    if n_tokens == 0:
        ttft = 0.0
        itl = 0.0
        tok_per_sec = 0.0
    else:
        ttft = (token_timestamps[0] - t_start) * 1000.0  # ms
        deltas = [
            (t2 - t1) * 1000.0 for t1, t2 in zip(token_timestamps[:-1], token_timestamps[1:])
        ]
        itl = sum(deltas) / len(deltas) if deltas else 0.0
        tok_per_sec = n_tokens / total_time if total_time > 0 else 0.0

    mem_mb = estimate_model_memory_mb(model)

    return BenchmarkReport(
        tokens_per_second=tok_per_sec,
        time_to_first_token_ms=ttft,
        inter_token_latency_ms=itl,
        total_time_seconds=total_time,
        total_tokens_generated=n_tokens,
        model_memory_mb=mem_mb,
    )
