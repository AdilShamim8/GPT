from gpt.evaluation.benchmark import BenchmarkReport, benchmark_generation_speed
from gpt.evaluation.hellaswag import evaluate_multiple_choice_example
from gpt.evaluation.perplexity import calculate_perplexity

__all__ = [
    "BenchmarkReport",
    "benchmark_generation_speed",
    "calculate_perplexity",
    "evaluate_multiple_choice_example",
]
