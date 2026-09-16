import time
from typing import Any, Dict
from gpt.inference.generator import TextGenerator
from gpt.serve.types import CompletionRequest


def handle_completion(
    generator: TextGenerator,
    request: CompletionRequest,
) -> Dict[str, Any]:
    """Process OpenAI-compatible text completion request."""
    t0 = time.time()
    result = generator.generate(
        prompt=request.prompt,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        repetition_penalty=request.repetition_penalty,
    )
    completion_text = result.text[len(request.prompt) :]

    return {
        "id": f"cmpl-{int(time.time() * 1000)}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "text": completion_text,
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": len(generator.tokenizer.encode(request.prompt)),
            "completion_tokens": result.num_generated,
            "total_tokens": len(generator.tokenizer.encode(result.text)),
        },
    }
