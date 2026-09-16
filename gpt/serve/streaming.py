import json
import time
from typing import Generator
from gpt.inference.generator import TextGenerator
from gpt.serve.chat import format_chat_messages
from gpt.serve.types import ChatCompletionRequest


def stream_chat_completion_sse(
    generator: TextGenerator,
    request: ChatCompletionRequest,
) -> Generator[str, None, None]:
    """Yield Server-Sent Events (SSE) streaming data chunks formatted as OpenAI response."""
    prompt = format_chat_messages(request.messages)
    req_id = f"chatcmpl-{int(time.time() * 1000)}"

    # Initial role announcement chunk
    first_chunk = {
        "id": req_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(first_chunk)}\n\n"

    # Streaming token delta chunks
    for token_str in generator.generate_stream(
        prompt=prompt,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        repetition_penalty=request.repetition_penalty,
    ):
        chunk = {
            "id": req_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{"index": 0, "delta": {"content": token_str}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    # Final termination chunk
    final_chunk = {
        "id": req_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"
