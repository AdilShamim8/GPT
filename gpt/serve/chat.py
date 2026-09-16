import time
from typing import Any, Dict, List
from gpt.inference.generator import TextGenerator
from gpt.serve.types import ChatCompletionRequest


def format_chat_messages(messages: List[Dict[str, str]]) -> str:
    """Format structured chat messages into autoregressive language model prompt."""
    formatted = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            formatted.append(f"System: {content}\n")
        elif role == "user":
            formatted.append(f"User: {content}\n")
        elif role == "assistant":
            formatted.append(f"Assistant: {content}\n")
    formatted.append("Assistant: ")
    return "\n".join(formatted)


def handle_chat_completion(
    generator: TextGenerator,
    request: ChatCompletionRequest,
) -> Dict[str, Any]:
    """Process OpenAI-compatible chat completion request."""
    prompt = format_chat_messages(request.messages)

    result = generator.generate(
        prompt=prompt,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        repetition_penalty=request.repetition_penalty,
    )
    # Extract only assistant's response
    assistant_reply = result.text[len(prompt) :].strip()

    return {
        "id": f"chatcmpl-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": assistant_reply,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": len(generator.tokenizer.encode(prompt)),
            "completion_tokens": result.num_generated,
            "total_tokens": len(generator.tokenizer.encode(result.text)),
        },
    }
