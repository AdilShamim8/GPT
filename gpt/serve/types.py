import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ChatMessage:
    role: str  # 'system', 'user', 'assistant'
    content: str


@dataclass
class ChatCompletionRequest:
    messages: List[Dict[str, str]]
    model: str = "nano-gpt"
    temperature: float = 0.8
    top_p: float = 0.95
    max_tokens: int = 256
    stream: bool = False
    repetition_penalty: float = 1.0


@dataclass
class CompletionRequest:
    prompt: str
    model: str = "nano-gpt"
    temperature: float = 0.8
    top_p: float = 0.95
    max_tokens: int = 256
    stream: bool = False
    repetition_penalty: float = 1.0


@dataclass
class ModelCard:
    id: str
    object: str = "model"
    created: int = field(default_factory=lambda: int(time.time()))
    owned_by: str = "nano-gpt"
