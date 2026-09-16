import json
import time
from typing import Any, Dict, List
from gpt.model.gpt import GPT
from gpt.serve.types import ModelCard


class ModelRegistry:
    """Tracks active loaded models for API exposure."""

    def __init__(self):
        self._models: Dict[str, GPT] = {}

    def register(self, model_id: str, model: GPT) -> None:
        self._models[model_id] = model

    def get(self, model_id: str) -> GPT:
        if model_id not in self._models:
            # Fallback to first registered model if available
            if self._models:
                return next(iter(self._models.values()))
            raise KeyError(f"No model registered with id '{model_id}'")
        return self._models[model_id]

    def list_cards(self) -> List[Dict[str, Any]]:
        cards = []
        for name, m in self._models.items():
            card = {
                "id": name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "nano-gpt-prod",
                "parameters": m.get_num_params(),
                "vocab_size": m.config.vocab_size,
                "block_size": m.config.block_size,
            }
            cards.append(card)
        return cards
