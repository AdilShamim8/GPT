import json
from pathlib import Path
from typing import Dict, List, Tuple, Union
import torch
from torch.utils.data import Dataset
from gpt.tokenizer.base import BaseTokenizer


class InstructionDataset(Dataset):
    """Instruction fine-tuning dataset with prompt loss masking."""

    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: BaseTokenizer,
        block_size: int = 512,
    ):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.examples: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for item in data:
            instruction = item.get("instruction") or item.get("prompt", "")
            response = item.get("response") or item.get("completion", "")
            if not instruction or not response:
                continue

            full_prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
            prompt_ids = tokenizer.encode(full_prompt)
            resp_ids = tokenizer.encode(response)

            eos_id = tokenizer.eos_token_id or 0
            all_ids = prompt_ids + resp_ids + [eos_id]

            if len(all_ids) < 2:
                continue

            # Truncate if longer than block_size + 1
            all_ids = all_ids[: block_size + 1]

            x = torch.tensor(all_ids[:-1], dtype=torch.long)
            y = torch.tensor(all_ids[1:], dtype=torch.long)

            # Mask out prompt positions in target y with -1 (no loss computed on prompt tokens)
            prompt_len = min(len(prompt_ids) - 1, len(y))
            if prompt_len > 0:
                y[:prompt_len] = -1

            self.examples.append((x, y))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.examples[idx]

    @classmethod
    def from_jsonl(
        cls,
        path: Union[str, Path],
        tokenizer: BaseTokenizer,
        block_size: int = 512,
    ) -> "InstructionDataset":
        """Load instruction dataset from JSONL file."""
        path = Path(path)
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return cls(data=data, tokenizer=tokenizer, block_size=block_size)
