from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F


class DataCollatorWithPadding:
    """Collator that pads sequences to the maximum length in the micro-batch."""

    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id

    def __call__(
        self,
        batch: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        xs, ys = zip(*batch)
        max_len_x = max(len(x) for x in xs)
        max_len_y = max(len(y) for y in ys)

        padded_xs = []
        for x in xs:
            pad_amount = max_len_x - len(x)
            if pad_amount > 0:
                padded = F.pad(x, (0, pad_amount), value=self.pad_token_id)
            else:
                padded = x
            padded_xs.append(padded)

        padded_ys = []
        for y in ys:
            pad_amount = max_len_y - len(y)
            if pad_amount > 0:
                # -1 indicates cross-entropy loss should ignore this index
                padded = F.pad(y, (0, pad_amount), value=-1)
            else:
                padded = y
            padded_ys.append(padded)

        return torch.stack(padded_xs), torch.stack(padded_ys)
