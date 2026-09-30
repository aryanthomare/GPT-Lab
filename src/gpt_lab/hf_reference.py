"""Wrap Hugging Face GPT-2 checkpoints so they share our model interface.

ref = HFCausalLM("gpt2")          # 124M;  also "gpt2-medium" (350M), "gpt2-large", ...
logits, loss = ref(token_ids, targets)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class HFCausalLM(nn.Module):
    def __init__(self, model_or_name: str | nn.Module):
        super().__init__()
        if isinstance(model_or_name, str):
            from transformers import AutoModelForCausalLM

            model_or_name = AutoModelForCausalLM.from_pretrained(model_or_name)
        self.model = model_or_name
        cfg = self.model.config
        self.seq_len = getattr(cfg, "n_positions", None) or cfg.max_position_embeddings

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        logits = self.model(input_ids=idx).logits
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.float().view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1
            )
        return logits, loss
