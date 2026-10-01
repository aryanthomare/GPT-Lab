"""Text generation from checkpoints, for the UI's Generate page."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import torch

from gpt_lab.checkpoint import load_model_from_checkpoint
from gpt_lab.generate import sample_text
from gpt_lab.tokenizer import encode


class Sampler:
    """Keeps the last checkpoint's model in memory, on the CPU between requests.

    A request moves the model to the GPU only when the GPU is free, and moves it back
    afterwards, so sampling never holds GPU memory that a training run needs.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._key: tuple[str, int] | None = None
        self._model: torch.nn.Module | None = None
        self._seq_len = 0

    def sample(
        self,
        checkpoint: Path,
        prompt: str,
        *,
        num_samples: int,
        max_new_tokens: int,
        temperature: float,
        top_k: int | None,
        top_p: float | None,
        seed: int,
        use_gpu: bool,
    ) -> dict[str, Any]:
        with self._lock:
            key = (str(checkpoint), checkpoint.stat().st_mtime_ns)
            if key != self._key:
                self._model = self._key = None  # let the old model go before loading
                model, cfg, _ = load_model_from_checkpoint(checkpoint, "cpu")
                self._model, self._seq_len, self._key = model, cfg.model.seq_len, key
            # An out-of-range token id is a device-side assert on the GPU, which breaks CUDA
            # for the whole server process, so check before running the model.
            vocab = self._model.cfg.vocab_size
            if max(encode(prompt), default=0) >= vocab:
                raise ValueError(
                    f"The prompt uses tokens this model doesn't know (its vocabulary has {vocab})."
                )
            device = "cuda" if use_gpu else "cpu"
            start = time.time()
            model = self._model.to(device)
            try:
                samples = [
                    sample_text(
                        model,
                        prompt,
                        seq_len=self._seq_len,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        device=device,
                        seed=seed + i,
                    )
                    for i in range(num_samples)
                ]
            finally:
                self._model.to("cpu")
                if device == "cuda":
                    torch.cuda.empty_cache()
            return {
                "prompt": prompt,
                "samples": samples,
                "device": device,
                "seconds": time.time() - start,
            }
