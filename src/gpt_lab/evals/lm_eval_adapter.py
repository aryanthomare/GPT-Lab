"""Adapter so EleutherAI's lm-evaluation-harness can score any model with our interface.

Using the same adapter for our checkpoints and for Hugging Face GPT-2 (via HFCausalLM)
means both are scored by identical code. Requires `pip install -e ".[eval]"`.

Suggested zero-shot tasks: arc_easy, arc_challenge, piqa, winogrande, lambada_openai.
Unbatched and simple: fine for 124M-350M models on a GPU (minutes per task).
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
import torch.nn.functional as F

from gpt_lab.tokenizer import EOT_TOKEN, GPT2_VOCAB_SIZE, decode, encode

try:
    from lm_eval.api.model import LM
except ImportError:  # keep this module importable without lm-eval installed
    LM = object


class GPTLabLM(LM):
    def __init__(
        self,
        model: torch.nn.Module,
        seq_len: int,
        device: str | torch.device = "cuda",
        autocast_ctx=None,
        vocab_limit: int = GPT2_VOCAB_SIZE,
    ):
        if LM is object:
            raise ImportError("lm-eval is not installed: pip install -e '.[eval]'")
        super().__init__()
        self.model = model.eval()
        self.seq_len = seq_len
        self.device = device
        self.ctx = autocast_ctx
        self.vocab_limit = vocab_limit

    @torch.no_grad()
    def _logprobs(self, tokens: list[int]) -> torch.Tensor:
        """Log-probabilities (len(tokens), vocab) of the next token at each position."""
        idx = torch.tensor([tokens], dtype=torch.long, device=self.device)
        with self.ctx or nullcontext():
            logits, _ = self.model(idx)
        return F.log_softmax(logits[0, :, : self.vocab_limit].float(), dim=-1)

    def loglikelihood(self, requests) -> list[tuple[float, bool]]:
        results = []
        for req in requests:
            context, continuation = req.args
            ctx_tokens = encode(context) if context else [EOT_TOKEN]
            cont_tokens = encode(continuation)
            full = (ctx_tokens + cont_tokens)[-(self.seq_len + 1) :]
            inp = full[:-1]
            logprobs = self._logprobs(inp)[-len(cont_tokens) :]
            target = torch.tensor(cont_tokens, device=logprobs.device)
            ll = logprobs.gather(1, target[:, None]).sum().item()
            greedy = bool((logprobs.argmax(dim=-1) == target).all().item())
            results.append((ll, greedy))
        return results

    def loglikelihood_rolling(self, requests) -> list[float]:
        results = []
        for req in requests:
            (text,) = req.args
            tokens = [EOT_TOKEN] + encode(text)
            total, s = 0.0, 1
            while s < len(tokens):
                e = min(s + self.seq_len, len(tokens))  # score targets tokens[s:e]
                a = max(0, e - 1 - self.seq_len)  # input tokens[a:e-1]
                logprobs = self._logprobs(tokens[a : e - 1])
                rows = torch.arange(s - 1 - a, e - 1 - a, device=logprobs.device)
                target = torch.tensor(tokens[s:e], device=logprobs.device)
                total += logprobs[rows, target].sum().item()
                s = e
            results.append(total)
        return results

    def generate_until(self, requests) -> list[str]:
        results = []
        for req in requests:
            context, gen_kwargs = req.args
            until: list[str] = gen_kwargs.get("until", []) or []
            if isinstance(until, str):
                until = [until]
            max_new = int(gen_kwargs.get("max_gen_toks", 256))
            tokens = encode(context) if context else [EOT_TOKEN]
            out: list[int] = []
            text = ""
            for _ in range(max_new):
                logprobs = self._logprobs((tokens + out)[-self.seq_len :])
                nxt = int(logprobs[-1].argmax().item())
                if nxt == EOT_TOKEN:
                    break
                out.append(nxt)
                text = decode(out)
                if any(stop in text for stop in until):
                    break
            for stop in until:
                text = text.split(stop)[0]
            results.append(text)
        return results


def run_lm_eval(
    model: torch.nn.Module,
    tasks: list[str],
    *,
    seq_len: int,
    device: str | torch.device,
    autocast_ctx=None,
    limit: int | None = None,
) -> dict[str, Any]:
    import lm_eval

    lm = GPTLabLM(model, seq_len, device, autocast_ctx)
    out = lm_eval.simple_evaluate(
        model=lm, tasks=tasks, num_fewshot=0, batch_size=1, limit=limit, log_samples=False
    )
    return out["results"]
