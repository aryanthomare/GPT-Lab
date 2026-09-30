"""Text sampling from a trained checkpoint.

python -m gpt_lab.generate --checkpoint runs/modern_124m --prompt "Once upon a time"
"""

from __future__ import annotations

import argparse

import torch

from gpt_lab.tokenizer import GPT2_VOCAB_SIZE


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    *,
    seq_len: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    vocab_limit: int | None = GPT2_VOCAB_SIZE,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Autoregressively extend idx (batch, time). temperature=0 means greedy decoding.

    vocab_limit drops the padding entries of the embedding table so they are never sampled.
    No key/value cache: fine for short samples at this model size.
    """
    was_training = model.training
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -seq_len:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :vocab_limit].float()
        if temperature == 0:
            next_tok = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k is not None and top_k < logits.size(-1):
                kth = torch.topk(logits, top_k, dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                cum = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                # Remove tokens once the cumulative probability *before* them exceeds top_p.
                remove = cum - sorted_logits.softmax(dim=-1) > top_p
                sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                logits = torch.full_like(logits, float("-inf")).scatter(
                    -1, sorted_idx, sorted_logits
                )
            probs = logits.softmax(dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1, generator=generator)
        idx = torch.cat([idx, next_tok.to(idx.dtype)], dim=1)
    model.train(was_training)
    return idx


def sample_text(
    model: torch.nn.Module,
    prompt: str,
    *,
    seq_len: int,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    top_k: int | None = 50,
    top_p: float | None = None,
    device: str | torch.device = "cpu",
    seed: int = 0,
) -> str:
    from gpt_lab.tokenizer import decode, encode

    gen = torch.Generator(device=device).manual_seed(seed)
    idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    out = generate(
        model,
        idx,
        max_new_tokens,
        seq_len=seq_len,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        generator=gen,
    )
    return decode(out[0].tolist())


def main(argv: list[str] | None = None) -> None:
    from gpt_lab.checkpoint import load_model_from_checkpoint
    from gpt_lab.train import resolve_device

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="checkpoint file or run directory")
    p.add_argument("--prompt", default="Hello, I'm a language model,")
    p.add_argument("--num-samples", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    device = resolve_device(args.device)
    model, cfg, _ = load_model_from_checkpoint(args.checkpoint, device)
    for i in range(args.num_samples):
        text = sample_text(
            model,
            args.prompt,
            seq_len=cfg.model.seq_len,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device,
            seed=args.seed + i,
        )
        print(f"--- sample {i} ---\n{text}\n")


if __name__ == "__main__":
    main()
