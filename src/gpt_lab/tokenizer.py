"""GPT-2 byte-pair-encoding tokenizer (via tiktoken), shared by data prep, sampling and evals."""

from __future__ import annotations

from functools import lru_cache

GPT2_VOCAB_SIZE = 50257
EOT_TOKEN = 50256  # <|endoftext|>


@lru_cache(maxsize=1)
def get_tokenizer():
    import tiktoken

    return tiktoken.get_encoding("gpt2")


def encode(text: str) -> list[int]:
    return get_tokenizer().encode_ordinary(text)


def decode(tokens: list[int]) -> str:
    return get_tokenizer().decode(tokens)
