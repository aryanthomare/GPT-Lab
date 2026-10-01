import json

import pytest
import torch

from gpt_lab.evals.hellaswag import evaluate_hellaswag, parquet_to_jsonl, render_example
from gpt_lab.evals.val_loss import evaluate_val_loss
from gpt_lab.generate import generate
from gpt_lab.model import Transformer

from .conftest import VOCAB


def char_encode(text: str) -> list[int]:
    return [ord(c) % VOCAB for c in text]


class PreferTokenModel(torch.nn.Module):
    """Stub LM that strongly predicts `token` at every position."""

    def __init__(self, token: int):
        super().__init__()
        self.token = token

    def forward(self, idx, targets=None):
        logits = torch.zeros(*idx.shape, VOCAB)
        logits[..., self.token] = 10.0
        return logits, None


EXAMPLE = {"ctx": "ab", "endings": ["cd", "zzzz", "ef", "gh"], "label": 1}


def test_render_example():
    tokens, mask, label = render_example(EXAMPLE, char_encode)
    assert label == 1
    assert tokens.shape == (4, 2 + 5)  # "ab" + " zzzz"
    assert mask[1].tolist() == [0, 0, 1, 1, 1, 1, 1]
    assert mask[0].tolist() == [0, 0, 1, 1, 1, 0, 0]  # padding is masked out


def test_hellaswag_picks_preferred_ending(tmp_path):
    path = tmp_path / "hs.jsonl"
    path.write_text(json.dumps(EXAMPLE) + "\n")
    model = PreferTokenModel(char_encode("z")[0])
    out = evaluate_hellaswag(model, path, encode=char_encode)
    assert out == {"hellaswag_acc": 1.0, "hellaswag_acc_norm": 1.0, "hellaswag_n": 1}


def test_hellaswag_parquet_converts_to_scoreable_jsonl(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    # The Hugging Face copy's layout: extra columns, and the label stored as a string.
    table = pa.table(
        {
            "ind": [7],
            "activity_label": ["x"],
            "ctx_a": ["a"],
            "ctx_b": ["b"],
            "ctx": [EXAMPLE["ctx"]],
            "endings": [EXAMPLE["endings"]],
            "label": ["1"],
        }
    )
    pq.write_table(table, tmp_path / "val.parquet")
    assert parquet_to_jsonl(tmp_path / "val.parquet", tmp_path / "val.jsonl") == 1
    row = json.loads((tmp_path / "val.jsonl").read_text())
    assert row == {"ind": 7, **EXAMPLE}
    model = PreferTokenModel(char_encode("z")[0])
    assert (
        evaluate_hellaswag(model, tmp_path / "val.jsonl", encode=char_encode)["hellaswag_acc"]
        == 1.0
    )


def test_val_loss_is_deterministic(shard_dir, model_cfg):
    model = Transformer(model_cfg)
    kw = dict(batch_size=2, seq_len=32, max_tokens=10_000, device="cpu")
    a = evaluate_val_loss(model, str(shard_dir / "val_*.npy"), **kw)
    b = evaluate_val_loss(model, str(shard_dir / "val_*.npy"), **kw)
    assert a == b
    # 20 * 64 tokens -> 39 chunks of 32 -> 19 full batches of 2
    assert a["val_tokens"] == 19 * 2 * 32


def test_generate_greedy_and_sampling(model_cfg):
    model = PreferTokenModel(5)
    idx = torch.zeros((2, 3), dtype=torch.long)
    out = generate(model, idx, 4, seq_len=8, temperature=0, vocab_limit=None)
    assert out.shape == (2, 7)
    assert (out[:, 3:] == 5).all()

    real = Transformer(model_cfg)
    g = torch.Generator().manual_seed(0)
    out = generate(
        real,
        idx,
        40,
        seq_len=model_cfg.seq_len,
        top_k=5,
        top_p=0.9,
        vocab_limit=VOCAB - 10,
        generator=g,
    )
    assert out.shape == (2, 43)
    assert out.max() < VOCAB - 10  # padded vocabulary entries are never sampled


def test_hf_reference_wrapper():
    transformers = pytest.importorskip("transformers")
    from gpt_lab.hf_reference import HFCausalLM

    hf = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(n_layer=1, n_head=2, n_embd=16, vocab_size=VOCAB, n_positions=32)
    )
    ref = HFCausalLM(hf).eval()
    idx = torch.randint(0, VOCAB, (2, 10))
    logits, loss = ref(idx[:, :-1], idx[:, 1:])
    assert logits.shape == (2, 9, VOCAB)
    assert ref.seq_len == 32
    expected = hf(input_ids=idx[:, :-1], labels=idx[:, :-1]).loss  # HF shifts internally
    _, loss_same = ref(idx[:, :-1][:, :-1], idx[:, :-1][:, 1:])
    torch.testing.assert_close(loss_same, expected)
