import torch

from gpt_lab.config import ModelConfig
from gpt_lab.model import Transformer, apply_rope, configure_optimizer, precompute_rope

from .conftest import VOCAB, tiny_model_config


def test_forward_shapes_and_loss(model_cfg):
    model = Transformer(model_cfg)
    idx = torch.randint(0, VOCAB, (3, 20))
    targets = torch.randint(0, VOCAB, (3, 20))
    logits, loss = model(idx, targets)
    assert logits.shape == (3, 20, VOCAB)
    assert loss.ndim == 0
    # Near-uniform predictions at initialization.
    assert abs(loss.item() - torch.log(torch.tensor(float(VOCAB))).item()) < 0.5


def test_124m_parameter_count():
    with torch.device("meta"):
        model = Transformer(ModelConfig())
    assert model.cfg.ffn_dim == 2048
    assert model.num_params() == 123_588_864


def test_causality(model_cfg):
    model = Transformer(model_cfg).eval()
    idx = torch.randint(0, VOCAB, (1, 16))
    changed = idx.clone()
    changed[0, 10:] = (changed[0, 10:] + 1) % VOCAB
    a, _ = model(idx)
    b, _ = model(changed)
    torch.testing.assert_close(a[:, :10], b[:, :10])
    assert not torch.allclose(a[:, 10:], b[:, 10:])


def test_rope_depends_only_on_relative_position():
    head_dim, T = 16, 40
    cos, sin = precompute_rope(head_dim, T)
    q = torch.randn(head_dim)
    k = torch.randn(head_dim)

    def score(m: int, n: int) -> torch.Tensor:
        x = torch.stack([q, k])[None, None]  # (1, 1, 2, head_dim)
        rq = apply_rope(x[:, :, :1], cos[m : m + 1], sin[m : m + 1])
        rk = apply_rope(x[:, :, 1:], cos[n : n + 1], sin[n : n + 1])
        return (rq * rk).sum()

    torch.testing.assert_close(score(5, 2), score(25, 22))
    torch.testing.assert_close(score(0, 0), (q * k).sum())


def test_grouped_query_attention_runs():
    cfg = tiny_model_config(n_kv_head=2)
    model = Transformer(cfg)
    logits, _ = model(torch.randint(0, VOCAB, (2, 8)))
    assert logits.shape == (2, 8, VOCAB)
    assert model.blocks[0].attn.wkv.weight.shape[0] == 2 * 2 * cfg.head_dim


def test_tied_embeddings(model_cfg):
    model = Transformer(model_cfg)
    assert model.lm_head.weight is model.tok_emb.weight
    untied = Transformer(tiny_model_config(tie_embeddings=False))
    assert untied.num_params() == model.num_params() + VOCAB * model_cfg.d_model


def test_activation_checkpointing_matches(model_cfg):
    idx = torch.randint(0, VOCAB, (2, 16))
    torch.manual_seed(1)
    plain = Transformer(model_cfg)
    torch.manual_seed(1)
    ckpt = Transformer(tiny_model_config(activation_checkpointing=True))
    _, l1 = plain(idx, idx)
    _, l2 = ckpt(idx, idx)
    l1.backward()
    l2.backward()
    torch.testing.assert_close(l1, l2)
    torch.testing.assert_close(plain.tok_emb.weight.grad, ckpt.tok_emb.weight.grad)


def test_overfits_single_batch(model_cfg):
    model = Transformer(model_cfg)
    opt = configure_optimizer(model, lr=1e-2, weight_decay=0.0)
    idx = torch.randint(0, VOCAB, (2, 17))
    x, y = idx[:, :-1], idx[:, 1:]
    for _ in range(400):
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.5


def test_weight_decay_groups(model_cfg):
    model = Transformer(model_cfg)
    opt = configure_optimizer(model, lr=1e-3, weight_decay=0.1)
    decay, no_decay = opt.param_groups
    assert all(p.dim() >= 2 for p in decay["params"])
    assert all(p.dim() == 1 for p in no_decay["params"])
    n_opt = sum(p.numel() for g in opt.param_groups for p in g["params"])
    assert n_opt == model.num_params()
