import math

import pytest

from gpt_lab.config import Config, load_config, save_config
from gpt_lab.schedule import lr_at


def test_load_yaml_and_overrides(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("model:\n  n_layer: 4\ntrain:\n  lr: 6e-4\n  compile: true\n")
    cfg = load_config(path, ["--train.lr=1e-3", "train.compile=false", "model.n_kv_head=4"])
    assert cfg.model.n_layer == 4
    assert cfg.train.lr == pytest.approx(1e-3) and isinstance(cfg.train.lr, float)
    assert cfg.train.compile is False
    assert cfg.model.n_kv_head == 4


def test_yaml_scientific_notation_without_dot_is_float(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("train:\n  lr: 6e-4\n")
    assert load_config(path).train.lr == pytest.approx(6e-4)


def test_unknown_keys_rejected(tmp_path):
    with pytest.raises(KeyError):
        load_config(None, ["train.not_a_key=1"])
    path = tmp_path / "c.yaml"
    path.write_text("bogus:\n  a: 1\n")
    with pytest.raises(KeyError):
        load_config(path)


def test_save_roundtrip(tmp_path):
    cfg = load_config(None, ["train.run_name=abc", "model.d_model=128"])
    save_config(cfg, tmp_path / "c.yaml")
    again = load_config(tmp_path / "c.yaml")
    assert again.to_dict() == cfg.to_dict()
    assert Config.from_dict(cfg.to_dict()).to_dict() == cfg.to_dict()


KW = dict(max_lr=1.0, total_steps=100, warmup_steps=10)


def test_warmup():
    assert lr_at(0, **KW) == pytest.approx(0.1)
    assert lr_at(9, **KW) == pytest.approx(1.0)


def test_wsd():
    kw = dict(KW, schedule="wsd", decay_frac=0.2)
    assert lr_at(10, **kw) == 1.0
    assert lr_at(79, **kw) == 1.0  # decay covers steps 80..99
    assert lr_at(80, **kw) == pytest.approx(1 - 1 / 20)
    assert lr_at(99, **kw) == pytest.approx(0.0)
    kw["min_lr_ratio"] = 0.1
    assert lr_at(99, **kw) == pytest.approx(0.1)


def test_cosine():
    kw = dict(KW, schedule="cosine", min_lr_ratio=0.1)
    assert lr_at(10, **kw) == pytest.approx(1.0)
    assert lr_at(55, **kw) == pytest.approx(0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * 0.5)))
    assert lr_at(100, **kw) == pytest.approx(0.1)


def test_unknown_schedule():
    with pytest.raises(ValueError):
        lr_at(50, **KW, schedule="nope")
