import json

import torch

from gpt_lab.checkpoint import (
    checkpoint_path,
    list_checkpoints,
    load_model_from_checkpoint,
    prune_checkpoints,
    save_checkpoint,
)
from gpt_lab.compare import build_table
from gpt_lab.config import Config
from gpt_lab.model import Transformer
from gpt_lab.train import train


def test_smoke_run_learns_and_writes_outputs(tiny_config):
    tiny_config.train.max_steps = 30
    tiny_config.train.ckpt_every = 10
    out = train(tiny_config)
    losses = out["losses"]
    assert len(losses) == 30
    assert sum(losses[-5:]) / 5 < sum(losses[:5]) / 5 - 0.5
    assert "val_loss" in out
    run_dir = tiny_config.train.run_dir
    assert (run_dir / "config.yaml").exists()
    assert list((run_dir / "tb").glob("events.out.tfevents.*"))
    steps = [s for s, _ in list_checkpoints(run_dir / "checkpoints")]
    assert steps == [20, 30]  # keep_last=2


def test_resume_is_exact(tiny_config, tmp_path):
    full = train(tiny_config)["losses"]

    tiny_config.train.run_name = "resumed"
    first = train(tiny_config, stop_at=3)["losses"]
    rest = train(tiny_config, resume=True)["losses"]
    assert len(first) == 3 and len(rest) == 3
    assert first + rest == full  # bit-identical on CPU


def test_load_model_from_run_dir(tiny_config):
    train(tiny_config)
    model, cfg, ckpt = load_model_from_checkpoint(tiny_config.train.run_dir)
    assert ckpt["step"] == tiny_config.train.max_steps
    assert cfg.model == tiny_config.model
    logits, _ = model(torch.zeros((1, 4), dtype=torch.long))
    assert logits.shape[-1] == tiny_config.model.vocab_size


def test_prune_keeps_milestones(tmp_path, model_cfg):
    model = Transformer(model_cfg)
    for step in [100, 200, 300, 400, 500]:
        save_checkpoint(
            checkpoint_path(tmp_path, step), model=model, optimizer=None, step=step, config=Config()
        )
    prune_checkpoints(tmp_path, keep_last=2, keep_every=200)
    assert [s for s, _ in list_checkpoints(tmp_path)] == [200, 400, 500]


def test_compare_table(tmp_path):
    rows = [
        {
            "name": "ours",
            "params": 123.6e6,
            "val_loss": 3.1,
            "hellaswag_acc_norm": 0.31,
            "lm_eval": {"piqa": {"acc_norm,none": 0.62}},
        },
        {"name": "hf:gpt2", "params": 124.4e6, "val_loss": 3.29},
    ]
    table = build_table(json.loads(json.dumps(rows)))
    assert "| ours | 123.6 | 3.1000 | 0.3100 | 0.6200 |" in table
    assert "hf:gpt2" in table
