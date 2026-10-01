import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

from gpt_lab.checkpoint import (
    checkpoint_path,
    list_checkpoints,
    load_model_from_checkpoint,
    prune_checkpoints,
    save_checkpoint,
)
from gpt_lab.compare import build_table
from gpt_lab.config import Config, save_config
from gpt_lab.model import Transformer
from gpt_lab.train import train


def _status(run_dir: Path) -> dict:
    return json.loads((run_dir / "status.json").read_text())


def _logged_train_steps(run_dir: Path) -> list[int]:
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    return [r["step"] for r in rows if "train/loss" in r]


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

    assert _logged_train_steps(run_dir) == list(range(1, 31))
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert any("eval/val_loss" in r for r in rows)
    status = _status(run_dir)
    assert status["state"] == "finished"
    assert status["step"] == 30
    assert status["last_checkpoint"] == "checkpoints/step_000030.pt"
    assert status["loss"] == losses[-1]  # the latest step, for live views
    assert status["lr"] is not None and status["step_seconds"] > 0


def test_resume_is_exact(tiny_config, tmp_path):
    full = train(tiny_config)["losses"]

    tiny_config.train.run_name = "resumed"
    run_dir = tiny_config.train.run_dir
    first = train(tiny_config, stop_at=3)["losses"]
    assert _status(run_dir)["state"] == "stopped"
    rest = train(tiny_config, resume=True)["losses"]
    assert len(first) == 3 and len(rest) == 3
    assert first + rest == full  # bit-identical on CPU
    assert _status(run_dir)["state"] == "finished"
    assert _logged_train_steps(run_dir) == [1, 2, 3, 4, 5, 6]


@pytest.mark.skipif(sys.platform == "win32", reason="Windows cannot catch SIGTERM")
def test_sigterm_saves_checkpoint_and_marks_interrupted(tiny_config, tmp_path):
    t = tiny_config.train
    t.max_steps, t.ckpt_every, t.eval_every = 100_000, 0, 0  # only the interrupt saves
    cfg_path = tmp_path / "config.yaml"
    save_config(tiny_config, cfg_path)
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    log_path = tmp_path / "train.log"
    with open(log_path, "w") as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "gpt_lab.train", "--config", str(cfg_path)],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )
        try:
            deadline = time.time() + 120
            while not ((t.run_dir / "status.json").exists() and _status(t.run_dir)["step"] >= 3):
                assert proc.poll() is None, log_path.read_text()
                assert time.time() < deadline, "training did not reach step 3"
                time.sleep(0.1)
            proc.send_signal(signal.SIGTERM)
            assert proc.wait(timeout=60) == 0, log_path.read_text()
        finally:
            if proc.poll() is None:
                proc.kill()
    status = _status(t.run_dir)
    assert status["state"] == "interrupted"
    assert [s for s, _ in list_checkpoints(t.run_dir / "checkpoints")] == [status["step"]]


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
