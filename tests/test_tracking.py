import json
import math

import pytest

from gpt_lab.tracking import Tracker, write_status


def _strict_rows(path) -> list[dict]:
    """Parse metrics.jsonl the way a browser would: NaN and Infinity are errors."""

    def reject(name):
        raise ValueError(f"invalid JSON constant {name}")

    return [json.loads(line, parse_constant=reject) for line in path.read_text().splitlines()]


def test_resume_drops_rows_after_checkpoint_and_partial_lines(tmp_path):
    path = tmp_path / "metrics.jsonl"
    tracker = Tracker(tmp_path / "tb", enabled=False, metrics_path=path)
    for step in range(1, 6):
        tracker.scalars({"train/loss": 1.0 / step}, step)
    tracker.close()
    with open(path, "a") as f:
        f.write('{"step": 6, "train/lo')  # cut off by a crash

    tracker = Tracker(tmp_path / "tb", enabled=False, metrics_path=path, start_step=3)
    tracker.scalars({"train/loss": 0.2}, 4)
    tracker.close()
    assert [r["step"] for r in _strict_rows(path)] == [1, 2, 3, 4]


def test_fresh_run_starts_with_empty_metrics(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"step": 1, "train/loss": 2.0}\n')
    Tracker(tmp_path / "tb", enabled=False, metrics_path=path).close()
    assert path.read_text() == ""


def test_non_finite_values_become_null_and_text_is_logged(tmp_path):
    path = tmp_path / "metrics.jsonl"
    tracker = Tracker(tmp_path / "tb", enabled=False, metrics_path=path)
    tracker.scalars({"train/loss": math.nan, "train/grad_norm": math.inf}, 1)
    tracker.text("samples/0", "Hello, world", 1)
    tracker.close()
    rows = _strict_rows(path)
    assert rows[0]["train/loss"] is None and rows[0]["train/grad_norm"] is None
    assert rows[1]["samples/0"] == "Hello, world"
    assert all(r["step"] == 1 and "time" in r for r in rows)


def test_write_status_replaces_file_atomically(tmp_path):
    path = tmp_path / "status.json"
    write_status(path, {"state": "running", "step": 1})
    write_status(path, {"state": "finished", "step": 2})
    assert json.loads(path.read_text()) == {"state": "finished", "step": 2}
    assert [p.name for p in tmp_path.iterdir()] == ["status.json"]  # no .tmp left behind


def test_write_status_never_raises(tmp_path, capsys):
    write_status(tmp_path / "missing_dir" / "status.json", {"state": "running"})
    assert "could not write" in capsys.readouterr().out


@pytest.mark.parametrize("enabled", [True, False])
def test_tracker_without_metrics_file_still_works(tmp_path, enabled):
    tracker = Tracker(tmp_path / "tb", enabled=enabled)
    tracker.scalars({"train/loss": 1.0}, 1)
    tracker.text("samples/0", "x", 1)
    tracker.flush()
    tracker.close()
    assert not (tmp_path / "metrics.jsonl").exists()
