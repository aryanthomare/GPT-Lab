import json
import sys
import time

import numpy as np
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from gpt_lab.config import save_config  # noqa: E402
from gpt_lab.data import write_shard  # noqa: E402
from gpt_lab.ui.app import create_app  # noqa: E402
from gpt_lab.ui.jobs import JobError, JobManager  # noqa: E402
from gpt_lab.ui.telemetry import parse_line  # noqa: E402

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="the UI runs jobs as POSIX sessions"
)


def wait_for(fn, timeout=120.0, what="condition"):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = fn()
        if value:
            return value
        time.sleep(0.2)
    raise AssertionError(f"timed out waiting for {what}")


@pytest.fixture
def lab(tmp_path, tiny_config):
    """A GPT-Lab root whose only preset trains the tiny test model on the test shards."""
    root = tmp_path / "lab"
    (root / "configs").mkdir(parents=True)
    tiny_config.train.device = "cpu"
    save_config(tiny_config, root / "configs" / "tiny_test.yaml")
    return root


@pytest.fixture
def client(lab):
    app = create_app(lab)
    with TestClient(app, base_url="http://localhost") as c:
        yield c
    for job in app.state.jobs.running():  # never leave a test's job running
        app.state.jobs.kill(job["id"])


def start(client, name, **overrides):
    r = client.post(
        "/api/runs", json={"preset": "tiny_test", "run_name": name, "overrides": overrides}
    )
    assert r.status_code == 200, r.text
    return r.json()


def run_state(client, name):
    return client.get(f"/api/runs/{name}").json()


def test_gpu_history_endpoint_and_parsing(client):
    out = client.get("/api/system/history", params={"since": 0}).json()
    assert isinstance(out["samples"], list) and out["interval"] == 2  # empty without a GPU
    line = "NVIDIA GeForce RTX 5080, 97, 13021, 16303, 68, [N/A]"
    sample = parse_line(line, 123.0)
    assert sample["util"] == 97 and sample["mem_total_mb"] == 16303 and sample["power_w"] is None
    assert parse_line("not, enough", 1.0) is None


def test_rejects_requests_for_other_hosts(lab):
    with TestClient(create_app(lab), base_url="http://evil.example") as c:
        assert c.get("/api/health").status_code == 400


def test_presets_and_schema(client):
    presets = client.get("/api/presets").json()
    assert [p["name"] for p in presets] == ["tiny_test"]
    assert presets[0]["config"]["model"]["d_model"] == 32
    schema = client.get("/api/schema").json()
    keys = {f["key"]: f for fields in schema.values() for f in fields}
    assert keys["model.n_layer"]["type"] == "int"
    assert keys["model.n_kv_head"]["optional"] is True
    assert keys["train.schedule"]["choices"] == ["wsd", "cosine"]
    assert keys["train.sample_prompts"]["type"] == "list"
    assert "train.run_name" not in keys and "train.out_dir" not in keys


def test_check_reports_numbers_and_problems(client):
    def check(name="a", **overrides):
        body = {"preset": "tiny_test", "run_name": name, "overrides": overrides}
        return client.post("/api/runs/check", json=body).json()

    ok = check()
    assert ok["errors"] == []
    assert ok["derived"]["grad_accum"] == 2 and ok["derived"]["params"] > 0
    assert ok["derived"]["train_shards"] == 3
    assert ok["config"]["train"]["out_dir"] == "runs"
    assert "divisible" in check(**{"model.n_head": 5})["errors"][0]
    assert "model.bogus" in check(**{"model.bogus": 1})["errors"][0]
    assert "multiple" in check(**{"train.micro_batch_size": 3})["errors"][0]
    assert "Run names" in check(name="bad name!")["errors"][0]
    assert "whole number" in check(**{"train.max_steps": "abc"})["errors"][0]
    assert "one of" in check(**{"train.schedule": "linear"})["errors"][0]
    assert "No training shards" in check(**{"data.train_pattern": "nope/*.npy"})["errors"][0]


def test_training_run_end_to_end(client):
    start(client, "e2e")
    assert (
        client.post("/api/runs", json={"preset": "tiny_test", "run_name": "e2e"}).status_code == 400
    )
    run = wait_for(
        lambda: (r := run_state(client, "e2e"))["state"] == "finished" and r, what="finish"
    )
    assert run["step"] == 6 and run["loss"] is not None and run["val_loss"] is not None
    assert [c["step"] for c in run["checkpoints"]] == [3, 6]
    assert run["model_stats"]["params"] > 0

    first = client.get("/api/runs/e2e/metrics").json()
    assert [r["step"] for r in first["rows"] if "train/loss" in r] == [1, 2, 3, 4, 5, 6]
    again = client.get("/api/runs/e2e/metrics", params={"cursor": first["cursor"]}).json()
    assert again["rows"] == [] and again["reset"] is False
    assert "saved checkpoint" in client.get("/api/runs/e2e/log").json()["text"]

    listed = client.get("/api/runs").json()
    assert listed[0]["name"] == "e2e" and listed[0]["state"] == "finished"
    assert [p[0] for p in listed[0]["loss_trend"]] == [1, 2, 3, 4, 5, 6]  # for the sparkline
    # The run reports "finished" just before its process exits and the job records that.
    job = wait_for(lambda: (j := client.get("/api/jobs").json()[0])["state"] != "running" and j)
    assert job["state"] == "succeeded" and job["run"] == "e2e"
    assert client.post("/api/runs/e2e/resume").status_code == 409  # already finished

    # The test model knows 64 token ids; "!" is GPT-2 token 0, while "ab" is token 397.
    body = {"run": "e2e", "prompt": "!", "max_new_tokens": 5}
    out = client.post("/api/generate", json=body).json()
    assert len(out["samples"]) == 1 and out["checkpoint"] == "step_000006.pt"
    r = client.post("/api/generate", json={**body, "prompt": "ab"})
    assert r.status_code == 400 and "vocabulary has 64" in r.json()["detail"]
    assert client.post("/api/generate", json=body).status_code == 200  # still healthy


def test_stop_saves_a_checkpoint_and_resume_continues(client):
    start(
        client, "long", **{"train.max_steps": 100_000, "train.ckpt_every": 0, "train.eval_every": 0}
    )
    wait_for(lambda: run_state(client, "long")["step"] >= 3, what="step 3")
    assert client.post("/api/runs/long/stop").status_code == 200
    run = wait_for(
        lambda: (r := run_state(client, "long"))["state"] == "interrupted" and r, what="stop"
    )
    stopped_at = run["step"]
    assert [c["step"] for c in run["checkpoints"]] == [stopped_at]
    wait_for(lambda: client.get("/api/jobs").json()[0]["state"] == "stopped", what="job stopped")

    assert client.post("/api/runs/long/resume").status_code == 200
    wait_for(
        lambda: run_state(client, "long")["step"] > stopped_at + 2, what="progress after resume"
    )
    assert client.post("/api/runs/long/resume").status_code == 409  # already running
    client.post("/api/runs/long/stop")
    wait_for(lambda: run_state(client, "long")["state"] == "interrupted", what="second stop")
    steps = [
        r["step"] for r in client.get("/api/runs/long/metrics").json()["rows"] if "train/loss" in r
    ]
    assert steps == sorted(set(steps))  # the resumed run never repeats a step


def test_job_records_exit_code_and_output(lab):
    jobs = JobManager(lab)
    job = jobs.start("test", "Fails", [sys.executable, "-c", "print('hello'); raise SystemExit(3)"])
    done = wait_for(lambda: (j := jobs.get(job["id"]))["state"] != "running" and j, what="job end")
    assert done["state"] == "failed" and done["exit_code"] == 3
    assert "hello" in open(done["log"]).read()


def test_gpu_jobs_run_one_at_a_time(lab):
    jobs = JobManager(lab)
    sleeper = [sys.executable, "-c", "import time; time.sleep(60)"]
    first = jobs.start("test", "First GPU job", sleeper, gpu=True)
    with pytest.raises(JobError, match="First GPU job"):
        jobs.start("test", "Second GPU job", sleeper, gpu=True)
    jobs.start("test", "CPU job", [sys.executable, "-c", "pass"])  # CPU jobs don't wait
    jobs.stop(first["id"])
    assert wait_for(lambda: jobs.get(first["id"])["state"] == "stopped", what="stop")


def test_data_summary_and_prepare_guards(client, lab):
    write_shard(lab / "data" / "tiny" / "train_000000.npy", np.arange(1000) % 64)
    write_shard(lab / "data" / "tiny" / "val_000000.npy", np.arange(100) % 64)
    tiny = next(d for d in client.get("/api/data").json()["datasets"] if d["key"] == "tiny")
    assert tiny["train_shards"] == 1 and tiny["train_tokens"] == 1000 and tiny["val_tokens"] == 100
    r = client.post("/api/data/prepare", json={"dataset": "tiny"})
    assert r.status_code == 409 and "already has 2 shards" in r.json()["detail"]


def test_evaluate_and_compare(client, lab):
    start(client, "ev")
    wait_for(lambda: run_state(client, "ev")["state"] == "finished", what="training")
    body = {"run": "ev", "val_loss": True, "hellaswag": False}
    job = client.post("/api/evaluate", json=body).json()
    done = wait_for(
        lambda: (j := client.get(f"/api/jobs/{job['id']}").json())["state"] != "running" and j,
        what="evaluation",
    )
    assert done["state"] == "succeeded", client.get(f"/api/jobs/{job['id']}/log").json()["text"]
    (lab / "runs" / "reference").mkdir()
    (lab / "runs" / "reference" / "gpt2.json").write_text(
        json.dumps({"name": "hf:gpt2", "params": 124e6, "val_loss": 3.29})
    )
    rows = {r["name"]: r for r in client.get("/api/compare").json()["rows"]}
    assert rows["hf:gpt2"]["reference"] is True
    ours = next(r for r in rows.values() if r["run"] == "ev")
    assert ours["val_loss"] is not None and ours["step"] == 6
    assert client.post("/api/evaluate", json={"hf": "gpt5"}).status_code == 400
