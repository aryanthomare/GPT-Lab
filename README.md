# GPT-Lab

Train a modern, GPT-2-sized (~124M parameter) language model from scratch on a single
RTX 5080, then compare it with OpenAI's GPT-2 on the same benchmarks.

- **Model**: Llama-style decoder: rotary position embeddings, RMSNorm, SwiGLU, QK-norm,
  no biases, tied embeddings. 12 layers x 768 wide = 123.6M parameters, matching GPT-2 small.
- **Data**: FineWeb-Edu `sample-10BT` (~10 billion tokens), GPT-2 tokenizer, so
  validation loss is directly comparable to GPT-2 per token.
- **Training**: bfloat16 autocast, `torch.compile`, fused AdamW, warmup-stable-decay
  learning rate, 524,288 tokens per step, exact resume from checkpoints, TensorBoard.
- **Evaluation**: FineWeb-Edu validation loss, HellaSwag, and optionally
  lm-evaluation-harness tasks (ARC, PIQA, WinoGrande, LAMBADA). The same code scores
  our checkpoints and Hugging Face `gpt2` / `gpt2-medium`.

## Layout

```
configs/            YAML experiment configs (main run, 350M stretch, tiny CPU smoke test)
src/gpt_lab/
  model.py          the transformer
  data.py           uint16 token shards + resumable shuffled loader
  train.py          training loop               python -m gpt_lab.train
  generate.py       text sampling               python -m gpt_lab.generate
  evaluate.py       evaluation suite            python -m gpt_lab.evaluate
  compare.py        results table               python -m gpt_lab.compare
  evals/            validation loss, HellaSwag, lm-evaluation-harness adapter
  hf_reference.py   Hugging Face GPT-2 wrapper for comparisons
  ui/               local web UI                python -m gpt_lab.ui
scripts/            dataset preparation, GPU benchmark, UI launchers
tests/              CPU-only unit tests (pytest)
```

## Setup on Windows + WSL2 (RTX 5080)

1. **Windows side**: install the latest NVIDIA Game Ready or Studio driver for Windows.
   Do **not** install an NVIDIA driver inside WSL; the Windows driver provides CUDA to WSL.
2. Install Ubuntu in WSL2 (`wsl --install -d Ubuntu-24.04` in PowerShell).
3. Give WSL enough memory for data preparation. Create `C:\Users\<you>\.wslconfig`:
   ```ini
   [wsl2]
   memory=24GB      # adjust to ~75% of your RAM
   processors=12    # adjust to your CPU
   ```
   Then run `wsl --shutdown` and reopen Ubuntu.
4. Inside Ubuntu, **keep the repository and data on the Linux filesystem** (for example
   `~/GPT-Lab`), not under `/mnt/c`. Reads through `/mnt/c` are many times slower.
   ```bash
   sudo apt update && sudo apt install -y python3-venv python3-dev build-essential git
   git clone https://github.com/aryanthomare/GPT-Lab.git ~/GPT-Lab && cd ~/GPT-Lab
   python3 -m venv .venv && source .venv/bin/activate
   # The RTX 50 series (Blackwell) needs PyTorch >= 2.7 built for CUDA 12.8:
   pip install torch --index-url https://download.pytorch.org/whl/cu128
   pip install -e ".[data,eval,dev]"
   python -c "import torch; print(torch.__version__, torch.cuda.get_device_name())"
   pytest -q
   ```

## Run guide

### 0. Smoke test (about 2 minutes, CPU)
```bash
python scripts/prepare_tiny.py
python -m gpt_lab.train --config configs/tiny_cpu.yaml
python -m gpt_lab.generate --checkpoint runs/tiny_cpu --device cpu
```

### 1. Prepare FineWeb-Edu (a few hours; ~20 GB of shards)
```bash
python scripts/prepare_fineweb_edu.py --out-dir data/fineweb_edu_10B
```
This writes `val_000000.npy` (100M held-out tokens) and `train_000001.npy`, and so on.
The raw download is cached in `~/.cache/huggingface`, and you can delete that cache afterwards.

### 2. Benchmark the GPU (about 10 minutes)
```bash
python scripts/benchmark.py --config configs/modern_124m_fineweb10b.yaml
```
Choose the largest `micro_batch_size` that fits with about 1-2 GB of free memory. It must
divide 524,288 / 1024 = 512, so use 8, 16 or 32. Set it in the config or pass
`--train.micro_batch_size=32`. The script also prints the estimated time for the full run.

### 3. Sanity run (about 200M tokens)
```bash
python -m gpt_lab.train --config configs/modern_124m_fineweb10b.yaml \
    --train.run_name=sanity --train.max_steps=400 --train.warmup_steps=100 \
    --train.eval_every=100 --train.hellaswag_every=200 --train.hellaswag_limit=1000
tensorboard --logdir runs    # open http://localhost:6006 in Windows
```
The loss should fall below about 5 within 200 steps. Check tokens per second, peak memory,
and that the samples start to look like English.

### 4. Full run (~10B tokens; estimated 25-45 hours)
```bash
python -m gpt_lab.train --config configs/modern_124m_fineweb10b.yaml
# after any interruption (Ctrl-C, reboot, crash):
python -m gpt_lab.train --config configs/modern_124m_fineweb10b.yaml --resume
```
Pressing Ctrl-C (or sending SIGTERM) saves a checkpoint before exiting. Checkpoints are also
written every 500 steps. Tips for a long run: pause Windows Update and turn off sleep.
**Keep an Ubuntu terminal open, or use the web UI** (below). WSL shuts Ubuntu down a few
seconds after its last terminal closes, and that stops training even inside `tmux` or `nohup`.

### 5. Evaluate and compare with GPT-2
```bash
python -m gpt_lab.evaluate --checkpoint runs/modern_124m_fineweb10b
python -m gpt_lab.evaluate --hf gpt2
python -m gpt_lab.evaluate --hf gpt2-medium
# optional broader benchmark (adds a few minutes per task):
python -m gpt_lab.evaluate --checkpoint runs/modern_124m_fineweb10b \
    --lm-eval arc_easy,arc_challenge,piqa,winogrande,lambada_openai
python -m gpt_lab.compare runs/*/eval.json runs/reference/*.json
```

**Reference points** (moderate confidence, measured with this style of scoring by others):
OpenAI GPT-2 124M reaches about 3.29 FineWeb-Edu validation loss and about 29.5% HellaSwag.
A GPT-2-architecture model trained on 10B FineWeb-Edu tokens reaches about 30-31% HellaSwag.
A modern architecture should match or beat that.

## Web UI

A local web app for everything above: start runs from a preset (with any setting changed,
and the model drawn in 3D as you edit it), watch loss, HellaSwag, throughput and samples,
stop and resume, prepare datasets, evaluate checkpoints against GPT-2, and generate text.

Each run also has a **live window** (Live window on its page, or the Live tab): one screen
refreshed every two seconds with the current step, loss, throughput, time left, a timeline
of the run's schedule, evaluations and checkpoints, charts of loss, learning rate, gradient
norm, throughput and HellaSwag, the GPU's utilization, memory, temperature and power over
the last half hour, the latest samples, the log, and a list of events.

```bash
pip install -e ".[ui]"
python -m gpt_lab.ui          # then open http://localhost:8000, from Windows too
```

From Windows, `scripts/windows/Start-GPTLabUI.ps1` starts the server in WSL (unless it's
already running) and opens the browser:

```powershell
powershell -ExecutionPolicy Bypass -File \\wsl.localhost\Ubuntu\home\<you>\GPT-Lab\scripts\windows\Start-GPTLabUI.ps1
```

The server runs under a hidden `wsl.exe`, which also keeps WSL running, so training carries
on after you close every terminal and browser tab. Stop it with Settings > Shut down server.
Jobs the UI starts run in sessions of their own, so restarting the server doesn't touch
them. The server only answers on localhost, and its log is `runs/.ui-server.log`.

## Configuration

Every run is described by a YAML file in `configs/` (see `src/gpt_lab/config.py` for all
fields and defaults). Any field can be overridden with `--section.key=value`, for example
`--model.n_kv_head=4 --train.lr=1.5e-3`. Each run directory `runs/<run_name>/` stores the
resolved `config.yaml`, the git commit, TensorBoard logs, checkpoints and `eval.json`, plus
two files for tools that follow a run: `metrics.jsonl` (every logged metric and text sample,
one JSON object per line) and `status.json` (state, step, process ID, latest checkpoint, and
the latest step's loss, learning rate and duration, updated every step).

## Development

```bash
pytest -q          # CPU-only tests
ruff check . && ruff format .
```
