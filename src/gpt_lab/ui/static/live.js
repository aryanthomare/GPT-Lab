// The live training window: everything about one run on one screen, refreshed every two
// seconds. A run's page opens it in a window of its own; the Live tab shows the active run.
// Loaded on demand by app.js, whose helpers it shares.

import {
  api, currentToken, enc, every, fill, fmt, h, ICONS, LIVE, lineChart, lrPlan, onLeave, runBadge,
  s, setLog, sparkline, stillOn, toast,
} from "./app.js";

const main = document.getElementById("main");
const GPU_WINDOW = 30 * 60; // seconds of GPU history on screen
const clock = (t) => new Date(t * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
const clockSec = (t) => new Date(t * 1000).toLocaleTimeString();
const finishAt = (t) => new Date(t * 1000).toLocaleString([], { weekday: "short", hour: "2-digit", minute: "2-digit" });

// ---- stat tiles -------------------------------------------------------------------------
function tile(label) {
  const v = h("span", { class: "v", text: "—" });
  const sub = h("span", { class: "sub" });
  const spark = h("span", { class: "tile-spark" });
  const el = h("div", { class: "tile" }, h("span", { class: "k", text: label }), v, sub, spark);
  return {
    el,
    set(value, subText, points) {
      v.textContent = value;
      sub.textContent = subText || "";
      fill(spark, points && points.length > 1 ? sparkline(points, 120, 26) : null);
    },
  };
}

// ---- run timeline -----------------------------------------------------------------------
// The whole run on one line: schedule phases, progress, evaluations and checkpoints.
function timeline() {
  const box = h("div", { class: "timeline" });
  const tip = h("div", { class: "tooltip", hidden: true });
  const legend = h("div", { class: "chart-legend" },
    h("span", {}, h("span", { class: "pbar-swatch", style: { background: "var(--ink)" } }), "done"),
    h("span", {}, h("span", { class: "pbar-swatch", style: { background: "var(--series-2)", borderRadius: "50%" } }), "evaluation"),
    h("span", {}, h("span", { class: "pbar-swatch", style: { background: "var(--ink-2)" } }), "checkpoint"));
  const el = h("div", { class: "timeline-wrap" }, box, tip, legend);
  let data = null;
  const ro = new ResizeObserver(() => render());
  ro.observe(box);
  onLeave(() => ro.disconnect());

  function render() {
    if (!data || !data.max) return fill(box);
    const W = box.clientWidth || 800;
    const H = 74;
    const l = 10;
    const r = 10;
    const X = (v) => l + (Math.min(Math.max(v, 0), data.max) / data.max) * (W - l - r);
    const ty = 30;
    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, style: `height:${H}px`, role: "img", "aria-label": `Step ${fmt.int(data.step)} of ${fmt.int(data.max)}` });
    // Schedule phases, labelled under the track where they fit.
    const bounds = [0, ...data.marks.map((m) => m.x), data.max];
    const names = data.marks.length === 2 ? ["warmup", "stable", "decay"] : data.marks.length === 1 ? ["warmup", "schedule"] : ["schedule"];
    svg.append(s("rect", { x: l, y: ty, width: W - l - r, height: 10, fill: "var(--rule)" }));
    bounds.slice(0, -1).forEach((b, i) => {
      const x0 = X(b);
      const x1 = X(bounds[i + 1]);
      if (i > 0) svg.append(s("line", { x1: x0, x2: x0, y1: ty - 4, y2: ty + 14, stroke: "var(--muted)", "stroke-width": 1 }));
      if (x1 - x0 > 60) svg.append(s("text", { class: "tl-label", x: (x0 + x1) / 2, y: ty + 30, "text-anchor": "middle" }, names[i] || ""));
    });
    svg.append(s("rect", { x: l, y: ty, width: Math.max(0, X(data.step) - l), height: 10, fill: "var(--ink)" }));
    // Events: evaluations above the track, checkpoints below.
    const marks = [];
    for (const [st, val] of data.evals) {
      svg.append(s("circle", { cx: X(st), cy: ty - 9, r: 3.5, fill: "var(--series-2)", stroke: "var(--sheet)", "stroke-width": 1.5 }));
      marks.push({ x: X(st), text: `Evaluation at step ${fmt.int(st)}: validation loss ${fmt.loss(val)}` });
    }
    for (const c of data.checkpoints) {
      svg.append(s("rect", { x: X(c.step) - 3, y: ty + 13, width: 6, height: 6, rx: 1, fill: "var(--ink-2)" }));
      marks.push({ x: X(c.step), text: `Checkpoint at step ${fmt.int(c.step)} (${fmt.bytes(c.bytes)}), saved ${clock(c.mtime)}` });
    }
    const cx = X(data.step);
    svg.append(s("line", { x1: cx, x2: cx, y1: 8, y2: ty + 22, stroke: "var(--focus)", "stroke-width": 2 }));
    const pct = ((100 * data.step) / data.max).toFixed(1);
    const anchor = cx > W - 140 ? "end" : cx < 140 ? "start" : "middle";
    svg.append(s("text", { class: "tl-now", x: cx + (anchor === "end" ? -4 : anchor === "start" ? 4 : 0), y: 9, "text-anchor": anchor }, `step ${fmt.int(data.step)} · ${pct}%`));
    const overlay = s("rect", { x: 0, y: 0, width: W, height: H, fill: "transparent" });
    overlay.addEventListener("pointermove", (e) => {
      const rect = svg.getBoundingClientRect();
      const px = ((e.clientX - rect.left) / rect.width) * W;
      const near = marks.reduce((best, m) => (Math.abs(m.x - px) < Math.abs((best ? best.x : Infinity) - px) ? m : best), null);
      if (!near || Math.abs(near.x - px) > 10) return (tip.hidden = true);
      tip.textContent = near.text;
      tip.hidden = false;
      tip.style.left = `${Math.min(Math.max(0, near.x - 120), W - 250)}px`;
      tip.style.top = "46px";
    });
    overlay.addEventListener("pointerleave", () => (tip.hidden = true));
    svg.append(overlay);
    fill(box, svg);
  }
  return {
    el,
    update(d) {
      data = d;
      render();
    },
  };
}

// ---- the page ---------------------------------------------------------------------------
export async function livePage(name) {
  const token = currentToken();
  document.body.classList.add("live-mode");
  onLeave(() => document.body.classList.remove("live-mode"));
  if (!name) {
    const runs = await api("/api/runs");
    if (!stillOn(token)) return;
    const pick = runs.find((r) => LIVE.has(r.state)) || runs[0];
    if (!pick) {
      main.append(h("div", { class: "panel empty" }, h("h3", { text: "No runs yet" }),
        h("p", { text: "Start a run and its live view appears here." }),
        h("div", { class: "actions" }, h("a", { class: "btn primary", href: "#/new", text: "New run" }))));
      return;
    }
    name = pick.name;
    history.replaceState(null, "", `#/live/${enc(name)}`);
  }

  // Header
  const liveMark = h("span", { class: "live-mark" }, h("span", { class: "dot" }), "LIVE");
  const badgeBox = h("span");
  const stopBtn = h("button", { class: "btn small", type: "button", text: "Stop and save", hidden: true });
  const head = h("header", { class: "live-head" },
    h("a", { class: "wordmark", href: "#/runs", text: "GPT-LAB" }), liveMark,
    h("h1", { class: "live-name", text: name }), badgeBox,
    h("div", { class: "live-links" }, stopBtn, h("a", { class: "btn small quiet", href: `#/runs/${enc(name)}`, text: "Run page" })));
  stopBtn.addEventListener("click", async () => {
    try {
      await api(`/api/runs/${enc(name)}/stop`, { method: "POST" });
      toast("Stopping. The run saves a checkpoint first.");
    } catch (e) {
      toast(e.message, "error");
    }
  });

  // Tiles
  const T = {
    step: tile("Step"),
    loss: tile("Loss"),
    val: tile("Validation loss"),
    hs: tile("HellaSwag"),
    speed: tile("Throughput"),
    eta: tile("Time left"),
    lr: tile("Learning rate"),
    util: tile("GPU"),
    mem: tile("GPU memory"),
    temp: tile("Temperature"),
  };
  const tiles = h("section", { class: "tiles", "aria-label": "Current numbers" }, Object.values(T).map((t) => t.el));
  const tl = timeline();

  // Charts
  const timeAxis = { xMin: "auto", xTick: clock, xLabel: clockSec, zero: true };
  const charts = {
    loss: lineChart({ title: "Loss", clip: true, format: fmt.loss, tick: (t) => String(+t.toFixed(3)), height: 420, series: [
      { key: "train/loss", label: "Train", color: "var(--series-1)" },
      { key: "eval/val_loss", label: "Validation", color: "var(--series-2)", dots: true },
    ] }),
    lr: lineChart({ title: "Learning rate", zero: true, format: fmt.sci, height: 180, series: [
      { key: "train/lr", label: "Actual", color: "var(--series-1)" },
      { key: "plan/lr", label: "Planned", color: "var(--series-2)", track: true },
    ] }),
    grad: lineChart({ title: "Gradient norm", format: (v) => v.toFixed(3), tick: (t) => String(+t.toFixed(2)), height: 180,
      emptyText: "Logged with the loss.", series: [{ key: "train/grad_norm", label: "Gradient norm", color: "var(--series-1)" }] }),
    speed: lineChart({ title: "Throughput", zero: true, format: fmt.rate, tick: fmt.compact, height: 180,
      series: [{ key: "perf/tokens_per_sec", label: "Tokens per second", color: "var(--series-1)" }] }),
    hs: lineChart({ title: "HellaSwag accuracy", format: fmt.pct, tick: (t) => `${(t * 100).toFixed(0)}%`, height: 180,
      emptyText: "Scored every hellaswag_every steps.", series: [{ key: "eval/hellaswag_acc_norm", label: "acc_norm", color: "var(--series-1)", dots: true }] }),
    util: lineChart({ title: "GPU utilization", format: (v) => `${v.toFixed(0)}%`, height: 180, ...timeAxis,
      emptyText: "No GPU readings yet.", series: [{ key: "util", label: "Utilization", color: "var(--series-1)" }] }),
    mem: lineChart({ title: "GPU memory", format: (v) => `${v.toFixed(1)} GB`, tick: (t) => `${+t.toFixed(1)}`, height: 180, ...timeAxis,
      emptyText: "No GPU readings yet.", series: [{ key: "mem", label: "Memory used", color: "var(--series-1)" }] }),
    temp: lineChart({ title: "GPU temperature", format: (v) => `${v.toFixed(0)}°C`, tick: (t) => `${t}°`, height: 180, ...timeAxis, zero: false,
      emptyText: "No GPU readings yet.", series: [{ key: "temp", label: "Temperature", color: "var(--series-1)" }] }),
    power: lineChart({ title: "GPU power", format: (v) => `${v.toFixed(0)} W`, tick: (t) => `${t}`, height: 180, ...timeAxis,
      emptyText: "No GPU readings yet.", series: [{ key: "power", label: "Power draw", color: "var(--series-1)" }] }),
  };
  charts.loss.el.classList.add("loss");

  // Samples, log, events
  const samples = h("div", { class: "panel live-samples" });
  const logPre = h("pre", { class: "log panel live-log" });
  const events = h("ol", { class: "panel events" });

  main.append(h("div", { class: "live" },
    head, tiles,
    h("section", { class: "panel timeline-panel", "aria-label": "Run timeline" }, h("div", { class: "chart-head" }, h("span", { class: "chart-title", text: "Run timeline" })), tl.el),
    h("section", { class: "live-charts" }, Object.values(charts).map((c) => c.el)),
    h("section", { class: "live-bottom" },
      h("div", {}, h("h2", { text: "Latest samples" }), samples),
      h("div", {}, h("h2", { text: "Log" }), logPre),
      h("div", {}, h("h2", { text: "Events" }), events))));

  // State kept between refreshes
  const series = {};
  const evalTime = new Map();
  let cursor = null;
  let plan = null;
  const stepLoss = [];
  const stepSpeed = [];
  let lastStep = null;
  const gpu = { util: [], mem: [], temp: [], power: [] };
  let gpuSince = Date.now() / 1000 - GPU_WINDOW;
  let ticks = 0;

  every(2000, async () => {
    ticks++;
    const [d, m, g] = await Promise.all([
      api(`/api/runs/${enc(name)}`),
      api(`/api/runs/${enc(name)}/metrics${cursor ? `?cursor=${enc(cursor)}` : ""}`),
      api(`/api/system/history?since=${gpuSince}`),
    ]);
    if (!stillOn(token)) return;
    const live = LIVE.has(d.state);

    // Metric rows
    if (m.reset) {
      for (const k of Object.keys(series)) delete series[k];
      evalTime.clear();
    }
    for (const row of m.rows) {
      if (Object.keys(row).some((k) => k.startsWith("eval/"))) evalTime.set(row.step, row.time);
      for (const [k, val] of Object.entries(row)) {
        if (k === "step" || k === "time" || typeof val !== "number") continue;
        (series[k] ||= []).push([row.step, val]);
      }
    }
    if (m.cursor) cursor = m.cursor;
    if (!plan && d.config) plan = lrPlan(d.config.train);

    // Per-step numbers from status.json, kept while this window is open
    if (d.step_loss != null && d.step !== lastStep) {
      lastStep = d.step;
      stepLoss.push([d.step, d.step_loss]);
      if (d.step_seconds && d.tokens_per_step) stepSpeed.push([d.step, d.tokens_per_step / d.step_seconds]);
      if (stepLoss.length > 240) stepLoss.shift();
      if (stepSpeed.length > 240) stepSpeed.shift();
    }

    // GPU history
    for (const smp of g.samples) {
      if (smp.util != null) gpu.util.push([smp.t, smp.util]);
      if (smp.mem_used_mb != null) gpu.mem.push([smp.t, smp.mem_used_mb / 1024]);
      if (smp.temp_c != null) gpu.temp.push([smp.t, smp.temp_c]);
      if (smp.power_w != null) gpu.power.push([smp.t, smp.power_w]);
      gpuSince = Math.max(gpuSince, smp.t);
    }
    const cutoff = g.now - GPU_WINDOW;
    for (const k of Object.keys(gpu)) while (gpu[k].length && gpu[k][0][0] < cutoff) gpu[k].shift();
    const latestGpu = g.samples.at(-1) || null;

    // Header and tiles
    fill(badgeBox, runBadge(d.state));
    liveMark.classList.toggle("on", live);
    stopBtn.hidden = !live;
    const pct = d.max_steps ? (100 * d.step) / d.max_steps : 0;
    T.step.set(fmt.int(d.step), `of ${fmt.int(d.max_steps)} · ${pct.toFixed(1)}%`);
    const lossNow = d.step_loss ?? d.loss;
    T.loss.set(fmt.loss(lossNow), d.loss != null ? `logged ${fmt.loss(d.loss)}` : "this step", stepLoss.length > 2 ? stepLoss : d.loss_trend);
    T.val.set(fmt.loss(d.val_loss), d.best_val_loss != null ? `best ${fmt.loss(d.best_val_loss)}` : "not scored yet", series["eval/val_loss"]);
    T.hs.set(fmt.pct(d.hellaswag), d.hellaswag != null ? "acc_norm" : "not scored yet", series["eval/hellaswag_acc_norm"]);
    const speedNow = d.step_seconds && d.tokens_per_step ? d.tokens_per_step / d.step_seconds : d.tokens_per_sec;
    T.speed.set(live && speedNow ? `${fmt.compact(speedNow)}/s` : "—", d.step_seconds ? `${d.step_seconds.toFixed(2)} s per step` : "tokens per second", stepSpeed.length > 2 ? stepSpeed : series["perf/tokens_per_sec"]);
    const elapsed = d.started_at ? (live ? Date.now() / 1000 : d.updated_at) - d.started_at : null;
    let etaSub = elapsed ? `ran ${fmt.duration(elapsed)}` : "";
    if (live) {
      etaSub = d.eta_seconds
        ? `done ${finishAt(Date.now() / 1000 + d.eta_seconds)} · ${fmt.duration(elapsed)} in`
        : `${fmt.duration(elapsed)} in · estimating`;
    }
    T.eta.set(live ? fmt.duration(d.eta_seconds) : d.state === "finished" ? "Done" : "—", etaSub);
    const phase = plan && plan.marks.length ? (d.step < plan.marks[0].x ? "warmup" : plan.marks[1] && d.step >= plan.marks[1].x ? "decay" : "stable") : "";
    T.lr.set(fmt.sci(d.step_lr ?? d.lr), phase, series["train/lr"]);
    if (latestGpu) {
      T.util.set(`${latestGpu.util ?? "—"}%`, latestGpu.name.replace(/^NVIDIA (GeForce )?/, ""), gpu.util.slice(-90));
      T.mem.set(`${(latestGpu.mem_used_mb / 1024).toFixed(1)} GB`, `of ${(latestGpu.mem_total_mb / 1024).toFixed(1)} GB`, gpu.mem.slice(-90));
      T.temp.set(latestGpu.temp_c != null ? `${latestGpu.temp_c}°C` : "—", latestGpu.power_w != null ? `${latestGpu.power_w.toFixed(0)} W` : "", gpu.temp.slice(-90));
    }
    document.title = live
      ? `${fmt.loss(lossNow)} · ${fmt.int(d.step)}/${fmt.int(d.max_steps)} · ${name}`
      : `${name} · ${d.state} · GPT-Lab live`;

    // Timeline and charts
    tl.update({
      max: d.max_steps,
      step: d.step,
      marks: plan ? plan.marks : [],
      evals: series["eval/val_loss"] || [],
      checkpoints: d.checkpoints,
    });
    const stepOpts = { xMax: d.max_steps };
    charts.loss.update(series, stepOpts);
    charts.lr.update({ ...series, "plan/lr": plan ? plan.points : [] }, { ...stepOpts, marks: plan ? plan.marks : [] });
    charts.grad.update(series, stepOpts);
    charts.speed.update(series, stepOpts);
    charts.hs.update(series, stepOpts);
    charts.util.update({ util: gpu.util });
    charts.mem.update({ mem: gpu.mem });
    charts.temp.update({ temp: gpu.temp });
    charts.power.update({ power: gpu.power });

    // Samples
    const prompts = (d.config && d.config.train && d.config.train.sample_prompts) || [];
    fill(samples, d.samples.length ? d.samples.map((smp) => {
      const i = Number(smp.tag.split("/")[1]);
      const prompt = prompts[i] || "";
      const text = smp.text || "";
      return h("div", { class: "sample" },
        h("span", { class: "meta", text: `PROMPT ${i + 1} · STEP ${fmt.int(smp.step)}` }),
        h("p", {}, prompt && text.startsWith(prompt) ? [h("span", { class: "prompt", text: prompt }), text.slice(prompt.length)] : text));
    }) : h("p", { class: "empty muted", text: "No samples yet. The run writes some every sample_every steps." }));

    // Log (every other refresh while live)
    if (d.has_log && (ticks === 1 || (live && ticks % 2 === 0))) {
      const { text } = await api(`/api/runs/${enc(name)}/log?lines=200`);
      setLog(logPre, text);
    } else if (!d.has_log) {
      logPre.textContent = "No log. Runs started from a terminal print there instead.";
    }

    // Events, newest first. `order` follows the training loop: evaluations, then the
    // checkpoint saved at the same step.
    const list = [];
    if (d.started_at) list.push({ order: -1, t: d.started_at, icon: ICONS.pending(), text: "Run started" });
    for (const [st, val] of series["eval/val_loss"] || []) {
      const best = val === d.best_val_loss ? " (best so far)" : "";
      list.push({ order: st, step: st, t: evalTime.get(st), icon: ICONS.done(), text: `Validation loss ${fmt.loss(val)}${best}` });
    }
    for (const [st, val] of series["eval/hellaswag_acc_norm"] || []) {
      list.push({ order: st, step: st, t: evalTime.get(st), icon: ICONS.done(), text: `HellaSwag ${fmt.pct(val)}` });
    }
    for (const c of d.checkpoints) {
      list.push({ order: c.step + 0.5, step: c.step, t: c.mtime, icon: ICONS.paused(), text: `Checkpoint saved (${fmt.bytes(c.bytes)})` });
    }
    if (d.state === "finished") list.push({ order: Infinity, t: d.updated_at, icon: ICONS.done(), text: "Run finished" });
    if (d.state === "failed" || d.state === "crashed") {
      const why = d.state === "failed" ? `Failed: ${d.error || "see the log"}` : "The training process stopped unexpectedly";
      list.push({ order: Infinity, t: d.updated_at, icon: ICONS.error(), text: why });
    }
    list.sort((a, b) => b.order - a.order);
    fill(events, list.slice(0, 40).map((ev) =>
      h("li", { class: "event" }, ev.icon,
        h("span", {}, ev.text, h("span", { class: "sub", text: [ev.step != null ? ` · step ${fmt.int(ev.step)}` : "", ev.t ? ` · ${clock(ev.t)}` : ""].join("") })))));
  });
}
