// GPT-Lab UI: a small single-page app over the JSON API in gpt_lab/ui/app.py.
// Text from the server always goes into the page through textContent (see h()).

import { modelSpec, modelView, paramBar } from "./model3d.js";

const main = document.getElementById("main");

// ---- DOM helpers ------------------------------------------------------------------------
function h(tag, props = {}, ...children) {
  const el = document.createElement(tag);
  setProps(el, props);
  return append(el, children);
}

function setProps(el, props) {
  for (const [k, v] of Object.entries(props || {})) {
    if (v == null || v === false) continue;
    if (k === "class") el.setAttribute("class", v);
    else if (k === "text") el.textContent = v;
    else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
    else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
    else if (v === true) el.setAttribute(k, "");
    else el.setAttribute(k, v);
  }
}

function append(el, children) {
  for (const c of children.flat(Infinity)) {
    if (c == null || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

// Like el.replaceChildren(), but skips null and false (replaceChildren would print "null").
function fill(el, ...children) {
  el.replaceChildren();
  return append(el, children);
}

const SVG_NS = "http://www.w3.org/2000/svg";
function s(tag, attrs = {}, ...children) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v != null) el.setAttribute(k, v);
  return append(el, children);
}

// ---- API --------------------------------------------------------------------------------
async function api(path, { method = "GET", body } = {}) {
  const init = { method };
  if (body !== undefined) {
    init.headers = { "Content-Type": "application/json" };
    init.body = JSON.stringify(body);
  }
  const res = await fetch(path, init);
  const text = await res.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = null;
  }
  if (!res.ok) {
    let msg = data && data.detail;
    if (Array.isArray(msg)) msg = msg.map((d) => `${(d.loc || []).slice(1).join(".")}: ${d.msg}`).join("; ");
    throw new Error(msg || `The server answered ${res.status} ${res.statusText}.`);
  }
  return data;
}

// ---- formatting -------------------------------------------------------------------------
const nf = new Intl.NumberFormat("en-US");
const fmt = {
  int: (v) => (v == null ? "—" : nf.format(Math.round(v))),
  compact(v) {
    if (v == null) return "—";
    const a = Math.abs(v);
    if (a >= 1e9) return `${(v / 1e9).toFixed(a >= 1e10 ? 1 : 2)}B`;
    if (a >= 1e6) return `${(v / 1e6).toFixed(1)}M`;
    if (a >= 1e3) return `${(v / 1e3).toFixed(1)}k`;
    return String(Math.round(v * 100) / 100);
  },
  loss: (v) => (v == null ? "—" : v.toFixed(3)),
  pct: (v) => (v == null ? "—" : `${(v * 100).toFixed(1)}%`),
  sci: (v) => (v == null ? "—" : v === 0 ? "0" : v.toExponential(1)),
  rate: (v) => (v == null ? "—" : `${fmt.compact(v)} tok/s`),
  bytes(v) {
    if (v == null) return "—";
    const units = ["B", "KB", "MB", "GB", "TB"];
    let i = 0;
    while (v >= 1024 && i < units.length - 1) {
      v /= 1024;
      i++;
    }
    return `${v.toFixed(v >= 100 || i === 0 ? 0 : 1)} ${units[i]}`;
  },
  duration(sec) {
    if (sec == null || !isFinite(sec)) return "—";
    if (sec < 90) return `${Math.max(0, Math.round(sec))} s`;
    if (sec < 5400) return `${Math.round(sec / 60)} min`;
    if (sec < 172800) return `${(sec / 3600).toFixed(1)} h`;
    return `${(sec / 86400).toFixed(1)} days`;
  },
  ago(t) {
    if (!t) return "—";
    const sec = Date.now() / 1000 - t;
    if (sec < 10) return "just now";
    if (sec < 5400) return `${fmt.duration(sec)} ago`;
    if (sec < 172800) return `${Math.round(sec / 3600)} h ago`;
    return new Date(t * 1000).toLocaleDateString();
  },
  date: (t) =>
    t ? new Date(t * 1000).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" }) : "—",
};
const enc = encodeURIComponent;

// ---- icons and state badges -------------------------------------------------------------
const ICONS = {
  live: () => s("svg", { viewBox: "0 0 12 12", "aria-hidden": "true" }, s("circle", { class: "dot", cx: 6, cy: 6, r: 5, fill: "var(--good)" })),
  pending: () => s("svg", { viewBox: "0 0 12 12", "aria-hidden": "true" }, s("circle", { cx: 6, cy: 6, r: 4.25, fill: "none", stroke: "var(--ink-2)", "stroke-width": 1.5 })),
  done: () =>
    s("svg", { viewBox: "0 0 12 12", "aria-hidden": "true" },
      s("circle", { cx: 6, cy: 6, r: 6, fill: "var(--good)" }),
      s("path", { d: "M3.2 6.2 5.2 8.1 8.9 4.2", fill: "none", stroke: "#fff", "stroke-width": 1.6, "stroke-linecap": "round", "stroke-linejoin": "round" })),
  paused: () =>
    s("svg", { viewBox: "0 0 12 12", "aria-hidden": "true" },
      s("rect", { x: 2, y: 1.5, width: 3, height: 9, rx: 1, fill: "var(--warning)" }),
      s("rect", { x: 7, y: 1.5, width: 3, height: 9, rx: 1, fill: "var(--warning)" })),
  error: () =>
    s("svg", { viewBox: "0 0 12 12", "aria-hidden": "true" },
      s("circle", { cx: 6, cy: 6, r: 6, fill: "var(--critical)" }),
      s("path", { d: "M4 4 8 8M8 4 4 8", stroke: "#fff", "stroke-width": 1.6, "stroke-linecap": "round" })),
  warn: () =>
    s("svg", { viewBox: "0 0 12 12", "aria-hidden": "true" },
      s("path", { d: "M6 .8 11.5 11H.5Z", fill: "var(--warning)" }),
      s("path", { d: "M6 4.5v3M6 9.2v.1", stroke: "#16222b", "stroke-width": 1.4, "stroke-linecap": "round" })),
  unknown: () => s("svg", { viewBox: "0 0 12 12", "aria-hidden": "true" }, s("circle", { cx: 6, cy: 6, r: 4.25, fill: "none", stroke: "var(--muted)", "stroke-width": 1.5 })),
};
const RUN_STATES = {
  running: ["Running", "live"],
  starting: ["Starting", "pending"],
  finished: ["Finished", "done"],
  interrupted: ["Stopped", "paused"],
  stopped: ["Paused", "paused"],
  failed: ["Failed", "error"],
  crashed: ["Crashed", "error"],
  unknown: ["No status", "unknown"],
};
const JOB_STATES = {
  running: ["Running", "live"],
  succeeded: ["Done", "done"],
  failed: ["Failed", "error"],
  stopped: ["Stopped", "paused"],
  lost: ["Lost", "error"],
};
function badge(table, state) {
  const [label, icon] = table[state] || [state, "unknown"];
  return h("span", { class: `badge ${icon}` }, ICONS[icon](), label);
}
const runBadge = (state) => badge(RUN_STATES, state);
const jobBadge = (state) => badge(JOB_STATES, state);
const LIVE = new Set(["running", "starting"]);

// ---- toasts -----------------------------------------------------------------------------
function toast(message, kind = "info") {
  const el = h("div", { class: `toast ${kind === "error" ? "error" : ""}`, text: message });
  document.getElementById("toasts").append(el);
  setTimeout(() => el.remove(), kind === "error" ? 9000 : 4500);
}

// ---- page lifecycle ---------------------------------------------------------------------
let cleanups = [];
let routeToken = 0;
const onLeave = (fn) => cleanups.push(fn);

// Call fn now and every `ms` while the page is open (skipped while the tab is hidden).
function every(ms, fn) {
  let stopped = false;
  let timer;
  const tick = async () => {
    if (stopped) return;
    if (!document.hidden) {
      try {
        await fn();
      } catch (e) {
        console.warn(e);
      }
    }
    if (!stopped) timer = setTimeout(tick, ms);
  };
  // Coming back to the tab refreshes at once instead of waiting for the next tick.
  const onVisible = () => {
    if (!document.hidden && !stopped) {
      clearTimeout(timer);
      tick();
    }
  };
  document.addEventListener("visibilitychange", onVisible);
  tick();
  onLeave(() => {
    stopped = true;
    clearTimeout(timer);
    document.removeEventListener("visibilitychange", onVisible);
  });
}

const routes = [
  [/^#\/runs\/([^/?]+)$/, (m) => runPage(decodeURIComponent(m[1]))],
  [/^#\/new(?:\?(.*))?$/, (m) => newRunPage(new URLSearchParams(m[1] || ""))],
  [/^#\/data$/, () => dataPage()],
  [/^#\/evaluate(?:\?(.*))?$/, (m) => evaluatePage(new URLSearchParams(m[1] || ""))],
  [/^#\/generate(?:\?(.*))?$/, (m) => generatePage(new URLSearchParams(m[1] || ""))],
  [/^(?:#\/?(?:runs)?)?$/, () => runsPage()],
];

function route() {
  cleanups.forEach((fn) => fn());
  cleanups = [];
  routeToken++;
  main.replaceChildren();
  const hash = location.hash;
  const section = (hash.match(/^#\/([a-z]+)/) || [null, "runs"])[1];
  for (const a of document.querySelectorAll(".nav a")) {
    if (a.dataset.nav === section) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  }
  for (const [re, page] of routes) {
    const m = hash.match(re);
    if (m) {
      Promise.resolve(page(m)).catch((e) => showError(e));
      window.scrollTo(0, 0);
      return;
    }
  }
  main.append(pageHead("Page not found"), h("p", {}, h("a", { href: "#/runs", text: "Go to runs" })));
}

function showError(e) {
  main.append(h("div", { class: "banner" }, ICONS.error(), h("span", { text: e.message || String(e) })));
}

// After an await: false if the person has moved to another page meanwhile.
function stillOn(token) {
  return token === routeToken;
}

function pageHead(title, intro, ...actions) {
  document.title = `${title} · GPT-Lab`;
  return h("div", { class: "page-head" },
    h("div", {}, h("h1", { text: title }), intro ? h("p", { text: intro }) : null),
    actions.length ? h("div", { class: "actions" }, actions) : null);
}

const th = (text, cls) => h("th", { class: cls, text });
const tdNum = (text) => h("td", { class: "num", text });

// ---- charts -----------------------------------------------------------------------------
function niceTicks(lo, hi, count) {
  const span = hi - lo;
  if (!(span > 0)) return [lo];
  const raw = span / Math.max(1, count);
  const mag = 10 ** Math.floor(Math.log10(raw));
  const norm = raw / mag;
  const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toPrecision(12));
  return out;
}

function nearest(pts, x) {
  let lo = 0;
  let hi = pts.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (pts[mid][0] < x) lo = mid;
    else hi = mid;
  }
  return Math.abs(pts[lo][0] - x) <= Math.abs(pts[hi][0] - x) ? lo : hi;
}

let chartIds = 0;

// A line chart over metric series. update(seriesByKey, {xMax}) redraws it.
function lineChart({ title, series, format, tick = format, clip = false, zero = false, emptyText = "Nothing logged yet.", headline = null }) {
  const id = ++chartIds;
  const now = h("span", { class: "chart-now" });
  const tableBtn = h("button", { class: "btn small quiet", type: "button", "aria-pressed": "false", text: "Table" });
  const legend = series.length > 1
    ? h("div", { class: "chart-legend" }, series.map((sr) => h("span", {},
        h("span", { class: "key", style: sr.track ? { background: sr.color, height: "6px", opacity: 0.3 } : { background: sr.color } }), sr.label)))
    : null;
  const plot = h("div", { class: "chart-plot" });
  const tableBox = h("div", { class: "chart-table", hidden: true });
  const el = h("section", { class: "panel chart-card" },
    h("div", { class: "chart-head" }, h("span", { class: "chart-title", text: title }), now, tableBtn),
    legend, plot, tableBox);

  let data = {};
  let opts = {};
  let showTable = false;
  let hoverStep = null;
  tableBtn.addEventListener("click", () => {
    showTable = !showTable;
    tableBtn.setAttribute("aria-pressed", String(showTable));
    tableBtn.textContent = showTable ? "Chart" : "Table";
    render();
  });
  const ro = new ResizeObserver(() => render());
  ro.observe(plot);
  onLeave(() => ro.disconnect());

  function update(all, o = {}) {
    data = all;
    opts = o;
    render();
  }

  function render() {
    const lines = series.map((sr) => ({ ...sr, pts: data[sr.key] || [] }));
    const last = lines[0].pts.at(-1);
    now.textContent = headline ? headline(lines) : last ? format(last[1]) : "";
    plot.hidden = showTable;
    tableBox.hidden = !showTable;
    if (showTable) return renderTable(lines);
    const primary = lines.find((l) => l.pts.length);
    if (!primary) {
      plot.replaceChildren(h("div", { class: "chart-empty", text: emptyText }));
      return;
    }
    const W = Math.max(280, plot.clientWidth || 600);
    const H = 200;
    const m = { t: 8, r: 14, b: 22, l: 52 };
    const iw = W - m.l - m.r;
    const ih = H - m.t - m.b;

    let xMaxSeen = 0;
    for (const l of lines) for (const p of l.pts) xMaxSeen = Math.max(xMaxSeen, p[0]);
    const x0 = 0;
    const x1 = Math.max(opts.xMax || 0, xMaxSeen, 1);
    // Leave the first stretch of training (a huge early loss) out of the y range.
    let ys = [];
    const cut = clip ? xMaxSeen * 0.1 : -Infinity;
    for (const l of lines) for (const p of l.pts) if (p[0] >= cut) ys.push(p[1]);
    if (ys.length < 3) ys = lines.flatMap((l) => l.pts.map((p) => p[1]));
    let y0 = Math.min(...ys);
    let y1 = Math.max(...ys);
    if (zero) y0 = Math.min(0, y0);
    if (y0 === y1) {
      const d = Math.abs(y0) * 0.1 || 1;
      y0 -= d;
      y1 += d;
    }
    const pad = (y1 - y0) * 0.08;
    y1 += pad;
    if (!(zero && y0 === 0)) y0 -= pad;
    const X = (v) => m.l + ((v - x0) / (x1 - x0)) * iw;
    const Y = (v) => m.t + (1 - (v - y0) / (y1 - y0)) * ih;

    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", tabindex: 0, "aria-label": `${title}: ${last ? format(last[1]) : "no data"} at step ${last ? fmt.int(last[0]) : 0}` });
    svg.append(s("defs", {}, s("clipPath", { id: `clip-${id}` }, s("rect", { x: m.l, y: m.t - 4, width: iw, height: ih + 8 }))));
    const axis = s("g", { class: "axis" });
    for (const t of niceTicks(y0, y1, 4)) {
      axis.append(s("line", { class: "gridline", x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t) }));
      axis.append(s("text", { x: m.l - 8, y: Y(t) + 3.5, "text-anchor": "end" }, tick(t)));
    }
    axis.append(s("line", { class: "baseline", x1: m.l, x2: W - m.r, y1: m.t + ih, y2: m.t + ih }));
    for (const t of niceTicks(x0, x1, Math.max(2, Math.floor(iw / 110)))) {
      axis.append(s("text", { x: X(t), y: H - 5, "text-anchor": "middle" }, fmt.compact(t)));
    }
    // Reference marks: a hairline at a step, labelled at the top (e.g. "warmup ends").
    for (const mark of opts.marks || []) {
      if (!(mark.x > x0 && mark.x < x1)) continue;
      const mx = X(mark.x);
      const flip = mx > m.l + iw * 0.7;
      axis.append(s("line", { class: "mark", x1: mx, x2: mx, y1: m.t, y2: m.t + ih }));
      axis.append(s("text", { class: "mark-label", x: mx + (flip ? -5 : 5), y: m.t + ih - 6, "text-anchor": flip ? "end" : "start" }, mark.label));
    }
    svg.append(axis);

    const plotG = s("g", { "clip-path": `url(#clip-${id})` });
    // "Track" series (a plan or reference) draw first, wide and faint, under the data.
    for (const l of [...lines].sort((a, b) => (b.track ? 1 : 0) - (a.track ? 1 : 0))) {
      if (!l.pts.length) continue;
      const d = "M" + l.pts.map((p) => `${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join("L");
      if (l.track) {
        plotG.append(s("path", { class: "series", d, stroke: l.color, "stroke-width": 6, opacity: 0.3 }));
        continue;
      }
      plotG.append(s("path", { class: "series", d, stroke: l.color }));
      const dots = l.dots && l.pts.length <= 120 ? l.pts : [l.pts.at(-1)];
      for (const p of dots) {
        plotG.append(s("circle", { cx: X(p[0]), cy: Y(p[1]), r: 4, fill: l.color, stroke: "var(--sheet)", "stroke-width": 2 }));
      }
    }
    svg.append(plotG);

    // Hover layer: a crosshair that snaps to the nearest logged step, one readout for all series.
    const cross = s("line", { class: "crosshair", y1: m.t, y2: m.t + ih, visibility: "hidden" });
    const marks = lines.map((l) => s("circle", { r: 4.5, fill: l.color, stroke: "var(--sheet)", "stroke-width": 2, visibility: "hidden" }));
    const overlay = s("rect", { x: m.l, y: m.t, width: iw, height: ih, fill: "transparent" });
    svg.append(cross, ...marks, overlay);
    const tip = h("div", { class: "tooltip", hidden: true });

    function show(step) {
      hoverStep = step;
      const hx = X(step);
      cross.setAttribute("x1", hx);
      cross.setAttribute("x2", hx);
      cross.setAttribute("visibility", "visible");
      const rows = [];
      lines.forEach((l, i) => {
        if (!l.pts.length) return marks[i].setAttribute("visibility", "hidden");
        const p = l.pts[nearest(l.pts, step)];
        marks[i].setAttribute("cx", X(p[0]));
        marks[i].setAttribute("cy", Math.min(Math.max(Y(p[1]), m.t), m.t + ih));
        marks[i].setAttribute("visibility", "visible");
        const note = p[0] !== step ? ` · step ${fmt.int(p[0])}` : "";
        rows.push(h("div", { class: "tt-row" }, h("span", { class: "key", style: { background: l.color } }), h("strong", { text: format(p[1]) }), h("span", { class: "tt-label", text: l.label + note })));
      });
      tip.replaceChildren(h("div", { class: "tt-head", text: `Step ${fmt.int(step)}` }), ...rows);
      tip.hidden = false;
      const scale = svg.getBoundingClientRect().width / W || 1;
      const left = hx * scale;
      const tipW = tip.offsetWidth || 160;
      tip.style.left = `${left + 14 + tipW > plot.clientWidth ? left - tipW - 14 : left + 14}px`;
    }
    function hide() {
      hoverStep = null;
      tip.hidden = true;
      cross.setAttribute("visibility", "hidden");
      marks.forEach((mk) => mk.setAttribute("visibility", "hidden"));
    }
    overlay.addEventListener("pointermove", (e) => {
      const rect = svg.getBoundingClientRect();
      const px = ((e.clientX - rect.left) / rect.width) * W;
      const step = x0 + ((px - m.l) / iw) * (x1 - x0);
      show(primary.pts[nearest(primary.pts, step)][0]);
    });
    overlay.addEventListener("pointerleave", hide);
    svg.addEventListener("keydown", (e) => {
      const pts = primary.pts;
      let i = hoverStep == null ? pts.length - 1 : nearest(pts, hoverStep);
      if (e.key === "ArrowLeft") i = Math.max(0, i - 1);
      else if (e.key === "ArrowRight") i = Math.min(pts.length - 1, i + 1);
      else if (e.key === "Home") i = 0;
      else if (e.key === "End") i = pts.length - 1;
      else if (e.key === "Escape") return hide();
      else return;
      e.preventDefault();
      show(pts[i][0]);
    });
    svg.addEventListener("focus", () => show(hoverStep ?? primary.pts.at(-1)[0]));
    svg.addEventListener("blur", hide);

    plot.replaceChildren(svg, tip);
    if (hoverStep != null) show(hoverStep);
  }

  function renderTable(lines) {
    const primary = lines.find((l) => l.pts.length);
    if (!primary) return tableBox.replaceChildren(h("p", { class: "muted", text: emptyText }));
    const byStep = lines.map((l) => new Map(l.pts.map((p) => [p[0], p[1]])));
    const steps = [...new Set(lines.flatMap((l) => l.pts.map((p) => p[0])))].sort((a, b) => b - a).slice(0, 40);
    tableBox.replaceChildren(h("table", { class: "grid" },
      h("thead", {}, h("tr", {}, th("Step", "num"), lines.map((l) => th(l.label, "num")))),
      h("tbody", {}, steps.map((st) => h("tr", {}, tdNum(fmt.int(st)), byStep.map((mp) => tdNum(mp.has(st) ? format(mp.get(st)) : "—")))))));
  }

  return { el, update };
}

// ---- logs and jobs ----------------------------------------------------------------------
function setLog(pre, text) {
  const atBottom = pre.scrollHeight - pre.scrollTop - pre.clientHeight < 32;
  pre.textContent = text || "No output yet.";
  if (atBottom) pre.scrollTop = pre.scrollHeight;
}

// A list of jobs that updates in place, so open logs and focus survive each refresh.
function jobList(emptyText) {
  const rows = new Map();
  const list = h("div", { class: "panel" });
  const empty = h("p", { class: "muted", text: emptyText });
  const el = h("div", {}, list, empty);
  function update(jobs) {
    list.hidden = !jobs.length;
    empty.hidden = jobs.length > 0;
    const seen = new Set();
    jobs.forEach((job, i) => {
      seen.add(job.id);
      let row = rows.get(job.id);
      if (!row) {
        row = jobRow();
        rows.set(job.id, row);
      }
      row.update(job);
      if (list.children[i] !== row.el) list.insertBefore(row.el, list.children[i] || null);
    });
    for (const [id, row] of rows) {
      if (!seen.has(id)) {
        row.el.remove();
        rows.delete(id);
      }
    }
  }
  return { el, update };
}

function jobRow() {
  let job = null;
  const badgeBox = h("span");
  const title = h("span", { class: "job-title" });
  const meta = h("span", { class: "job-meta" });
  const progress = h("div", { class: "job-progress", hidden: true });
  const logBox = h("pre", { class: "log", hidden: true });
  const stopBtn = h("button", { class: "btn small", type: "button", text: "Stop" });
  const killBtn = h("button", { class: "btn small quiet", type: "button", text: "Force stop", title: "Stop at once, without saving" });
  const logBtn = h("button", { class: "btn small quiet", type: "button", "aria-expanded": "false", text: "Show log" });
  const el = h("div", { class: "job" },
    h("div", { class: "job-line" }, badgeBox, title, meta, h("span", { class: "job-actions" }, stopBtn, killBtn, logBtn)),
    progress, logBox);

  async function refreshLog() {
    const { text } = await api(`/api/jobs/${job.id}/log?lines=400`);
    setLog(logBox, text);
  }
  logBtn.addEventListener("click", () => {
    const open = logBox.hidden;
    logBox.hidden = !open;
    logBtn.setAttribute("aria-expanded", String(open));
    logBtn.textContent = open ? "Hide log" : "Show log";
    if (open) refreshLog().catch((e) => toast(e.message, "error"));
  });
  stopBtn.addEventListener("click", async () => {
    try {
      await api(`/api/jobs/${job.id}/stop`, { method: "POST" });
      toast(job.kind === "train" ? "Stopping. Training saves a checkpoint first." : `Stopping ${job.title}.`);
    } catch (e) {
      toast(e.message, "error");
    }
  });
  killBtn.addEventListener("click", async () => {
    if (!confirm(`Force stop ${job.title}? It stops at once and saves nothing.`)) return;
    try {
      await api(`/api/jobs/${job.id}/kill`, { method: "POST" });
    } catch (e) {
      toast(e.message, "error");
    }
  });

  function update(j) {
    const wasRunning = job && job.state === "running";
    job = j;
    badgeBox.replaceChildren(jobBadge(j.state));
    title.replaceChildren(j.run ? h("a", { href: `#/runs/${enc(j.run)}`, text: j.title }) : j.title);
    const took = j.finished_at ? fmt.duration(j.finished_at - j.started_at) : null;
    const code = j.state === "failed" && j.exit_code != null ? ` · exit code ${j.exit_code}` : "";
    meta.textContent = j.state === "running" ? `started ${fmt.ago(j.started_at)}` : `${took ? `took ${took} · ` : ""}${fmt.ago(j.finished_at || j.started_at)}${code}`;
    stopBtn.hidden = killBtn.hidden = j.state !== "running";
    if (j.state === "running") {
      // The latest output line shows progress (download and tokenizing bars, eval steps).
      api(`/api/jobs/${j.id}/log?lines=1`).then(({ text }) => {
        progress.textContent = text;
        progress.hidden = !text;
      }).catch(() => {});
    } else {
      progress.hidden = true;
    }
    if (!logBox.hidden && (j.state === "running" || wasRunning)) refreshLog().catch(() => {});
  }
  return { el, update };
}

// ---- runs list --------------------------------------------------------------------------
function modelLabel(mdl) {
  return mdl && mdl.n_layer ? `${mdl.n_layer} layers · ${mdl.d_model} wide` : "";
}

function progressCell(r) {
  const p = r.max_steps ? Math.min(1, r.step / r.max_steps) : 0;
  return h("div", { class: "progress" },
    h("div", { class: "bar", role: "progressbar", "aria-label": `${r.name} progress`, "aria-valuenow": Math.round(p * 100), "aria-valuemin": 0, "aria-valuemax": 100 },
      h("span", { style: { width: `${p * 100}%` } })),
    h("span", { class: "num", text: `${fmt.int(r.step)} / ${fmt.int(r.max_steps)}` }));
}

function runsPage() {
  const tbody = h("tbody");
  const table = h("div", { class: "panel table-wrap", hidden: true },
    h("table", { class: "grid" },
      h("thead", {}, h("tr", {}, th("Run"), th("State"), th("Progress"), th("Loss", "num"), th("Val loss", "num"), th("HellaSwag", "num"), th("Speed", "num"), th("Time left", "num"), th("Updated", "num"))),
      tbody));
  const empty = h("div", { class: "panel empty", hidden: true },
    h("h3", { text: "No training runs yet" }),
    h("p", { text: "Start with the tiny CPU preset to check that everything works, then prepare FineWeb-Edu for the full run." }),
    h("div", { class: "actions" }, h("a", { class: "btn primary", href: "#/new?preset=tiny_cpu", text: "New tiny run" }), h("a", { class: "btn", href: "#/data", text: "Prepare data" })));
  const jobs = jobList("Dataset prep and evaluations appear here.");
  main.append(pageHead("Runs", null, h("a", { class: "btn primary", href: "#/new", text: "New run" })), table, empty, h("h2", { text: "Other jobs" }), jobs.el);

  every(4000, async () => {
    const [runs, allJobs] = await Promise.all([api("/api/runs"), api("/api/jobs?limit=30")]);
    table.hidden = !runs.length;
    empty.hidden = runs.length > 0;
    tbody.replaceChildren(...runs.map((r) => {
      const go = () => (location.hash = `#/runs/${enc(r.name)}`);
      return h("tr", { class: "link", onclick: go },
        h("td", {}, h("a", { class: "run-name", href: `#/runs/${enc(r.name)}`, text: r.name }), h("div", { class: "sub", text: modelLabel(r.model) })),
        h("td", {}, runBadge(r.state)),
        h("td", {}, progressCell(r)),
        h("td", { class: "num" }, h("span", { class: "spark-cell" }, sparkline(r.loss_trend), fmt.loss(r.loss))),
        tdNum(fmt.loss(r.val_loss)),
        tdNum(fmt.pct(r.hellaswag)),
        tdNum(LIVE.has(r.state) ? fmt.rate(r.tokens_per_sec) : "—"),
        tdNum(fmt.duration(r.eta_seconds)),
        tdNum(fmt.ago(r.updated_at)));
    }));
    jobs.update(allJobs.filter((j) => j.kind !== "train").slice(0, 12));
  });
}

// ---- one run ----------------------------------------------------------------------------
function titleBlock(name) {
  const v = {};
  const cell = (key, label, cls = "") => {
    v[key] = h("span", { class: `tb-value ${cls}` });
    return h("div", { class: `tb-cell ${key === "name" ? "tb-run" : ""}` }, h("span", { class: "tb-label", text: label }), v[key]);
  };
  const facts = h("div", { class: "facts-line", style: { display: "contents" } });
  const actions = h("div", { class: "actions" });
  const scaleFill = h("span", { style: { width: "0%" } });
  const scale = h("div", { class: "tb-progress", role: "progressbar", "aria-label": "Training progress", "aria-valuemin": 0, "aria-valuemax": 100 }, scaleFill);
  const el = h("section", { "aria-label": "Run summary" },
    h("div", { class: "titleblock" },
      h("div", { class: "tb-row r1" }, cell("name", "Run", "tb-name"), cell("state", "State"), cell("step", "Step"), cell("eta", "Time left")),
      h("div", { class: "tb-row r2" }, cell("model", "Layers × width"), cell("params", "Parameters"), cell("tokens", "Tokens"), cell("loss", "Loss"), cell("val", "Val loss"), cell("hs", "HellaSwag")),
      h("div", { class: "tb-foot" }, facts, actions)),
    scale);
  v.name.textContent = name;
  let actionsKey = null;

  function update(d, handlers) {
    v.state.replaceChildren(runBadge(d.state));
    v.step.replaceChildren(fmt.int(d.step), h("small", { text: ` / ${fmt.int(d.max_steps)}` }));
    v.eta.textContent = LIVE.has(d.state) ? fmt.duration(d.eta_seconds) : d.state === "finished" ? "Done" : "—";
    const mdl = d.model || {};
    v.model.textContent = mdl.n_layer ? `${mdl.n_layer} × ${mdl.d_model}` : "—";
    v.params.textContent = d.model_stats ? fmt.compact(d.model_stats.params) : "—";
    v.tokens.replaceChildren(fmt.compact(d.tokens), h("small", { text: ` / ${fmt.compact(d.total_tokens)}` }));
    v.loss.textContent = fmt.loss(d.loss);
    v.val.textContent = fmt.loss(d.val_loss);
    v.hs.textContent = fmt.pct(d.hellaswag);
    const p = d.max_steps ? Math.min(1, d.step / d.max_steps) : 0;
    scaleFill.style.width = `${p * 100}%`;
    scale.setAttribute("aria-valuenow", String(Math.round(p * 100)));
    scale.title = `${(p * 100).toFixed(1)}% of ${fmt.int(d.max_steps)} steps`;
    const fact = (k, val) => (val ? h("span", {}, h("b", { text: k }), val) : null);
    const lastCk = d.checkpoints.at(-1);
    fill(facts,
      fact("REV", d.commit ? d.commit.slice(0, 7) : null),
      fact("STARTED", d.started_at ? fmt.date(d.started_at) : null),
      fact("DEVICE", d.device),
      fact("CHECKPOINT", lastCk ? `step ${fmt.int(lastCk.step)}` : "none yet"),
      fact("SPEED", LIVE.has(d.state) ? fmt.rate(d.tokens_per_sec) : null));

    const key = `${d.state}|${d.checkpoints.length > 0}`;
    if (key === actionsKey) return;
    actionsKey = key;
    const btns = [];
    if (LIVE.has(d.state)) btns.push(h("button", { class: "btn", type: "button", text: "Stop and save", onclick: handlers.stop }));
    else if (d.state !== "finished" && d.checkpoints.length) btns.push(h("button", { class: "btn primary", type: "button", text: "Resume", onclick: handlers.resume }));
    if (d.checkpoints.length) {
      btns.push(h("a", { class: "btn quiet", href: `#/evaluate?run=${enc(name)}`, text: "Evaluate" }));
      btns.push(h("a", { class: "btn quiet", href: `#/generate?run=${enc(name)}`, text: "Generate text" }));
    }
    actions.replaceChildren(...btns);
  }
  return { el, update };
}

function configList(cfg) {
  if (!cfg) return h("p", { class: "muted", text: "No config.yaml in this run." });
  return h("div", { class: "table-wrap" }, h("table", { class: "grid" }, h("tbody", {},
    Object.entries(cfg).flatMap(([section, values]) =>
      Object.entries(values || {}).map(([k, val]) =>
        h("tr", {}, h("td", { class: "mono", style: { color: "var(--muted)", fontSize: "12.5px" }, text: `${section}.${k}` }), h("td", { class: "num", style: { textAlign: "left" }, text: JSON.stringify(val) })))))));
}

function runPage(name) {
  document.title = `${name} · GPT-Lab`;
  const tb = titleBlock(name);
  const banner = h("div", { class: "banner", hidden: true });
  const charts = [
    lineChart({ title: "Loss", clip: true, format: fmt.loss, tick: (t) => String(+t.toFixed(3)), series: [
      { key: "train/loss", label: "Train", color: "var(--series-1)" },
      { key: "eval/val_loss", label: "Validation", color: "var(--series-2)", dots: true },
    ] }),
    lineChart({ title: "HellaSwag accuracy", format: fmt.pct, tick: (t) => `${(t * 100).toFixed(0)}%`,
      emptyText: "No HellaSwag scores yet. The run scores it every hellaswag_every steps.",
      series: [{ key: "eval/hellaswag_acc_norm", label: "acc_norm", color: "var(--series-1)", dots: true }] }),
    lineChart({ title: "Throughput", zero: true, format: fmt.rate, tick: fmt.compact,
      series: [{ key: "perf/tokens_per_sec", label: "Tokens per second", color: "var(--series-1)" }] }),
    lineChart({ title: "Learning rate", zero: true, format: fmt.sci,
      series: [
        { key: "train/lr", label: "Actual", color: "var(--series-1)" },
        { key: "plan/lr", label: "Planned", color: "var(--series-2)", track: true },
      ] }),
  ];
  const model3d = modelView();
  const params3d = paramBar();
  onLeave(() => model3d.destroy());
  let plan = { points: [], marks: [] };
  const samples = h("div", { class: "panel" });
  const logPre = h("pre", { class: "log panel" });
  const ckBox = h("div", { class: "panel table-wrap" });
  const cfgBox = h("div", { class: "panel", style: { maxHeight: "420px", overflow: "auto" } });
  const evalBox = h("div");
  main.append(
    h("nav", { class: "crumbs", "aria-label": "Breadcrumb" }, h("a", { href: "#/runs", text: "Runs" }), " / ", name),
    tb.el, banner,
    h("div", { class: "charts" }, charts.map((c) => c.el)),
    h("section", {}, h("h2", { text: "Model" }), h("div", { class: "panel model-panel" }, model3d.el, params3d.el)),
    h("div", { class: "two-col" },
      h("section", {}, h("h2", { text: "Samples" }), samples),
      h("section", {}, h("h2", { text: "Log" }), logPre)),
    evalBox,
    h("div", { class: "two-col" },
      h("section", {}, h("h2", { text: "Checkpoints" }), ckBox),
      h("section", {}, h("h2", { text: "Configuration" }), cfgBox)));

  const handlers = {
    async stop() {
      try {
        await api(`/api/runs/${enc(name)}/stop`, { method: "POST" });
        toast("Stopping. The run saves a checkpoint first.");
      } catch (e) {
        toast(e.message, "error");
      }
    },
    async resume() {
      try {
        await api(`/api/runs/${enc(name)}/resume`, { method: "POST" });
        toast(`Resuming ${name} from its last checkpoint.`);
      } catch (e) {
        toast(e.message, "error");
      }
    },
  };

  const series = {};
  let cursor = null;
  let cfgShown = false;
  let lastState = null;
  let ckKey = null;
  let evalKey = null;
  every(4000, async () => {
    let d;
    try {
      d = await api(`/api/runs/${enc(name)}`);
    } catch (e) {
      banner.hidden = false;
      banner.replaceChildren(ICONS.error(), h("span", { text: e.message }));
      return;
    }
    tb.update(d, handlers);
    document.title = `${name} · ${(RUN_STATES[d.state] || [d.state])[0]} · GPT-Lab`;

    banner.hidden = !(d.state === "failed" || d.state === "crashed");
    if (!banner.hidden) {
      const lastCk = d.checkpoints.at(-1);
      const resume = lastCk ? ` Resume continues from step ${fmt.int(lastCk.step)}.` : "";
      banner.replaceChildren(ICONS.error(), h("span", {
        text: d.state === "failed"
          ? `Training stopped with an error: ${d.error || "see the log"}.${resume}`
          : `The training process ended without reporting why (killed, out of memory, or WSL shut down). The log shows its last output.${resume}`,
      }));
    }

    const res = await api(`/api/runs/${enc(name)}/metrics${cursor ? `?cursor=${enc(cursor)}` : ""}`);
    if (res.reset) for (const k of Object.keys(series)) delete series[k];
    for (const row of res.rows) {
      for (const [k, val] of Object.entries(row)) {
        if (k === "step" || k === "time" || typeof val !== "number") continue;
        (series[k] ||= []).push([row.step, val]);
      }
    }
    if (res.cursor) cursor = res.cursor;
    if (d.config && !plan.points.length) {
      plan = lrPlan(d.config.train);
      model3d.update(d.config.model);
      params3d.update(modelSpec(d.config.model));
    }
    charts.forEach((c, i) => c.update(i === 3 ? { ...series, "plan/lr": plan.points } : series,
      { xMax: d.max_steps, marks: i === 3 ? plan.marks : [] }));

    const prompts = (d.config && d.config.train && d.config.train.sample_prompts) || [];
    samples.replaceChildren(...(d.samples.length ? d.samples.map((smp) => {
      const i = Number(smp.tag.split("/")[1]);
      const prompt = prompts[i] || "";
      const text = smp.text || "";
      const body = prompt && text.startsWith(prompt)
        ? [h("span", { class: "prompt", text: prompt }), text.slice(prompt.length)]
        : [text];
      return h("div", { class: "sample" }, h("span", { class: "meta", text: `PROMPT ${i + 1} · STEP ${fmt.int(smp.step)}` }), h("p", {}, body));
    }) : [h("p", { class: "empty muted", text: "No samples yet. The run writes some every sample_every steps." })]));

    if (d.has_log && (LIVE.has(d.state) || lastState !== d.state)) {
      const { text } = await api(`/api/runs/${enc(name)}/log?lines=300`);
      setLog(logPre, text);
    } else if (!d.has_log) {
      logPre.textContent = "No log. Runs started from a terminal print to that terminal instead.";
    }
    lastState = d.state;

    const newCkKey = d.checkpoints.map((c) => c.file).join(",");
    if (newCkKey !== ckKey) {
      ckKey = newCkKey;
      ckBox.replaceChildren(d.checkpoints.length
        ? h("table", { class: "grid" },
            h("thead", {}, h("tr", {}, th("Step", "num"), th("File"), th("Size", "num"), th("Saved", "num"))),
            h("tbody", {}, [...d.checkpoints].reverse().map((c) =>
              h("tr", {}, tdNum(fmt.int(c.step)), h("td", { class: "mono", text: c.file }), tdNum(fmt.bytes(c.bytes)), tdNum(fmt.ago(c.mtime))))))
        : h("p", { class: "empty muted", text: "No checkpoints yet. The run saves one every ckpt_every steps, and whenever it's stopped." }));
    }
    if (!cfgShown && d.config) {
      cfgShown = true;
      cfgBox.replaceChildren(configList(d.config));
    }
    const newEvalKey = JSON.stringify(d.eval);
    if (newEvalKey !== evalKey) {
      evalKey = newEvalKey;
      fill(evalBox, d.eval ? h("section", {},
        h("h2", { text: "Evaluation" }),
        h("div", { class: "panel card" },
          h("div", { class: "stats" },
            stat("Checkpoint", `step ${fmt.int(d.eval.step)}`),
            stat("Val loss", fmt.loss(d.eval.val_loss)),
            stat("HellaSwag", fmt.pct(d.eval.hellaswag_acc_norm)),
            ...Object.entries(d.eval.lm_eval || {}).map(([t, vals]) => stat(t, fmt.pct(vals["acc_norm,none"] ?? vals["acc,none"])))),
          h("div", {}, h("a", { href: "#/evaluate", text: "Compare with GPT-2" })))) : null);
    }
  });
}

function stat(k, v) {
  return h("div", { class: "stat" }, h("span", { class: "k", text: k }), h("span", { class: "v", text: v }));
}

// ---- new run ----------------------------------------------------------------------------
// ---- shared visual helpers --------------------------------------------------------------
// The learning rate at a 0-indexed step; mirrors gpt_lab/schedule.py's lr_at.
function lrAt(step, t) {
  const maxLr = t.lr;
  const total = t.max_steps;
  const warm = t.warmup_steps;
  const minLr = maxLr * (t.min_lr_ratio || 0);
  if (warm > 0 && step < warm) return (maxLr * (step + 1)) / warm;
  if (t.schedule === "cosine") {
    const p = Math.min(Math.max((step - warm) / Math.max(1, total - warm), 0), 1);
    return minLr + 0.5 * (1 + Math.cos(Math.PI * p)) * (maxLr - minLr);
  }
  const decay = Math.max(1, Math.round(t.decay_frac * total));
  const start = total - decay;
  if (step < start) return maxLr;
  return maxLr - (maxLr - minLr) * Math.min((step - start + 1) / decay, 1);
}

// The planned schedule as chart points, plus marks where its phases change.
function lrPlan(t) {
  const ok = t && [t.lr, t.max_steps, t.warmup_steps, t.decay_frac].every((v) => typeof v === "number" && isFinite(v));
  if (!ok || t.max_steps < 1 || t.max_steps > 1e8) return { points: [], marks: [] };
  const total = t.max_steps;
  const n = Math.min(total, 320);
  const steps = new Set(Array.from({ length: n }, (_, i) => Math.round((i * (total - 1)) / Math.max(1, n - 1))));
  const marks = [];
  if (t.warmup_steps > 0 && t.warmup_steps < total) {
    steps.add(t.warmup_steps - 1).add(t.warmup_steps);
    marks.push({ x: t.warmup_steps, label: `warmup ends ${fmt.int(t.warmup_steps)}` });
  }
  if (t.schedule !== "cosine") {
    const start = total - Math.max(1, Math.round(t.decay_frac * total));
    if (start > 0 && start > t.warmup_steps) {
      steps.add(start - 1).add(start);
      marks.push({ x: start, label: `decay starts ${fmt.int(start)}` });
    }
  }
  const points = [...steps].filter((st) => st >= 0 && st < total).sort((a, b) => a - b).map((st) => [st, lrAt(st, t)]);
  return { points, marks };
}

// Two bars on one scale: tokens the run will read, and tokens prepared on disk.
function budgetBars() {
  const el = h("div", { class: "budget" });
  function update(read, available) {
    if (!read) return fill(el);
    const max = Math.max(read, available || 0);
    const row = (label, value, color, note) =>
      h("div", { class: "budget-row" },
        h("span", { class: "budget-label", text: label }),
        h("div", { class: "budget-track" }, h("span", { class: "budget-bar", style: { width: `${(100 * value) / max}%`, background: color } })),
        h("span", { class: "budget-value", text: note }));
    const passes = available ? read / available : null;
    fill(el,
      h("div", { class: "budget-title", text: "Training data budget" }),
      row("Run reads", read, "var(--series-1)", fmt.compact(read)),
      row("Prepared", available || 0, "var(--series-3)", available ? fmt.compact(available) : "none yet"),
      h("p", { class: "sub", text: passes == null
        ? "Prepare a dataset to train on."
        : passes <= 1
          ? `The run reads ${(passes * 100).toFixed(0)}% of the prepared data, so it sees each token at most once.`
          : `The run reads the prepared data ${passes.toFixed(2)} times over, so it repeats examples.` }));
  }
  return { el, update };
}

// A small trend line for table rows.
function sparkline(points, w = 72, ht = 20) {
  if (!points || points.length < 2) return null;
  const xs = points.map((p) => p[0]);
  const ys = points.map((p) => p[1]);
  const [xa, xb] = [Math.min(...xs), Math.max(...xs)];
  const [ya, yb] = [Math.min(...ys), Math.max(...ys)];
  const X = (v) => 1 + ((v - xa) / (xb - xa || 1)) * (w - 6);
  const Y = (v) => 2 + (1 - (v - ya) / (yb - ya || 1)) * (ht - 4);
  const d = "M" + points.map((p) => `${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join("L");
  const last = points.at(-1);
  return s("svg", { class: "spark", viewBox: `0 0 ${w} ${ht}`, width: w, height: ht, "aria-hidden": "true" },
    s("path", { d, fill: "none", stroke: "var(--series-1)", "stroke-width": 1.5, "stroke-linejoin": "round" }),
    s("circle", { cx: X(last[0]), cy: Y(last[1]), r: 2.5, fill: "var(--series-1)" }));
}

const LABELS = {
  "model.n_layer": "Layers",
  "model.d_model": "Width",
  "model.n_head": "Attention heads",
  "model.n_kv_head": "Key/value heads",
  "model.seq_len": "Context length",
  "data.train_pattern": "Training shards",
  "data.val_pattern": "Validation shards",
  "train.device": "Device",
  "train.max_steps": "Steps",
  "train.micro_batch_size": "Micro batch size",
  "train.total_batch_tokens": "Tokens per step",
  "train.lr": "Peak learning rate",
  "train.warmup_steps": "Warmup steps",
  "train.schedule": "Learning rate schedule",
  "train.decay_frac": "Decay fraction",
  "train.eval_every": "Validate every",
  "train.hellaswag_every": "HellaSwag every",
  "train.sample_every": "Sample every",
  "train.ckpt_every": "Checkpoint every",
};
const HINTS = {
  "model.n_kv_head": "Empty for standard attention",
  "train.micro_batch_size": "16 fits the RTX 5080 at 124M",
  "train.eval_every": "Steps; 0 turns it off",
  "train.hellaswag_every": "Steps; 0 turns it off",
  "train.sample_every": "Steps; 0 turns it off",
  "train.ckpt_every": "Steps; stopping always saves",
};
const GROUPS = [
  ["Model", ["model.n_layer", "model.d_model", "model.n_head", "model.n_kv_head", "model.seq_len"]],
  ["Data", ["data.train_pattern", "data.val_pattern"]],
  ["Training", ["train.device", "train.max_steps", "train.micro_batch_size", "train.total_batch_tokens", "train.lr", "train.warmup_steps", "train.schedule", "train.decay_frac"]],
  ["Evaluation and checkpoints", ["train.eval_every", "train.hellaswag_every", "train.sample_every", "train.ckpt_every"]],
];

function getKey(cfg, key) {
  const [section, name] = key.split(".");
  return cfg[section] ? cfg[section][name] : undefined;
}

function parseValue(field, text) {
  const t = text.trim();
  if (t === "" && field.optional) return { value: null };
  if (field.type === "int") {
    const n = Number(t.replaceAll(",", "").replaceAll("_", ""));
    return Number.isInteger(n) && t !== "" ? { value: n } : { error: "Enter a whole number." };
  }
  if (field.type === "float") {
    const n = Number(t.replaceAll("_", ""));
    return t !== "" && isFinite(n) ? { value: n } : { error: "Enter a number, like 0.001 or 1e-3." };
  }
  return { value: text };
}

function timestamp() {
  const d = new Date();
  const p = (n) => String(n).padStart(2, "0");
  return `${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}`;
}

async function newRunPage(params) {
  const token = routeToken;
  main.append(pageHead("New run", "Pick a preset, change what you need, and start. The checks on the right update as you type."));
  const [presets, schema] = await Promise.all([api("/api/presets"), api("/api/schema")]);
  if (!stillOn(token)) return;
  const valid = presets.filter((p) => p.config);
  if (!valid.length) {
    main.append(h("div", { class: "panel empty" }, h("h3", { text: "No presets" }), h("p", { text: "Add a YAML config to the configs/ folder, then reload." })));
    return;
  }
  const fields = Object.fromEntries(Object.values(schema).flat().map((f) => [f.key, f]));
  let preset = valid.find((p) => p.name === params.get("preset")) || valid[0];
  let values = {};
  let invalid = new Set();

  // Visuals that follow the form: the model in 3D with its parameter breakdown, the
  // learning-rate schedule, and the data budget. Built once and moved into their sections.
  const model3d = modelView();
  const params3d = paramBar();
  const lrChart = lineChart({
    title: "Learning rate schedule", zero: true, format: fmt.sci,
    emptyText: "Set steps, warmup and the learning rate to see the schedule.",
    headline: (lines) => (lines[0].pts.length ? `peak ${fmt.sci(Math.max(...lines[0].pts.map((p) => p[1])))}` : ""),
    series: [{ key: "lr", label: "Learning rate", color: "var(--series-1)" }],
  });
  const budget = budgetBars();
  onLeave(() => model3d.destroy());
  const sectionValues = (name) => Object.fromEntries(Object.entries(values)
    .filter(([key]) => key.startsWith(`${name}.`))
    .map(([key, val]) => [key.slice(name.length + 1), val]));
  function refreshVisuals() {
    const mc = sectionValues("model");
    model3d.update(mc);
    params3d.update(modelSpec(mc));
    const plan = lrPlan(sectionValues("train"));
    lrChart.update({ lr: plan.points }, { xMax: values["train.max_steps"], marks: plan.marks });
  }

  const runName = h("input", { class: "input", id: "run-name", spellcheck: "false", autocomplete: "off" });
  const presetBox = h("div", { class: "presets", role: "radiogroup", "aria-label": "Preset" });
  const groupsBox = h("div");
  const facts = h("dl", { class: "facts" });
  const problems = h("ul", { class: "problems" });
  const changedNote = h("p", { class: "note" });
  const startBtn = h("button", { class: "btn primary start", type: "button", text: "Start training", disabled: true });
  const panel = h("aside", { class: "panel checkpanel", "aria-live": "polite" },
    h("h3", { text: "Before you start" }), facts, problems, startBtn, changedNote,
    h("p", { class: "note", text: "Training runs in the background, so you can close this page. It keeps going while the GPT-Lab server is running." }));
  main.append(h("div", { class: "new-grid" },
    h("div", {},
      h("section", { class: "form-section" }, h("h2", { text: "Preset" }), presetBox),
      h("section", { class: "form-section" }, h("h2", { text: "Name" }),
        h("div", { class: "panel fields", style: { gridTemplateColumns: "minmax(0, 1fr)" } },
          h("div", { class: "field" }, h("label", { for: "run-name", text: "Run name" }), runName,
            h("span", { class: "hint", text: "Letters, digits, dots, dashes and underscores. The run's folder is runs/<name>." })))),
      groupsBox),
    panel));

  function renderPresets() {
    presetBox.replaceChildren(...valid.map((p) => h("button", {
      class: "preset", type: "button", role: "radio", "aria-checked": String(p === preset),
      onclick: () => selectPreset(p),
    }, h("span", { class: "name", text: p.name }), h("span", { class: "desc", text: p.description || "" }))));
  }

  function fieldEl(f) {
    const id = `f-${f.key.replace(".", "-")}`;
    const label = LABELS[f.key] || f.key.split(".")[1];
    const presetValue = getKey(preset.config, f.key);
    const value = values[f.key];
    const changed = h("span", { class: "changed", hidden: JSON.stringify(value) === JSON.stringify(presetValue) });
    changed.textContent = `changed from ${presetValue == null ? "none" : JSON.stringify(presetValue)}`;
    const wrap = h("div", { class: f.type === "str" && !f.choices ? "field full" : "field" });
    const set = (val) => {
      values[f.key] = val;
      changed.hidden = JSON.stringify(val) === JSON.stringify(presetValue);
      refreshVisuals();
      scheduleCheck();
    };
    let input;
    if (f.type === "bool") {
      input = h("input", { type: "checkbox", id, checked: !!value, onchange: (e) => set(e.target.checked) });
    } else if (f.choices) {
      input = h("select", { class: "input", id, onchange: (e) => set(e.target.value) },
        f.choices.map((c) => h("option", { value: c, selected: c === value, text: c })));
    } else if (f.type === "list") {
      input = h("textarea", { class: "input", id, rows: 3, spellcheck: "false", oninput: (e) => set(e.target.value.split("\n").filter((l) => l.trim())) });
      input.value = (value || []).join("\n");
    } else {
      input = h("input", {
        class: "input", id, spellcheck: "false", autocomplete: "off",
        inputmode: f.type === "int" || f.type === "float" ? "decimal" : null,
        placeholder: f.optional ? "none" : null,
        oninput: (e) => {
          const parsed = parseValue(f, e.target.value);
          wrap.classList.toggle("invalid", !!parsed.error);
          errorEl.hidden = !parsed.error;
          errorEl.textContent = parsed.error || "";
          if (parsed.error) {
            invalid.add(f.key);
            renderStartState();
            return;
          }
          invalid.delete(f.key);
          set(parsed.value);
        },
      });
      input.value = value == null ? "" : String(value);
    }
    const errorEl = h("span", { class: "hint", hidden: true, style: { color: "var(--ink)" } });
    append(wrap, [h("label", { for: id, text: label }), h("span", { class: "fkey", text: f.key }), input, errorEl, HINTS[f.key] ? h("span", { class: "hint", text: HINTS[f.key] }) : null, changed]);
    return wrap;
  }

  function renderFields() {
    const shown = new Set(GROUPS.flatMap(([, keys]) => keys));
    const rest = Object.keys(fields).filter((k) => !shown.has(k));
    const visual = {
      Model: () => h("div", { class: "panel model-panel" }, model3d.el, params3d.el),
      Data: () => h("div", { class: "panel viz-panel" }, budget.el),
      Training: () => lrChart.el,
    };
    groupsBox.replaceChildren(
      ...GROUPS.map(([title, keys]) => h("section", { class: "form-section" },
        h("h2", { text: title }),
        title === "Model" ? visual.Model() : null,
        h("div", { class: "panel fields" }, keys.filter((k) => fields[k]).map((k) => fieldEl(fields[k]))),
        title !== "Model" && visual[title] ? h("div", { class: "viz-after" }, visual[title]()) : null)),
      h("details", { class: "more" }, h("summary", { text: `All other settings (${rest.length})` }),
        h("div", { class: "panel fields", style: { marginTop: "12px" } }, rest.map((k) => fieldEl(fields[k])))));
  }

  function selectPreset(p) {
    preset = p;
    values = Object.fromEntries(Object.keys(fields).map((k) => [k, getKey(p.config, k)]));
    invalid = new Set();
    if (!runName.dataset.edited) runName.value = `${p.name}-${timestamp()}`;
    renderPresets();
    renderFields();
    refreshVisuals();
    scheduleCheck(0);
  }

  function overrides() {
    const out = {};
    for (const [k, val] of Object.entries(values)) {
      if (JSON.stringify(val) !== JSON.stringify(getKey(preset.config, k))) out[k] = val;
    }
    return out;
  }

  let report = null;
  let checking = false;
  let timer = null;
  let seq = 0;
  function renderStartState() {
    const blocked = checking || invalid.size > 0 || !report || report.errors.length > 0;
    startBtn.disabled = blocked;
    const n = Object.keys(overrides()).length;
    changedNote.textContent = n ? `${n} setting${n === 1 ? "" : "s"} changed from ${preset.name}.` : `Using ${preset.name} as it is.`;
  }
  function scheduleCheck(delay = 300) {
    clearTimeout(timer);
    checking = true;
    renderStartState();
    timer = setTimeout(runCheck, delay);
  }
  onLeave(() => clearTimeout(timer));
  runName.addEventListener("input", () => {
    runName.dataset.edited = "1";
    scheduleCheck();
  });

  async function runCheck() {
    const mine = ++seq;
    try {
      const r = await api("/api/runs/check", { method: "POST", body: { preset: preset.name, run_name: runName.value.trim(), overrides: overrides() } });
      if (mine !== seq || !stillOn(token)) return;
      report = r;
    } catch (e) {
      if (mine !== seq) return;
      report = { errors: [e.message], warnings: [], derived: null };
    }
    checking = false;
    const d = report.derived || {};
    budget.update(d.total_tokens, d.train_tokens);
    const row = (k, val) => [h("dt", { text: k }), h("dd", { text: val })];
    facts.replaceChildren(...[
      row("Parameters", fmt.compact(d.params)),
      row("Tokens per step", fmt.int(d.tokens_per_step)),
      row("Gradient accumulation", d.grad_accum ? `${d.grad_accum} micro batches` : "—"),
      row("Total tokens", fmt.compact(d.total_tokens)),
      row("Tokens per parameter", d.tokens_per_param ? d.tokens_per_param.toFixed(1) : "—"),
      row("Training data", d.train_shards ? `${fmt.compact(d.train_tokens)} in ${d.train_shards} shard${d.train_shards === 1 ? "" : "s"}` : "none found"),
      row("Device", d.device || "—"),
    ].flat());
    problems.replaceChildren(
      ...report.errors.map((msg) => h("li", { class: "problem" }, ICONS.error(), h("span", { text: msg }))),
      ...report.warnings.map((msg) => h("li", { class: "problem" }, ICONS.warn(), h("span", { text: msg }))));
    if (!report.errors.length && !report.warnings.length) problems.replaceChildren(h("li", { class: "problem" }, ICONS.done(), h("span", { text: "Ready to start." })));
    renderStartState();
  }

  startBtn.addEventListener("click", async () => {
    startBtn.disabled = true;
    const name = runName.value.trim();
    try {
      await api("/api/runs", { method: "POST", body: { preset: preset.name, run_name: name, overrides: overrides() } });
      toast(`Training ${name} started.`);
      location.hash = `#/runs/${enc(name)}`;
    } catch (e) {
      toast(e.message, "error");
      scheduleCheck(0);
    }
  });

  selectPreset(preset);
}

// ---- data -------------------------------------------------------------------------------
function dataPage() {
  main.append(pageHead("Data", "Training reads token shards from the data/ folder. Tiny Shakespeare takes seconds and suits the CPU smoke test. FineWeb-Edu is the real dataset: about 10 billion tokens, a few hours to prepare, and roughly 20 GB of shards."));
  const cards = h("div", { class: "cards" });
  const storage = h("div", { class: "panel card" });
  const jobs = jobList("No dataset prep has run yet.");
  main.append(cards, h("h2", { text: "Storage" }), storage, h("h2", { text: "Prep jobs" }), jobs.el);
  const cardByDir = new Map();
  let prepRunning = false;

  function makeCard(ds) {
    const statsBox = h("div", { class: "stats" });
    const status = h("span", { class: "sub" });
    const maxTokens = ds.key === "fineweb" ? h("input", { class: "input", placeholder: "all", inputmode: "numeric", "aria-label": "Tokens to prepare", title: "Leave empty to prepare all of FineWeb-Edu 10B" }) : null;
    const replace = h("input", { type: "checkbox" });
    const replaceRow = h("label", { class: "check" }, replace, "Replace the existing shards");
    const btn = h("button", { class: "btn primary", type: "button", text: "Prepare" });
    btn.addEventListener("click", async () => {
      const body = { dataset: ds.key, overwrite: replace.checked };
      if (maxTokens && maxTokens.value.trim()) {
        const n = Number(maxTokens.value.replaceAll(",", "").replaceAll("_", ""));
        if (!Number.isInteger(n) || n <= 0) return toast("Tokens to prepare must be a whole number, like 300000000.", "error");
        body.max_tokens = n;
      }
      btn.disabled = true;
      try {
        await api("/api/data/prepare", { method: "POST", body });
        toast(`Preparing ${ds.title}. Follow it under Prep jobs.`);
        replace.checked = false;
      } catch (e) {
        toast(e.message, "error");
      }
      refresh();
    });
    const actions = ds.key
      ? h("div", { class: "row" }, maxTokens ? h("label", { class: "field" }, h("span", { class: "fkey", text: "Tokens to prepare" }), maxTokens) : null, replaceRow, btn)
      : h("p", { class: "sub", text: "Found in data/. The UI doesn't prepare this one." });
    const prepFill = h("span");
    const prepText = h("span", { class: "sub" });
    const prepBar = h("div", { class: "prep-progress", hidden: true },
      h("div", { class: "bar", role: "progressbar", "aria-label": `${ds.title} tokens prepared` }, prepFill), prepText);
    const strip = h("div", { class: "shard-strip", role: "img" });
    const stripLegend = h("div", { class: "chart-legend", hidden: true },
      h("span", {}, h("span", { class: "pbar-swatch", style: { background: "var(--series-1)" } }), "validation shard"),
      h("span", {}, h("span", { class: "pbar-swatch", style: { background: "var(--series-3)" } }), "training shard"));
    const el = h("section", { class: "panel card" },
      h("div", {}, h("h3", { text: ds.title }), h("span", { class: "path", text: ds.dir })), status, prepBar,
      h("div", {}, strip, stripLegend), statsBox, actions);
    return {
      el,
      update(d, job) {
        // One cell per shard file: validation first (it's written first), then training.
        const cells = d.val_shards + d.train_shards;
        strip.hidden = stripLegend.hidden = !cells;
        if (strip.childElementCount !== cells) {
          strip.replaceChildren(...Array.from({ length: Math.min(cells, 400) }, (_, i) =>
            h("span", { class: `shard ${i < d.val_shards ? "val" : "train"}` })));
        }
        strip.setAttribute("aria-label", `${d.val_shards} validation and ${d.train_shards} training shards`);
        strip.title = `${d.val_shards} validation shard${d.val_shards === 1 ? "" : "s"} (blue), ${d.train_shards} training (green)`;
        // FineWeb-Edu shows how far its prep has got: shards land 100M tokens at a time.
        if (ds.key === "fineweb") {
          const prepared = d.train_tokens + d.val_tokens;
          const i = job ? job.cmd.indexOf("--max-tokens") : -1;
          const target = i >= 0 ? Number(job.cmd[i + 1]) : 1e10;
          const show = !!job || (prepared > 0 && prepared < target * 0.95);
          prepBar.hidden = !show;
          if (show) {
            const p = Math.min(1, prepared / target);
            prepFill.style.width = `${p * 100}%`;
            prepBar.firstChild.setAttribute("aria-valuenow", String(Math.round(p * 100)));
            prepText.textContent = `${fmt.compact(prepared)} of ${i >= 0 ? "" : "about "}${fmt.compact(target)} tokens${job ? " · preparing" : " · stopped early"}`;
          }
        }
        status.textContent = d.train_shards || d.val_shards ? "" : "Not prepared yet.";
        status.hidden = !!(d.train_shards || d.val_shards);
        statsBox.replaceChildren(
          stat("Training tokens", fmt.compact(d.train_tokens)),
          stat("Validation tokens", fmt.compact(d.val_tokens)),
          stat("Shards", `${d.train_shards} + ${d.val_shards}`),
          stat("On disk", fmt.bytes(d.bytes)));
        replaceRow.hidden = !(d.train_shards || d.val_shards);
        btn.textContent = d.train_shards || d.val_shards ? "Prepare again" : "Prepare";
        btn.disabled = prepRunning;
        btn.title = prepRunning ? "Another dataset is being prepared" : "";
      },
    };
  }

  async function refresh() {
    const [data, prepJobs] = await Promise.all([api("/api/data"), api("/api/jobs?kind=prepare&limit=20")]);
    prepRunning = prepJobs.some((j) => j.state === "running");
    data.datasets.forEach((ds, i) => {
      let card = cardByDir.get(ds.dir);
      if (!card) {
        card = makeCard(ds);
        cardByDir.set(ds.dir, card);
      }
      const job = prepJobs.find((j) => j.state === "running" && j.title.includes(ds.title));
      card.update(ds, job);
      if (cards.children[i] !== card.el) cards.insertBefore(card.el, cards.children[i] || null);
    });
    const used = data.disk.total - data.disk.free;
    storage.replaceChildren(
      h("div", { class: "disk" },
        h("div", { class: "bar", role: "img", "aria-label": `Disk ${Math.round((100 * used) / data.disk.total)}% used` },
          h("span", { style: { width: `${(100 * used) / data.disk.total}%` } })),
        h("span", { class: "sub", text: `${fmt.bytes(used)} used of ${fmt.bytes(data.disk.total)}` })),
      h("div", { class: "stats" },
      stat("Free disk space", `${fmt.bytes(data.disk.free)} of ${fmt.bytes(data.disk.total)}`),
      stat("Download cache", fmt.bytes(data.hf_cache.bytes))),
      h("p", { class: "sub", style: { margin: 0 } }, "Hugging Face keeps the raw FineWeb-Edu download in ", h("code", { text: data.hf_cache.path }), ". Once the shards exist, training doesn't need it and you can delete it."));
    jobs.update(prepJobs);
  }
  every(4000, refresh);
}

// ---- evaluate ---------------------------------------------------------------------------
const HF_LABELS = { gpt2: "GPT-2 small (124M)", "gpt2-medium": "GPT-2 medium (355M)", "gpt2-large": "GPT-2 large (774M)", "gpt2-xl": "GPT-2 XL (1.5B)" };
const TASKS = [
  ["arc_easy", "ARC easy"],
  ["arc_challenge", "ARC challenge"],
  ["piqa", "PIQA"],
  ["winogrande", "WinoGrande"],
  ["lambada_openai", "LAMBADA"],
];

async function evaluatePage(params) {
  const token = routeToken;
  main.append(pageHead("Evaluate", "Score a checkpoint and OpenAI's GPT-2 with the same code, then compare them below. Evaluations use the GPU, so they wait while a run trains."));
  const cks = await api("/api/checkpoints");
  if (!stillOn(token)) return;

  const model = h("select", { class: "input", id: "ev-model" },
    cks.length ? h("optgroup", { label: "Your runs (latest checkpoint)" }, cks.map((r) =>
      h("option", { value: `run:${r.run}`, selected: r.run === params.get("run"), text: `${r.run} · step ${fmt.int(r.checkpoints.at(-1).step)}` }))) : null,
    h("optgroup", { label: "OpenAI GPT-2 (downloaded on first use)" }, Object.entries(HF_LABELS).map(([k, label]) => h("option", { value: `hf:${k}`, text: label }))));
  const valLoss = h("input", { type: "checkbox", checked: true });
  const hellaswag = h("input", { type: "checkbox", checked: true });
  const hsLimit = h("input", { class: "input", placeholder: "all 10,042", inputmode: "numeric", id: "ev-limit" });
  const batch = h("input", { class: "input", value: "16", inputmode: "numeric", id: "ev-batch" });
  const taskBoxes = TASKS.map(([key, label]) => [key, h("input", { type: "checkbox" }), label]);
  const startBtn = h("button", { class: "btn primary", type: "button", text: "Start evaluation" });
  const form = h("section", { class: "panel" },
    h("div", { class: "form-row" },
      h("div", { class: "field wide" }, h("label", { for: "ev-model", text: "Model" }), model),
      h("div", { class: "field" }, h("label", { for: "ev-batch", text: "Batch size" }), batch)),
    h("div", { class: "form-row", style: { paddingTop: 0 } },
      h("fieldset", {}, h("legend", { text: "Benchmarks" }),
        h("div", { class: "checks" },
          h("label", { class: "check", title: "A run is scored on its own validation shards; GPT-2 on FineWeb-Edu's" }, valLoss, "Validation loss"),
          h("label", { class: "check" }, hellaswag, "HellaSwag"))),
      h("div", { class: "field" }, h("label", { for: "ev-limit", text: "HellaSwag examples" }), hsLimit)),
    h("div", { class: "form-row", style: { paddingTop: 0 } },
      h("fieldset", {}, h("legend", { text: "Extra benchmarks (lm-evaluation-harness, a few minutes each)" }),
        h("div", { class: "checks" }, taskBoxes.map(([, box, label]) => h("label", { class: "check" }, box, label))))),
    h("div", { class: "form-row", style: { paddingTop: 0 } }, startBtn));

  const compareBox = h("div");
  const jobs = jobList("No evaluations have run yet.");
  main.append(form, h("h2", { text: "Comparison" }), compareBox, h("h2", { text: "Evaluation jobs" }), jobs.el);

  startBtn.addEventListener("click", async () => {
    const [kind, name] = model.value.split(/:(.*)/s);
    const body = {
      val_loss: valLoss.checked,
      hellaswag: hellaswag.checked,
      lm_eval: taskBoxes.filter(([, box]) => box.checked).map(([key]) => key),
      batch_size: Number(batch.value) || 16,
    };
    body[kind === "run" ? "run" : "hf"] = name;
    if (hsLimit.value.trim()) body.hellaswag_limit = Number(hsLimit.value.replaceAll(",", ""));
    startBtn.disabled = true;
    try {
      await api("/api/evaluate", { method: "POST", body });
      toast("Evaluation started. Follow it under Evaluation jobs.");
    } catch (e) {
      toast(e.message, "error");
    }
    startBtn.disabled = false;
    refresh();
  });

  let compareKey = null;
  async function refresh() {
    const [cmp, evalJobs] = await Promise.all([api("/api/compare"), api("/api/jobs?kind=evaluate&limit=20")]);
    jobs.update(evalJobs);
    const key = JSON.stringify(cmp);
    if (key === compareKey) return;
    compareKey = key;
    compareBox.replaceChildren(compareTable(cmp));
  }
  every(5000, refresh);
}

// Horizontal bars per model for validation loss and HellaSwag, best first.
function compareCharts(rows) {
  const label = (r) => (r.reference ? `OpenAI ${r.name.replace(/^hf:/, "")}` : r.run);
  const color = (r) => (r.reference ? "var(--series-3)" : "var(--series-1)");
  const metric = (title, key, f, better) => {
    const items = rows.filter((r) => r[key] != null);
    if (!items.length) return null;
    const max = Math.max(...items.map((r) => r[key]));
    const sorted = [...items].sort((a, b) => (better === "Lower" ? a[key] - b[key] : b[key] - a[key]));
    return h("section", { class: "panel chart-card" },
      h("div", { class: "chart-head" }, h("span", { class: "chart-title", text: title }), h("span", { class: "chart-now sub", text: `${better} is better` })),
      h("div", { class: "hbars", role: "list" }, sorted.map((r) =>
        h("div", { class: "hbar-row", role: "listitem", title: `${label(r)}: ${f(r[key])}` },
          h("span", { class: "hbar-label", text: label(r) }),
          h("div", { class: "hbar-track" }, h("span", { class: "hbar-bar", style: { width: `${(100 * r[key]) / max}%`, background: color(r) } })),
          h("span", { class: "hbar-value", text: f(r[key]) })))));
  };
  const charts = [metric("Validation loss", "val_loss", fmt.loss, "Lower"), metric("HellaSwag accuracy", "hellaswag", fmt.pct, "Higher")].filter(Boolean);
  if (!charts.length) return null;
  const kinds = [...new Set(rows.map((r) => r.reference))].sort();
  return h("div", { class: "compare-charts" },
    kinds.length > 1
      ? h("div", { class: "chart-legend" }, kinds.map((ref) => h("span", {}, h("span", { class: "pbar-swatch", style: { background: ref ? "var(--series-3)" : "var(--series-1)" } }), ref ? " OpenAI GPT-2" : " Your runs")))
      : null,
    h("div", { class: "charts" }, charts));
}

function compareTable({ rows, tasks }) {
  if (!rows.length) {
    return h("div", { class: "panel empty" }, h("h3", { text: "No evaluations yet" }),
      h("p", { text: "Evaluate one of your checkpoints and OpenAI's GPT-2 small to see them side by side." }));
  }
  const sorted = [...rows].sort((a, b) => (a.reference - b.reference) || (a.val_loss ?? 99) - (b.val_loss ?? 99));
  const cols = [
    ["val_loss", "Val loss", fmt.loss, "min"],
    ["hellaswag", "HellaSwag", fmt.pct, "max"],
    ...tasks.map((t) => [`task:${t}`, (TASKS.find(([k]) => k === t) || [t, t])[1], fmt.pct, "max"]),
  ];
  const get = (r, key) => (key.startsWith("task:") ? r.lm_eval[key.slice(5)] : r[key]);
  const best = Object.fromEntries(cols.map(([key, , , dir]) => {
    const vals = rows.map((r) => get(r, key)).filter((v) => v != null);
    return [key, vals.length > 1 ? (dir === "min" ? Math.min(...vals) : Math.max(...vals)) : null];
  }));
  return h("div", {},
    compareCharts(rows),
    h("div", { class: "panel table-wrap" }, h("table", { class: "grid" },
      h("thead", {}, h("tr", {}, th("Model"), th("Checkpoint", "num"), th("Parameters", "num"), cols.map(([, label]) => th(label, "num")))),
      h("tbody", {}, sorted.map((r) => h("tr", {},
        h("td", {}, r.reference ? h("span", { text: `OpenAI ${r.name.replace(/^hf:/, "")}` }) : h("a", { class: "run-name", href: `#/runs/${enc(r.run)}`, text: r.run })),
        tdNum(r.step != null ? `step ${fmt.int(r.step)}` : "—"),
        tdNum(fmt.compact(r.params)),
        cols.map(([key, , f]) => {
          const val = get(r, key);
          return h("td", { class: `num ${val != null && val === best[key] ? "best" : ""}`, text: f(val) });
        })))))),
    h("p", { class: "sub", text: "Bold marks the best score in each column. Published reference for this scoring: GPT-2 small reaches about 3.29 validation loss and 29.5% HellaSwag." }));
}

// ---- generate ---------------------------------------------------------------------------
async function generatePage(params) {
  const token = routeToken;
  main.append(pageHead("Generate text", "Sample from any checkpoint. It uses the GPU when it's free, and the CPU while a run is training."));
  const cks = await api("/api/checkpoints");
  if (!stillOn(token)) return;
  if (!cks.length) {
    main.append(h("div", { class: "panel empty" }, h("h3", { text: "No checkpoints yet" }),
      h("p", { text: "Runs save a checkpoint every ckpt_every steps, and whenever they're stopped." }),
      h("div", { class: "actions" }, h("a", { class: "btn primary", href: "#/new", text: "New run" }))));
    return;
  }
  const runSel = h("select", { class: "input", id: "g-run" }, cks.map((r) => h("option", { value: r.run, selected: r.run === params.get("run"), text: r.run })));
  const ckSel = h("select", { class: "input", id: "g-ck" });
  const fillCks = () => {
    const r = cks.find((x) => x.run === runSel.value) || cks[0];
    ckSel.replaceChildren(...[...r.checkpoints].reverse().map((c) => h("option", { value: c.file, text: `step ${fmt.int(c.step)}` })));
  };
  runSel.addEventListener("change", fillCks);
  fillCks();
  const prompt = h("textarea", { class: "input", id: "g-prompt", rows: 4 });
  prompt.value = "The theory of relativity";
  const num = (id, value, label) => {
    const input = h("input", { class: "input", id, value: String(value), inputmode: "decimal" });
    return [input, h("div", { class: "field" }, h("label", { for: id, text: label }), input)];
  };
  const [len, lenField] = num("g-len", 96, "Length (tokens)");
  const [temp, tempField] = num("g-temp", 0.8, "Temperature");
  const [topk, topkField] = num("g-topk", 50, "Top-k");
  const [count, countField] = num("g-count", 3, "Samples");
  const [seed, seedField] = num("g-seed", 0, "Seed");
  const go = h("button", { class: "btn primary", type: "button", text: "Generate" });
  const out = h("div");
  main.append(
    h("section", { class: "panel" },
      h("div", { class: "form-row" },
        h("div", { class: "field wide" }, h("label", { for: "g-run", text: "Run" }), runSel),
        h("div", { class: "field" }, h("label", { for: "g-ck", text: "Checkpoint" }), ckSel)),
      h("div", { class: "form-row", style: { paddingTop: 0 } }, h("div", { class: "field wide" }, h("label", { for: "g-prompt", text: "Prompt" }), prompt)),
      h("div", { class: "form-row", style: { paddingTop: 0 } }, lenField, tempField, topkField, countField, seedField, go)),
    out);

  async function generate() {
    const body = {
      run: runSel.value,
      checkpoint: ckSel.value,
      prompt: prompt.value,
      max_new_tokens: Number(len.value),
      temperature: Number(temp.value),
      top_k: topk.value.trim() ? Number(topk.value) : null,
      num_samples: Number(count.value),
      seed: Number(seed.value) || 0,
    };
    go.disabled = true;
    go.textContent = "Generating…";
    try {
      const r = await api("/api/generate", { method: "POST", body });
      if (!stillOn(token)) return;
      const where = r.device === "cuda" ? "on the GPU" : "on the CPU";
      out.replaceChildren(
        h("h2", { text: `${r.samples.length} sample${r.samples.length === 1 ? "" : "s"} · ${r.run} · ${r.checkpoint}` }),
        h("div", { class: "panel" }, r.samples.map((text, i) => h("div", { class: "sample" },
          h("span", { class: "meta", text: `SAMPLE ${i + 1} · SEED ${body.seed + i}` }),
          h("p", {}, text.startsWith(r.prompt) ? [h("span", { class: "prompt", text: r.prompt }), text.slice(r.prompt.length)] : text)))),
        h("p", { class: "sub", text: `Generated ${where} in ${r.seconds.toFixed(1)} s.` }));
    } catch (e) {
      toast(e.message, "error");
    } finally {
      go.disabled = false;
      go.textContent = "Generate";
    }
  }
  go.addEventListener("click", generate);
  prompt.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) generate();
  });
}

// ---- top bar: GPU readout, theme, shutdown ----------------------------------------------
const gpuEl = document.getElementById("gpu");
const offline = document.getElementById("offline");
let serverStopped = false;

async function refreshSystem() {
  if (serverStopped) return;
  try {
    const sys = await api("/api/system");
    offline.hidden = true;
    const g = sys.gpu;
    if (!g) {
      gpuEl.textContent = "No GPU found";
      return;
    }
    const memPct = g.mem_total_mb ? (g.mem_used_mb / g.mem_total_mb) * 100 : 0;
    gpuEl.title = sys.gpu_user ? `In use: ${sys.gpu_user}` : "The GPU is free";
    gpuEl.replaceChildren(
      h("span", { text: g.name.replace(/^NVIDIA (GeForce )?/, "") }),
      h("span", { class: "detail", text: `${g.util ?? "—"}%` }),
      h("span", { class: "meter", role: "img", "aria-label": `GPU memory ${memPct.toFixed(0)}% used` }, h("span", { style: { width: `${memPct}%` } })),
      h("span", { class: "detail", text: `${(g.mem_used_mb / 1024).toFixed(1)}/${(g.mem_total_mb / 1024).toFixed(1)} GB` }),
      h("span", { class: "detail", text: g.temp_c != null ? `${g.temp_c}°C` : "" }));
  } catch {
    offline.hidden = false;
  }
}
refreshSystem();
setInterval(() => !document.hidden && refreshSystem(), 5000);

function applyTheme(theme) {
  if (theme === "light" || theme === "dark") document.documentElement.dataset.theme = theme;
  else delete document.documentElement.dataset.theme;
  for (const b of document.querySelectorAll("#theme button")) b.setAttribute("aria-pressed", String(b.dataset.theme === theme));
  try {
    if (theme === "system") localStorage.removeItem("gptlab-theme");
    else localStorage.setItem("gptlab-theme", theme);
  } catch {
    // Storage can be unavailable; the theme still applies for this visit.
  }
}
for (const b of document.querySelectorAll("#theme button")) b.addEventListener("click", () => applyTheme(b.dataset.theme));
applyTheme(document.documentElement.dataset.theme || "system");

const dialog = document.getElementById("shutdown-dialog");
const shutdownText = document.getElementById("shutdown-text");
const forceBtn = document.getElementById("shutdown-force");
const confirmBtn = document.getElementById("shutdown-confirm");
document.getElementById("shutdown").addEventListener("click", () => {
  document.getElementById("menu").open = false;
  shutdownText.textContent = "This page stops updating. Unless an Ubuntu terminal is open, WSL shuts down a few seconds later.";
  forceBtn.hidden = true;
  confirmBtn.hidden = false;
  dialog.showModal();
});
document.getElementById("shutdown-cancel").addEventListener("click", () => dialog.close());
async function shutdown(force) {
  try {
    await api("/api/shutdown", { method: "POST", body: { force } });
  } catch (e) {
    shutdownText.textContent = e.message;
    forceBtn.hidden = false;
    confirmBtn.hidden = true;
    return;
  }
  dialog.close();
  serverStopped = true;
  cleanups.forEach((fn) => fn());
  cleanups = [];
  main.replaceChildren(h("div", { class: "panel empty" }, h("h3", { text: "The server has stopped" }),
    h("p", { text: "Start it again from Windows with Start-GPTLabUI.ps1, or with python -m gpt_lab.ui in Ubuntu." })));
  gpuEl.replaceChildren();
}
confirmBtn.addEventListener("click", () => shutdown(false));
forceBtn.addEventListener("click", () => shutdown(true));

window.addEventListener("hashchange", route);
route();
