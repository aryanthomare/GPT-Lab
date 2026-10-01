// A 3D drawing of the transformer that a model config describes, plus a parameter
// breakdown bar. The drawing uses a plain 2D canvas: boxes are projected orthographically
// and painted back to front, with ink edges like an engineering drawing. Two boxes that
// overlap on screen are ordered along an axis that separates them, which is exact for
// disjoint axis-aligned boxes. Drag to turn it, double-click to reset.

const nf = new Intl.NumberFormat("en-US");
const int = (v) => nf.format(Math.round(v));
function compact(v) {
  const a = Math.abs(v);
  if (a >= 1e9) return `${(v / 1e9).toFixed(2)}B`;
  if (a >= 1e6) return `${(v / 1e6).toFixed(1)}M`;
  if (a >= 1e3) return `${(v / 1e3).toFixed(1)}k`;
  return String(Math.round(v));
}
const pct = (v, total) => `${((100 * v) / total).toFixed(1)}%`;

function mk(tag, cls, text) {
  const el = document.createElement(tag);
  if (cls) el.className = cls;
  if (text != null) el.textContent = text;
  return el;
}

// ---- the model's numbers (mirrors gpt_lab/model.py) -------------------------------------
export function modelSpec(c) {
  if (!c) return null;
  const d = Number(c.d_model);
  const L = Number(c.n_layer);
  const H = Number(c.n_head);
  const KV = Number(c.n_kv_head || c.n_head);
  const vocab = Number(c.vocab_size);
  const seq = Number(c.seq_len);
  if (![d, L, H, KV, vocab, seq].every((v) => Number.isInteger(v) && v > 0)) return null;
  if (d % H || H % KV || L > 512 || d > 65536) return null;
  const hd = d / H;
  let ffn = c.ffn_hidden;
  if (ffn == null) {
    const m = Number(c.ffn_multiple_of) || 1;
    ffn = m * Math.ceil(Math.floor((8 * d) / 3) / m);
  }
  ffn = Number(ffn);
  if (!(Number.isInteger(ffn) && ffn > 0)) return null;
  const attn = d * d + d * 2 * KV * hd + d * d + (c.qk_norm ? 2 * hd : 0); // Wq, Wkv, Wo, QK-norm
  const mlp = 3 * d * ffn; // gate and up (d -> ffn each), down (ffn -> d)
  const embedding = vocab * d;
  const head = c.tie_embeddings ? 0 : embedding;
  const norms = L * 2 * d + d;
  const params = { attention: L * attn, mlp: L * mlp, embedding, head, norms };
  const total = Object.values(params).reduce((a, b) => a + b, 0);
  return { d, L, H, KV, hd, ffn, vocab, seq, tied: !!c.tie_embeddings, qkNorm: !!c.qk_norm, perLayer: { attn, mlp }, params, total };
}

// ---- geometry ---------------------------------------------------------------------------
// x: feature width, y: up through the network, z: depth (rows; grows with context length).
// Widths grow with the square root of the dimension, depth with the log of the context
// length; a 12-layer model comes out about 1.5 times taller than wide.
const widthOf = (n) => 1.1 * Math.sqrt(n);

function build(s) {
  const W = widthOf(s.d);
  const F = Math.max(widthOf(s.ffn), W + 2);
  const D = Math.min(44, Math.max(10, 2 * (7 + 2.2 * Math.log2(Math.max(s.seq, 16) / 64))));
  const k = Math.min(1, 110 / (s.L * 6.7)); // squeeze deep stacks so they stay in proportion
  const normH = Math.max(0.12, 0.28 * k);
  const subH = Math.max(0.3, 2.0 * k);
  const gap = 0.4 * k;
  const layerGap = Math.max(0.2, 1.0 * k);
  const E = Math.min(64, D * 1.6);
  const embH = 2.2;
  const resX1 = -F / 2 - 3.5;
  const resX0 = resX1 - 1.4;
  const boxes = [];
  const add = (id, kind, x0, x1, y0, y1, z0, z1, extra = {}) =>
    boxes.push({ id, kind, min: [x0, y0, z0], max: [x1, y1, z1], ...extra });
  const stubH = Math.min(0.25, subH / 3);
  const stub = (id, x1, yMid, layer) => add(id, "stub", resX1, x1, yMid - stubH, yMid + stubH, -0.55, 0.55, { layer });

  add("embed", "embed", -W / 2, W / 2, 0, embH, -E / 2, E / 2);
  stub("r0", -W / 2, embH / 2);
  let y = embH + 2.5;
  const yStart = y;
  const qz1 = -D / 2 + D * 0.56 - 0.4;
  for (let l = 0; l < s.L; l++) {
    add(`L${l}.n1`, "norm", -W / 2, W / 2, y, y + normH, -D / 2, D / 2, { layer: l, part: "attention" });
    y += normH + gap;
    add(`L${l}.q`, "attn", -W / 2, W / 2, y, y + subH, -D / 2, qz1, { layer: l, div: s.H, role: "q" });
    add(`L${l}.kv`, "attn", -W / 2, W / 2, y, y + subH, qz1 + 0.8, D / 2, { layer: l, div: s.KV, role: "kv" });
    stub(`L${l}.r1`, -W / 2, y + subH / 2, l);
    y += subH + gap;
    add(`L${l}.n2`, "norm", -W / 2, W / 2, y, y + normH, -D / 2, D / 2, { layer: l, part: "MLP" });
    y += normH + gap;
    add(`L${l}.gate`, "mlp", -F / 2, F / 2, y, y + subH, -D / 2, -0.4, { layer: l, role: "gate" });
    add(`L${l}.up`, "mlp", -F / 2, F / 2, y, y + subH, 0.4, D / 2, { layer: l, role: "up" });
    stub(`L${l}.r2`, -F / 2, y + subH / 2, l);
    y += subH + (l === s.L - 1 ? gap : layerGap);
  }
  const yEnd = y;
  add("final", "norm", -W / 2, W / 2, y, y + normH, -D / 2, D / 2, { part: "final" });
  stub("rF", -W / 2, y + normH / 2);
  const yRes = y + normH / 2 + stubH;
  y += normH + 2.5;
  add("head", "head", -W / 2, W / 2, y, y + embH, -E / 2, E / 2, { wire: s.tied });
  add("res", "residual", resX0, resX1, embH / 2 - stubH, yRes, -0.7, 0.7);

  const mid = Math.floor((s.L - 1) / 2);
  const dims = [
    { // feature width, along the front of the embedding
      a: [-W / 2, 0, E / 2 + 5], b: [W / 2, 0, E / 2 + 5],
      ea: [-W / 2, 0, E / 2], eb: [W / 2, 0, E / 2],
      text: `d_model ${int(s.d)}`,
    },
    { // layer count, up the left of the residual stream
      a: [resX0 - 4, yStart, D / 2], b: [resX0 - 4, yEnd, D / 2],
      ea: [resX0, yStart, D / 2], eb: [resX0, yEnd, D / 2],
      text: `${int(s.L)} layer${s.L === 1 ? "" : "s"}`,
    },
    { // context length, across the top block
      a: [F / 2 + 4, yEnd, -D / 2], b: [F / 2 + 4, yEnd, D / 2],
      ea: [F / 2, yEnd, -D / 2], eb: [F / 2, yEnd, D / 2],
      text: `context ${int(s.seq)}`,
    },
  ];
  const callouts = [
    { id: "head", title: "Output head", detail: s.tied ? "same weights as the embedding" : `${int(s.d)} → ${int(s.vocab)} · ${compact(s.params.head)}` },
    { id: `L${mid}.gate`, title: `SwiGLU MLP × ${s.L}`, detail: `${int(s.d)} → ${int(s.ffn)} → ${int(s.d)} · ${compact(s.perLayer.mlp)} each` },
    { id: `L${mid}.q`, title: `Attention × ${s.L}`, detail: `${s.H} heads × ${s.hd}${s.KV !== s.H ? ` · ${s.KV} K/V` : ""} · ${compact(s.perLayer.attn)} each` },
    { id: "res", title: "Residual stream", detail: `${int(s.d)} wide · no weights` },
    { id: "embed", title: "Token embedding", detail: `${int(s.vocab)} × ${int(s.d)} · ${compact(s.params.embedding)}` },
  ];
  return { boxes, dims, callouts };
}

// What hovering a part says about it.
function describe(b, s) {
  const layer = b.layer != null ? `Layer ${b.layer + 1} · ` : "";
  switch (b.kind) {
    case "embed":
      return ["Token embedding", `${int(s.vocab)} tokens × ${int(s.d)} features`, `${int(s.params.embedding)} parameters`];
    case "head":
      return s.tied
        ? ["Output head", "Reuses the token embedding's weights (tied)", "no extra parameters"]
        : ["Output head", `${int(s.d)} → ${int(s.vocab)} logits`, `${int(s.params.head)} parameters`];
    case "norm":
      return b.part === "final"
        ? ["Final RMSNorm", `${int(s.d)} scale weights`, "normalizes before the output head"]
        : [`${layer}RMSNorm`, `before the ${b.part}`, `${int(s.d)} scale weights`];
    case "attn":
      return b.role === "q"
        ? [`${layer}attention · query heads`, `${s.H} heads of ${s.hd} dims${s.qkNorm ? ", QK-norm" : ""}`, `${int(s.perLayer.attn)} parameters (Q, K, V, output)`]
        : [`${layer}attention · key/value heads`,
           s.KV === s.H ? `${s.KV} K/V heads, one per query head` : `${s.KV} K/V heads, each shared by ${s.H / s.KV} query heads`,
           `${int(s.perLayer.attn)} parameters (Q, K, V, output)`];
    case "mlp":
      return [`${layer}SwiGLU MLP · ${b.role} projection`, `${int(s.d)} → ${int(s.ffn)}, then back to ${int(s.d)}`, `${int(s.perLayer.mlp)} parameters (gate, up, down)`];
    default:
      return ["Residual stream", `${int(s.d)} features carried through every layer`, "each block adds its output to it · no weights"];
  }
}

// ---- projection and paint order ---------------------------------------------------------
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

function camera(yaw, pitch) {
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  return { dir: [sy * cp, sp, cy * cp], right: [cy, 0, -sy], up: [-sy * sp, cp, -cy * sp] };
}

function corners(b) {
  const [x0, y0, z0] = b.min, [x1, y1, z1] = b.max;
  return [[x0, y0, z0], [x1, y0, z0], [x0, y1, z0], [x1, y1, z0], [x0, y0, z1], [x1, y0, z1], [x0, y1, z1], [x1, y1, z1]];
}

function faces(b) {
  const [x0, y0, z0] = b.min, [x1, y1, z1] = b.max;
  return [
    ["px", [1, 0, 0], [[x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1]]],
    ["nx", [-1, 0, 0], [[x0, y0, z1], [x0, y1, z1], [x0, y1, z0], [x0, y0, z0]]],
    ["py", [0, 1, 0], [[x0, y1, z0], [x0, y1, z1], [x1, y1, z1], [x1, y1, z0]]],
    ["ny", [0, -1, 0], [[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]]],
    ["pz", [0, 0, 1], [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]]],
    ["nz", [0, 0, -1], [[x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0]]],
  ];
}

// -1: draw a first (it's behind b); 1: draw b first; 0: no constraint.
function behind(a, b, d) {
  for (let k = 0; k < 3; k++) {
    if (a.box.max[k] <= b.box.min[k] + 1e-9) return d[k] > 0 ? -1 : d[k] < 0 ? 1 : 0;
    if (b.box.max[k] <= a.box.min[k] + 1e-9) return d[k] > 0 ? 1 : d[k] < 0 ? -1 : 0;
  }
  return 0;
}

function paintOrder(items, d) {
  const n = items.length;
  const next = Array.from({ length: n }, () => []);
  const indeg = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    const a = items[i];
    for (let j = i + 1; j < n; j++) {
      const b = items[j];
      if (a.rx1 < b.rx0 || b.rx1 < a.rx0 || a.ry1 < b.ry0 || b.ry1 < a.ry0) continue;
      const c = behind(a, b, d);
      if (c < 0) { next[i].push(j); indeg[j]++; }
      else if (c > 0) { next[j].push(i); indeg[i]++; }
    }
  }
  const byDepth = [...items.keys()].sort((p, q) => items[p].depth - items[q].depth);
  const out = [];
  const done = new Array(n).fill(false);
  const ready = byDepth.filter((i) => indeg[i] === 0);
  while (ready.length) {
    const i = ready.shift();
    done[i] = true;
    out.push(items[i]);
    for (const j of next[i]) if (--indeg[j] === 0) ready.push(j);
  }
  for (const i of byDepth) if (!done[i]) out.push(items[i]); // a cycle: fall back to depth
  return out;
}

// ---- color ------------------------------------------------------------------------------
function rgb(hex) {
  const h = hex.trim().replace("#", "");
  const v = h.length === 3 ? h.split("").map((c) => c + c).join("") : h;
  return [0, 2, 4].map((i) => parseInt(v.slice(i, i + 2), 16) || 0);
}
const mix = (a, b, t) => a.map((v, i) => Math.round(v + (b[i] - v) * t));
const css = (c, alpha = 1) => `rgba(${c[0]}, ${c[1]}, ${c[2]}, ${alpha})`;
const SHADE = { // lit from the upper front left; positive mixes toward white
  light: { py: 0.45, pz: 0.22, nx: 0.1, px: -0.1, nz: -0.18, ny: -0.3 },
  dark: { py: 0.22, pz: 0.08, nx: 0.0, px: -0.18, nz: -0.28, ny: -0.42 },
};

function palette(el) {
  const st = getComputedStyle(el);
  const v = (name) => rgb(st.getPropertyValue(name) || "#888888");
  const sheet = v("--sheet");
  const dark = (sheet[0] + sheet[1] + sheet[2]) / 3 < 128;
  return {
    dark,
    sheet,
    ink: v("--ink"),
    ink2: v("--ink-2"),
    muted: v("--muted"),
    frame: v("--frame"),
    focus: v("--focus"),
    kinds: {
      attn: v("--series-1"),
      mlp: v("--series-2"),
      embed: v("--series-3"),
      head: v("--series-3"),
      norm: v("--m-norm"),
      residual: v("--ink-2"),
      stub: v("--ink-2"),
    },
  };
}

function shade(base, face, pal) {
  const t = SHADE[pal.dark ? "dark" : "light"][face];
  return t >= 0 ? mix(base, [255, 255, 255], t) : mix(base, [0, 0, 0], -t);
}

const FONT_UI = '"Atkinson Hyperlegible Next", "Atkinson Hyperlegible", system-ui, sans-serif';
const FONT_MONO = '"B612 Mono", ui-monospace, Consolas, monospace';
const DEFAULT_VIEW = { yaw: -0.62, pitch: 0.42, zoom: 1 };
const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2);
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

function lerpBox(a, b, t) {
  return { ...b, min: a.min.map((v, i) => v + (b.min[i] - v) * t), max: a.max.map((v, i) => v + (b.max[i] - v) * t) };
}
function collapse(b) {
  const y = b.min[1];
  return { ...b, min: [b.min[0], y, b.min[2]], max: [b.max[0], y, b.max[2]] };
}

function pointInPoly(x, y, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i];
    const [xj, yj] = poly[j];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

// ---- the 3D view ------------------------------------------------------------------------
export function modelView() {
  const canvas = mk("canvas", "model-canvas");
  canvas.tabIndex = 0;
  canvas.setAttribute("role", "img");
  canvas.title = "Drag or use the arrow keys to turn the model. Double-click or press 0 to reset.";
  const tip = mk("div", "tooltip model-tip");
  tip.hidden = true;
  const reset = mk("button", "btn small quiet", "Reset view");
  const zoomOut = mk("button", "btn small quiet", "−");
  const zoomIn = mk("button", "btn small quiet", "+");
  for (const b of [reset, zoomOut, zoomIn]) b.type = "button";
  zoomOut.setAttribute("aria-label", "Zoom out");
  zoomIn.setAttribute("aria-label", "Zoom in");
  const controls = mk("div", "model-controls");
  controls.append(mk("span", "model-hint", "Drag to turn · hover for details"), zoomOut, zoomIn, reset);
  const el = mk("div", "model-view");
  el.append(canvas, tip, controls);

  const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
  let spec = null;
  let geo = null;
  let specKey = "";
  let shown = new Map();
  let from = null;
  let to = null;
  let t0 = 0;
  let morphing = false;
  let view = { ...DEFAULT_VIEW };
  let viewAnim = null;
  let fit = null;
  let drag = null;
  let hover = null;
  let hits = [];
  let raf = 0;

  function schedule() {
    if (!raf) raf = requestAnimationFrame(frame);
  }

  function update(modelCfg) {
    const s = modelSpec(modelCfg);
    const key = JSON.stringify(s);
    if (key === specKey) return;
    specKey = key;
    spec = s;
    if (!s) {
      geo = null;
      shown = new Map();
      morphing = false;
      canvas.setAttribute("aria-label", "No drawing: the model shape isn't valid.");
      schedule();
      return;
    }
    geo = build(s);
    const target = new Map(geo.boxes.map((b) => [b.id, b]));
    if (reduceMotion || !shown.size) {
      shown = target;
      morphing = false;
      fit = null;
    } else {
      from = new Map();
      to = new Map();
      for (const [id, b] of target) {
        from.set(id, shown.get(id) || collapse(b));
        to.set(id, b);
      }
      for (const [id, b] of shown) {
        if (!target.has(id)) {
          from.set(id, b);
          to.set(id, collapse(b));
        }
      }
      t0 = performance.now();
      morphing = true;
    }
    canvas.setAttribute(
      "aria-label",
      `3D drawing of the model: ${s.L} layers, ${s.d} wide, ${s.H} attention heads` +
        `${s.KV !== s.H ? ` sharing ${s.KV} key/value heads` : ""}, MLP width ${s.ffn}, ` +
        `context ${s.seq} tokens, ${compact(s.total)} parameters.`,
    );
    schedule();
  }

  function frame(now) {
    raf = 0;
    if (morphing) {
      const p = Math.min(1, (now - t0) / 560);
      const e = ease(p);
      shown = new Map();
      for (const [id, a] of from) shown.set(id, lerpBox(a, to.get(id), e));
      if (p >= 1) {
        morphing = false;
        shown = new Map(geo.boxes.map((b) => [b.id, b]));
      }
    }
    if (viewAnim) {
      const p = Math.min(1, (now - viewAnim.t0) / 450);
      const e = ease(p);
      for (const k of ["yaw", "pitch", "zoom"]) view[k] = viewAnim.from[k] + (viewAnim.to[k] - viewAnim.from[k]) * e;
      if (p >= 1) viewAnim = null;
    }
    const settled = draw();
    if (morphing || viewAnim || !settled) schedule();
  }

  function draw() {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (!w || !h) return true;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      fit = null;
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const pal = palette(canvas);
    hits = [];
    if (!shown.size || !spec) {
      ctx.fillStyle = css(pal.muted);
      ctx.font = `13px ${FONT_UI}`;
      ctx.textAlign = "center";
      ctx.fillText("Fix the model shape to see it drawn.", w / 2, h / 2);
      return true;
    }

    const cam = camera(view.yaw, view.pitch);
    const showCallouts = w >= 600;
    const area = { x: 20, y: 24, w: w - 40 - (showCallouts ? 230 : 0), h: h - 64 };

    // Fit the drawing (and its dimension lines) to the area, easing toward the new fit.
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    const extend = (p) => {
      const sx = dot(p, cam.right);
      const sy = -dot(p, cam.up);
      x0 = Math.min(x0, sx); x1 = Math.max(x1, sx); y0 = Math.min(y0, sy); y1 = Math.max(y1, sy);
    };
    for (const b of shown.values()) corners(b).forEach(extend);
    for (const dm of geo.dims) { extend(dm.a); extend(dm.b); }
    const scale = view.zoom * 0.92 * Math.min(area.w / Math.max(1e-6, x1 - x0), area.h / Math.max(1e-6, y1 - y0));
    const target = { s: scale, cx: area.x + area.w / 2 - ((x0 + x1) / 2) * scale, cy: area.y + area.h / 2 - ((y0 + y1) / 2) * scale };
    let settled = true;
    if (!fit) fit = target;
    else {
      const a = 0.22;
      const next = { s: fit.s + (target.s - fit.s) * a, cx: fit.cx + (target.cx - fit.cx) * a, cy: fit.cy + (target.cy - fit.cy) * a };
      settled = Math.abs(next.s - target.s) / target.s < 0.002 && Math.abs(next.cx - target.cx) < 0.5 && Math.abs(next.cy - target.cy) < 0.5;
      fit = settled ? target : next;
    }
    const P = (p) => [fit.cx + dot(p, cam.right) * fit.s, fit.cy - dot(p, cam.up) * fit.s];

    // Project every box, then paint back to front.
    const items = [];
    for (const box of shown.values()) {
      const pts = corners(box).map(P);
      const xs = pts.map((p) => p[0]);
      const ys = pts.map((p) => p[1]);
      const c = box.min.map((v, i) => (v + box.max[i]) / 2);
      items.push({
        box,
        rx0: Math.min(...xs), rx1: Math.max(...xs), ry0: Math.min(...ys), ry1: Math.max(...ys),
        depth: dot(c, cam.dir),
        center: P(c),
      });
    }
    const ordered = paintOrder(items, cam.dir);
    ctx.lineJoin = "round";
    for (const it of ordered) {
      const b = it.box;
      const base = pal.kinds[b.kind] || pal.ink2;
      const isHover = hover === b.id;
      const visible = faces(b).filter(([, n]) => dot(n, cam.dir) > 1e-6);
      for (const [name, , quad] of visible) {
        const poly = quad.map(P);
        ctx.beginPath();
        poly.forEach(([px, py], i) => (i ? ctx.lineTo(px, py) : ctx.moveTo(px, py)));
        ctx.closePath();
        if (!b.wire) {
          ctx.fillStyle = css(isHover ? mix(shade(base, name, pal), [255, 255, 255], 0.18) : shade(base, name, pal));
          ctx.fill();
        }
        ctx.setLineDash(b.wire ? [5, 4] : []);
        ctx.strokeStyle = css(b.wire ? pal.ink2 : pal.frame, b.wire ? 0.9 : 0.75);
        ctx.lineWidth = 1;
        ctx.stroke();
        hits.push({ id: b.id, poly });
        // Head divisions: one line per head across the faces that run along x.
        if (b.div > 1 && (name === "py" || name === "pz" || name === "nz")) {
          const [bx0, by0, bz0] = b.min;
          const [bx1, by1, bz1] = b.max;
          const step = (bx1 - bx0) / b.div;
          if (step * fit.s >= 2.5) {
            ctx.beginPath();
            for (let i = 1; i < b.div; i++) {
              const x = bx0 + i * step;
              const seg = name === "py" ? [[x, by1, bz0], [x, by1, bz1]] : name === "pz" ? [[x, by0, bz1], [x, by1, bz1]] : [[x, by0, bz0], [x, by1, bz0]];
              const [p, q] = seg.map(P);
              ctx.moveTo(p[0], p[1]);
              ctx.lineTo(q[0], q[1]);
            }
            ctx.strokeStyle = css(pal.frame, 0.35);
            ctx.stroke();
          }
        }
      }
      ctx.setLineDash([]);
      if (isHover) {
        for (const [, , quad] of visible) {
          const poly = quad.map(P);
          ctx.beginPath();
          poly.forEach(([px, py], i) => (i ? ctx.lineTo(px, py) : ctx.moveTo(px, py)));
          ctx.closePath();
          ctx.strokeStyle = css(pal.focus);
          ctx.lineWidth = 2;
          ctx.stroke();
        }
      }
    }

    drawDims(ctx, P, pal);
    if (showCallouts) drawCallouts(ctx, P, pal, items, w, area);
    return settled;
  }

  function drawDims(ctx, P, pal) {
    ctx.save();
    ctx.strokeStyle = css(pal.ink2, 0.85);
    ctx.fillStyle = css(pal.ink2);
    ctx.lineWidth = 1;
    for (const dm of geo.dims) {
      const [a, b, ea, eb] = [dm.a, dm.b, dm.ea, dm.eb].map(P);
      const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
      if (len < 18) continue;
      ctx.globalAlpha = 0.6;
      for (const [e, p] of [[ea, a], [eb, b]]) { // extension lines, overshooting a little
        const ux = (p[0] - e[0]) / (Math.hypot(p[0] - e[0], p[1] - e[1]) || 1);
        const uy = (p[1] - e[1]) / (Math.hypot(p[0] - e[0], p[1] - e[1]) || 1);
        ctx.beginPath();
        ctx.moveTo(e[0] + ux * 3, e[1] + uy * 3);
        ctx.lineTo(p[0] + ux * 4, p[1] + uy * 4);
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
      ctx.beginPath();
      ctx.moveTo(a[0], a[1]);
      ctx.lineTo(b[0], b[1]);
      ctx.stroke();
      const ux = (b[0] - a[0]) / len;
      const uy = (b[1] - a[1]) / len;
      for (const [p, sgn] of [[a, 1], [b, -1]]) { // arrowheads
        ctx.beginPath();
        ctx.moveTo(p[0], p[1]);
        ctx.lineTo(p[0] + sgn * (ux * 7 - uy * 2.6), p[1] + sgn * (uy * 7 + ux * 2.6));
        ctx.lineTo(p[0] + sgn * (ux * 7 + uy * 2.6), p[1] + sgn * (uy * 7 - ux * 2.6));
        ctx.closePath();
        ctx.fill();
      }
      let angle = Math.atan2(b[1] - a[1], b[0] - a[0]);
      if (angle > Math.PI / 2) angle -= Math.PI;
      if (angle < -Math.PI / 2) angle += Math.PI;
      ctx.save();
      ctx.translate((a[0] + b[0]) / 2, (a[1] + b[1]) / 2);
      ctx.rotate(angle);
      ctx.font = `11px ${FONT_MONO}`;
      ctx.textAlign = "center";
      ctx.lineWidth = 4;
      ctx.strokeStyle = css(pal.sheet);
      ctx.strokeText(dm.text, 0, -5);
      ctx.fillText(dm.text, 0, -5);
      ctx.restore();
    }
    ctx.restore();
  }

  function drawCallouts(ctx, P, pal, items, w, area) {
    const byId = new Map(items.map((it) => [it.box.id, it]));
    const labels = geo.callouts
      .filter((c) => byId.has(c.id))
      .map((c) => ({ ...c, anchor: byId.get(c.id).center }))
      .sort((a, b) => a.anchor[1] - b.anchor[1]);
    // Spread the labels down the right column, at least 42 px apart, in anchor order.
    const lx = w - 222;
    const maxW = w - lx - 10;
    // Details too long for the column wrap at their " · " separators.
    ctx.font = `10.5px ${FONT_MONO}`;
    for (const l of labels) {
      l.lines = [];
      for (const part of l.detail.split(" · ")) {
        const last = l.lines.length ? `${l.lines.at(-1)} · ${part}` : part;
        if (l.lines.length && ctx.measureText(last).width <= maxW) l.lines[l.lines.length - 1] = last;
        else l.lines.push(part);
      }
    }
    const top = area.y + 8;
    const bottom = area.y + area.h - 8;
    const height = (l) => 26 + 13 * l.lines.length;
    let y = top;
    for (const l of labels) {
      l.y = Math.max(y, Math.min(l.anchor[1], bottom));
      y = l.y + height(l);
    }
    for (let i = labels.length - 1, limit = bottom; i >= 0; i--) {
      labels[i].y = Math.min(labels[i].y, limit);
      limit = labels[i].y - (i > 0 ? height(labels[i - 1]) : 0);
    }
    ctx.save();
    for (const l of labels) {
      ctx.strokeStyle = css(pal.ink2, 0.7);
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(l.anchor[0], l.anchor[1]);
      ctx.lineTo(lx - 14, l.y);
      ctx.lineTo(lx - 4, l.y);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(l.anchor[0], l.anchor[1], 3, 0, Math.PI * 2);
      ctx.fillStyle = css(pal.ink);
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = css(pal.sheet);
      ctx.stroke();
      ctx.textAlign = "left";
      ctx.fillStyle = css(pal.ink);
      ctx.font = `700 12.5px ${FONT_UI}`;
      ctx.fillText(l.title, lx, l.y - 2);
      ctx.fillStyle = css(pal.ink2);
      ctx.font = `10.5px ${FONT_MONO}`;
      l.lines.forEach((line, i) => ctx.fillText(line, lx, l.y + 13 + 13 * i));
    }
    ctx.restore();
  }

  // ---- interaction ----------------------------------------------------------------------
  function hitAt(e) {
    const r = canvas.getBoundingClientRect();
    const x = e.clientX - r.left;
    const y = e.clientY - r.top;
    for (let i = hits.length - 1; i >= 0; i--) if (pointInPoly(x, y, hits[i].poly)) return { id: hits[i].id, x, y };
    return null;
  }
  function showTip(hit) {
    const box = shown.get(hit.id);
    if (!box || !spec) return hideTip();
    const [title, line1, line2] = describe(box, spec);
    tip.replaceChildren(mk("div", "tt-head", title), mk("div", "tt-label", line1), mk("strong", "", line2));
    tip.hidden = false;
    const tw = tip.offsetWidth || 220;
    tip.style.left = `${hit.x + 16 + tw > el.clientWidth ? hit.x - tw - 16 : hit.x + 16}px`;
    tip.style.top = `${Math.max(4, hit.y - 20)}px`;
  }
  function hideTip() {
    tip.hidden = true;
  }
  function animateView(toView) {
    viewAnim = reduceMotion ? null : { from: { ...view }, to: toView, t0: performance.now() };
    if (reduceMotion) view = { ...toView };
    schedule();
  }

  canvas.addEventListener("pointerdown", (e) => {
    if (e.button !== 0) return;
    canvas.setPointerCapture(e.pointerId);
    drag = { x: e.clientX, y: e.clientY, yaw: view.yaw, pitch: view.pitch };
    viewAnim = null;
    hideTip();
  });
  canvas.addEventListener("pointermove", (e) => {
    if (drag) {
      view.yaw = drag.yaw - (e.clientX - drag.x) * 0.008;
      view.pitch = clamp(drag.pitch + (e.clientY - drag.y) * 0.006, -0.2, 1.45);
      schedule();
      return;
    }
    const hit = hitAt(e);
    const id = hit ? hit.id : null;
    if (id !== hover) {
      hover = id;
      schedule();
    }
    if (hit) showTip(hit);
    else hideTip();
  });
  const endDrag = () => {
    drag = null;
  };
  canvas.addEventListener("pointerup", endDrag);
  canvas.addEventListener("pointercancel", endDrag);
  canvas.addEventListener("pointerleave", () => {
    if (hover) {
      hover = null;
      schedule();
    }
    hideTip();
  });
  canvas.addEventListener("dblclick", () => animateView({ ...DEFAULT_VIEW }));
  canvas.addEventListener("wheel", (e) => {
    if (!(e.ctrlKey || e.metaKey)) return; // plain scrolling keeps scrolling the page
    e.preventDefault();
    view.zoom = clamp(view.zoom * Math.exp(-e.deltaY * 0.0015), 0.5, 4);
    schedule();
  }, { passive: false });
  canvas.addEventListener("keydown", (e) => {
    const steps = { ArrowLeft: ["yaw", 0.12], ArrowRight: ["yaw", -0.12], ArrowUp: ["pitch", -0.08], ArrowDown: ["pitch", 0.08] };
    if (steps[e.key]) {
      const [k, dv] = steps[e.key];
      view[k] = k === "pitch" ? clamp(view[k] + dv, -0.2, 1.45) : view[k] + dv;
    } else if (e.key === "+" || e.key === "=") view.zoom = clamp(view.zoom * 1.2, 0.5, 4);
    else if (e.key === "-") view.zoom = clamp(view.zoom / 1.2, 0.5, 4);
    else if (e.key === "0" || e.key === "Home") return animateView({ ...DEFAULT_VIEW });
    else return;
    e.preventDefault();
    schedule();
  });
  reset.addEventListener("click", () => animateView({ ...DEFAULT_VIEW }));
  zoomIn.addEventListener("click", () => animateView({ ...view, zoom: clamp(view.zoom * 1.3, 0.5, 4) }));
  zoomOut.addEventListener("click", () => animateView({ ...view, zoom: clamp(view.zoom / 1.3, 0.5, 4) }));

  const ro = new ResizeObserver(() => schedule());
  ro.observe(canvas);
  // Repaint when the theme changes, and once the fonts arrive.
  const mo = new MutationObserver(() => schedule());
  mo.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  const mq = matchMedia("(prefers-color-scheme: dark)");
  mq.addEventListener("change", schedule);
  if (document.fonts) document.fonts.ready.then(schedule);

  function destroy() {
    cancelAnimationFrame(raf);
    ro.disconnect();
    mo.disconnect();
    mq.removeEventListener("change", schedule);
  }
  return { el, update, destroy };
}

// ---- parameter breakdown ----------------------------------------------------------------
const PARTS = [ // fixed order, so each part keeps its color
  ["attention", "Attention", "var(--series-1)"],
  ["mlp", "MLP", "var(--series-2)"],
  ["embedding", "Token embedding", "var(--series-3)"],
  ["head", "Output head", "var(--series-4)"],
];

export function paramBar() {
  const title = mk("div", "pbar-title");
  const bar = mk("div", "pbar");
  bar.setAttribute("role", "img");
  const legend = mk("div", "pbar-legend");
  const tip = mk("div", "tooltip");
  tip.hidden = true;
  const el = mk("div", "pbar-wrap");
  el.append(title, bar, legend, tip);

  function update(s) {
    bar.replaceChildren();
    legend.replaceChildren();
    el.hidden = !s;
    if (!s) return;
    title.replaceChildren(mk("span", "", "Where the parameters are"), mk("strong", "", compact(s.total)));
    const parts = PARTS.filter(([key]) => s.params[key] > 0);
    bar.setAttribute("aria-label", parts.map(([key, label]) => `${label} ${compact(s.params[key])} (${pct(s.params[key], s.total)})`).join(", "));
    for (const [key, label, color] of parts) {
      const value = s.params[key];
      const seg = mk("span", "pbar-seg");
      seg.style.flexGrow = String(value);
      seg.style.background = color;
      seg.addEventListener("pointermove", (e) => {
        tip.replaceChildren(mk("strong", "", compact(value)), mk("span", "tt-label", ` ${label} · ${pct(value, s.total)}`));
        tip.hidden = false;
        const r = el.getBoundingClientRect();
        const x = e.clientX - r.left;
        const tw = tip.offsetWidth || 180;
        tip.style.left = `${Math.min(Math.max(0, x - tw / 2), el.clientWidth - tw)}px`;
        tip.style.top = `${bar.offsetTop + bar.offsetHeight + 6}px`;
      });
      seg.addEventListener("pointerleave", () => (tip.hidden = true));
      bar.append(seg);
      const item = mk("span", "pbar-item");
      const sw = mk("span", "pbar-swatch");
      sw.style.background = color;
      item.append(sw, mk("span", "", `${label} `), mk("strong", "", compact(value)), mk("span", "muted", ` ${pct(value, s.total)}`));
      legend.append(item);
    }
    if (s.tied) legend.append(mk("span", "pbar-item muted", "Output head shares the embedding"));
    legend.append(mk("span", "pbar-item muted", `Norms ${compact(s.params.norms)}`));
  }
  return { el, update };
}
