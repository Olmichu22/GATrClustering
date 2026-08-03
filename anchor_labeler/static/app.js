/* Anchor labeler — front-end.
 *
 * Talks only to /api/*, so it is dataset agnostic: classes, colours, axis names
 * and sampling parameters all come from the server config. The 3D view remaps
 * the configured beam axis onto the horizontal screen axis (pointing right)
 * while keeping full free rotation. */

const S = {
  cfg: null,
  queue: [],
  cursor: 0,
  summary: {},
  event: null,
  camera: null,          // preserved across events until the user hits "r"
  fullDetector: true,    // fixed axes on the whole detector envelope ("f")
};

const $ = (id) => document.getElementById(id);

// ─── plumbing ────────────────────────────────────────────────────────────────

async function api(path, opts) {
  const res = await fetch(path, opts);
  const data = await res.json().catch(() => ({ error: "respuesta no JSON" }));
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}
const post = (path, body) => api(path, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body || {}),
});

let toastTimer = null;
function toast(msg, kind) {
  const t = $("toast");
  t.textContent = msg;
  t.className = "toast " + (kind || "");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.add("hidden"), 4000);
}

const classOf = (i) => (S.cfg ? S.cfg.classes.find((c) => c.index === i) : null);
const classNameOf = (i) => (classOf(i) ? classOf(i).name : (i === null || i === undefined ? "—" : i));
const classColorOf = (i) => (classOf(i) ? classOf(i).color : "#888");

// ─── rendering ───────────────────────────────────────────────────────────────

function renderClassControls() {
  const box = $("class-buttons");
  box.innerHTML = "";
  S.cfg.classes.forEach((c) => {
    const b = document.createElement("button");
    b.className = "cls";
    b.style.borderColor = c.color;
    b.dataset.index = c.index;
    b.innerHTML = `<span class="dot" style="background:${c.color}"></span>${c.name} <kbd>${c.key}</kbd>`;
    b.onclick = () => decide("kept", c.index);
    box.appendChild(b);
  });

  const filter = $("class-filter");
  filter.innerHTML = "<span class='muted'>clases a muestrear</span>";
  S.cfg.classes.forEach((c) => {
    const lab = document.createElement("label");
    lab.className = "chip" + (c.samplable ? "" : " disabled");
    lab.title = c.samplable ? "" : "esta clase no tiene fuente de candidatos en el config";
    lab.innerHTML = `<input type="checkbox" ${c.samplable ? "checked" : "disabled"}
       data-index="${c.index}"> <span class="dot" style="background:${c.color}"></span>${c.name}`;
    filter.appendChild(lab);
  });
}

function renderSummary() {
  const s = S.summary || {};
  const per = Object.entries(s.kept_per_class || {})
    .map(([ci, n]) => `<span class="chip"><span class="dot" style="background:${classColorOf(+ci)}"></span>${classNameOf(+ci)}: <b>${n}</b></span>`)
    .join(" ") || "<span class='muted'>—</span>";
  $("summary").innerHTML = `
    <div class="kv"><span>Ronda</span><b>${s.round_index ?? 0}</b></div>
    <div class="kv"><span>Cola</span><b>${s.queue_len ?? 0}</b> (pendientes ${s.pending_in_queue ?? 0})</div>
    <div class="kv"><span>Anchors guardados</span><b>${s.kept ?? 0}</b></div>
    <div class="kv sub"><span>confirmados / corregidos</span><b>${s.kept_confirmed ?? 0} / ${s.kept_corrected ?? 0}</b></div>
    <div class="kv"><span>Ignorados</span><b>${s.ignored ?? 0}</b></div>
    <div class="kv"><span>Saltados</span><b>${s.skipped ?? 0}</b></div>
    <div class="per-class">${per}</div>`;
}

function renderQueue() {
  const box = $("queue");
  box.innerHTML = "";
  $("queue-count").textContent = S.queue.length ? `${S.cursor + 1} / ${S.queue.length}` : "";
  S.queue.forEach((q, i) => {
    const el = document.createElement("div");
    const st = q.status || "pending";
    el.className = "qitem " + st + (i === S.cursor ? " current" : "");
    const shown = q.status === "kept" ? q.label : q.proposed;
    el.innerHTML = `<span class="dot" style="background:${classColorOf(shown)}"></span>
      <span class="qidx">#${q.index}</span>
      <span class="qst">${st === "pending" ? "" : st}</span>`;
    el.onclick = () => goto(i);
    box.appendChild(el);
  });
  const cur = box.querySelector(".current");
  if (cur) cur.scrollIntoView({ block: "nearest" });
}

function renderEventInfo() {
  const ev = S.event;
  if (!ev) return;
  $("ev-index").textContent = `#${ev.index}`;
  $("ev-position").textContent = S.queue.length ? `(${S.cursor + 1} de ${S.queue.length})` : "";
  const prop = ev.proposed;
  $("ev-proposed").innerHTML = prop === null || prop === undefined
    ? "<span class='muted'>sin propuesta</span>"
    : `<span class="muted">propuesta:</span> <span class="tag" style="background:${classColorOf(prop)}">${classNameOf(prop)}</span>`;
  const d = ev.decision;
  $("ev-status").innerHTML = !d
    ? "<span class='muted'>sin decidir</span>"
    : d.status === "kept"
      ? `<span class="tag kept">guardado: ${classNameOf(d.label)}${d.label === d.proposed ? " (confirmado)" : " (corregido)"}</span>`
      : `<span class="tag ${d.status}">${d.status}</span>`;

  $("meta").innerHTML = Object.entries(ev.meta || {})
    .map(([k, v]) => `<span class="kvchip"><span class="muted">${k}</span> <b>${typeof v === "number" ? (Number.isInteger(v) ? v : v.toFixed(3)) : v}</b></span>`)
    .join("");

  document.querySelectorAll("#class-buttons .cls").forEach((b) => {
    b.classList.toggle("proposed", +b.dataset.index === prop);
    b.classList.toggle("chosen", !!d && d.status === "kept" && +b.dataset.index === d.label);
  });
}

/* Wireframe of the detector envelope: 12 edges as one line trace with nulls
 * between segments, so the reviewer always sees the full volume as reference. */
function envelopeTrace(bx, by, bz) {
  const [x0, x1] = bx, [y0, y1] = by, [z0, z1] = bz;
  const C = [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
             [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]];
  const EDGES = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4],
                 [0, 4], [1, 5], [2, 6], [3, 7]];
  const x = [], y = [], z = [];
  EDGES.forEach(([a, b]) => {
    x.push(C[a][0], C[b][0], null);
    y.push(C[a][1], C[b][1], null);
    z.push(C[a][2], C[b][2], null);
  });
  return {
    type: "scatter3d", mode: "lines", x, y, z,
    line: { color: "#33405c", width: 2 },
    hoverinfo: "skip", showlegend: false, name: "envelope",
  };
}

function plotEvent(ev) {
  const coords = ev.coords || {};
  const beam = (S.cfg.dataset.beam_axis || "z");
  const names = Object.keys(coords);
  const others = names.filter((n) => n !== beam);
  const ax = [beam, others[0], others[1]].filter(Boolean);   // screen X, Y, Z
  const [aX, aY, aZ] = ax;

  const color = ev.color;
  const marker = {
    size: 3.2,
    opacity: 0.9,
    color: color || "#4da3ff",
    colorscale: "Viridis",
    showscale: !!color,
    colorbar: color ? { title: { text: S.cfg.dataset.hit_color_label || "" }, thickness: 10, len: 0.6 } : undefined,
  };

  const hover = [];
  const n = (coords[aX] || []).length;
  for (let i = 0; i < n; i++) {
    let s = `${aX}=${coords[aX][i]}<br>${aY}=${coords[aY][i]}<br>${aZ}=${coords[aZ][i]}`;
    if (color) s += `<br>${S.cfg.dataset.hit_color_label || "c"}=${color[i]}`;
    for (const [k, v] of Object.entries(ev.hit_extra || {})) s += `<br>${k}=${v[i]}`;
    hover.push(s);
  }

  const trace = {
    type: "scatter3d", mode: "markers",
    x: coords[aX], y: coords[aY], z: coords[aZ],
    marker, hoverinfo: "text", text: hover, name: "hits",
  };

  // Fixed detector envelope: every event is drawn at the SAME scale, so a
  // 70-hit muon and a 1500-hit shower are directly comparable and the plot does
  // not "breathe" from event to event. `S.fullDetector = false` falls back to
  // per-event autoscaling.
  const bounds = (S.cfg.dataset.bounds) || {};
  const fixed = S.fullDetector && bounds[aX] && bounds[aY] && bounds[aZ];

  const axis = (title, rng) => ({
    title: { text: title }, color: "#9aa4b2",
    gridcolor: "#2a3346", zerolinecolor: "#3b465c",
    backgroundcolor: "#0e1421", showbackground: true,
    ...(rng ? { range: rng.slice(), autorange: false } : {}),
  });

  const data = [trace];
  if (fixed && S.cfg.dataset.show_envelope) {
    data.unshift(envelopeTrace(bounds[aX], bounds[aY], bounds[aZ]));
  }

  const layout = {
    margin: { l: 0, r: 0, t: 0, b: 0 },
    paper_bgcolor: "#0e1421", plot_bgcolor: "#0e1421",
    font: { color: "#d7dce5" },
    showlegend: false,
    scene: {
      aspectmode: "data",
      xaxis: axis(`${aX} (haz →)`, fixed ? bounds[aX] : null),
      yaxis: axis(aY, fixed ? bounds[aY] : null),
      zaxis: axis(aZ, fixed ? bounds[aZ] : null),
      camera: S.camera || { eye: { x: 0.15, y: -2.1, z: 0.75 }, up: { x: 0, y: 0, z: 1 } },
    },
  };
  Plotly.react("plot", data, layout, { displaylogo: false, responsive: true });
  const gd = $("plot");
  if (!gd._camHooked) {
    gd.on("plotly_relayout", (e) => { if (e["scene.camera"]) S.camera = e["scene.camera"]; });
    gd._camHooked = true;
  }
}

// ─── actions ─────────────────────────────────────────────────────────────────

async function loadEvent(index) {
  if (index === null || index === undefined) {
    S.event = null;
    Plotly.purge("plot");
    $("ev-index").textContent = "—";
    $("meta").innerHTML = "<span class='muted'>cola vacía: lanza un muestreo</span>";
    return;
  }
  S.event = await api(`/api/event/${index}`);
  plotEvent(S.event);
  renderEventInfo();
}

async function goto(i) {
  if (!S.queue.length) return;
  S.cursor = Math.max(0, Math.min(i, S.queue.length - 1));
  await post("/api/cursor", { cursor: S.cursor });
  renderQueue();
  await loadEvent(S.queue[S.cursor].index);
}

async function decide(status, label) {
  if (!S.event) return;
  const index = S.event.index;
  const res = await post("/api/decision", { index, status, label: label ?? null });
  S.summary = res.summary;
  const q = S.queue.find((x) => x.index === index);
  if (q) { q.status = res.decision.status; q.label = res.decision.label; }
  renderSummary();
  const next = Math.min(S.cursor + 1, S.queue.length - 1);
  if (next !== S.cursor) { await goto(next); } else { renderQueue(); await loadEvent(index); }
  markSaved();
}

async function undo() {
  if (!S.event) return;
  const res = await post("/api/undo", { index: S.event.index });
  S.summary = res.summary;
  const q = S.queue.find((x) => x.index === S.event.index);
  if (q) { q.status = null; q.label = null; }
  renderSummary(); renderQueue();
  await loadEvent(S.event.index);
  markSaved();
}

function samplingParams() {
  const classes = [...document.querySelectorAll("#class-filter input:checked")]
    .map((i) => +i.dataset.index);
  return {
    strategy: $("p-strategy").value,
    n_per_class: +$("p-n").value,
    seed: +$("p-seed").value,
    window_std: +$("p-wstd").value,
    prefer_unseen: $("p-unseen").checked,
    classes,
    append: $("p-append").checked,
  };
}

async function sample() {
  $("btn-sample").disabled = true;
  try {
    const st = await post("/api/sample", samplingParams());
    applyState(st);
    const rows = (st.round_info.stats || []).map((s) =>
      `<div class="kv"><span><span class="dot" style="background:${classColorOf(s.class_index)}"></span>${s.class_name}</span>
       <b>${s.drawn}</b> <span class="muted">de ${s.pool}${s.nhits_mean !== null ? `, nHits ~${s.nhits_mean.toFixed(0)} [${s.nhits_min.toFixed(0)}-${s.nhits_max.toFixed(0)}]` : ""}</span></div>`).join("");
    $("round-stats").innerHTML = rows;
    toast(`Ronda ${st.summary.round_index}: ${st.round_info.n_proposals} propuestas`, "ok");
  } catch (e) {
    toast("Muestreo fallido: " + e.message, "err");
  } finally {
    $("btn-sample").disabled = false;
  }
}

async function exportAnchors() {
  const withH5 = confirm("¿Escribir también una copia del h5 con anchor_label?\n\n" +
                         "Aceptar = json + csv + yml + h5\nCancelar = solo json + csv + yml");
  try {
    const res = await post("/api/export", { write_h5: withH5 });
    const per = Object.entries(res.per_class).map(([k, v]) => `${k}:${v}`).join("  ");
    toast(`Exportados ${res.n_anchors} anchors (${per}) → ${res.json}${res.h5 ? " + h5" : ""}`, "ok");
    console.log("export:", res);
  } catch (e) {
    toast("Export fallido: " + e.message, "err");
  }
}

let savedTimer = null;
function markSaved() {
  const p = $("save-state");
  p.textContent = "guardado " + new Date().toLocaleTimeString();
  p.classList.add("ok");
  clearTimeout(savedTimer);
  savedTimer = setTimeout(() => p.classList.remove("ok"), 1500);
}

function applyState(st) {
  S.cfg = st.config;
  S.queue = st.queue;
  S.cursor = st.cursor || 0;
  S.summary = st.summary;
  renderSummary();
  renderQueue();
  loadEvent(S.queue.length ? S.queue[Math.min(S.cursor, S.queue.length - 1)].index : null);
}

function toggleFullDetector() {
  S.fullDetector = $("p-full").checked;
  if (S.event) plotEvent(S.event);
}

function bindKeys() {
  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
    const k = e.key.toLowerCase();
    if (k === "enter" || k === " ") { e.preventDefault(); decide("kept", S.event ? S.event.proposed : null); return; }
    if (k === "i") { decide("ignored", null); return; }
    if (k === "s") { decide("skipped", null); return; }
    if (k === "u") { undo(); return; }
    if (k === "arrowright") { goto(S.cursor + 1); return; }
    if (k === "arrowleft") { goto(S.cursor - 1); return; }
    if (k === "r") { S.camera = null; if (S.event) plotEvent(S.event); return; }
    if (k === "f") { $("p-full").checked = !$("p-full").checked; toggleFullDetector(); return; }
    const cls = S.cfg.classes.find((c) => (c.key || String(c.index)).toLowerCase() === k);
    if (cls) decide("kept", cls.index);
  });
}

async function init() {
  const st = await api("/api/state");
  S.cfg = st.config;
  $("cfg-name").textContent = `· ${st.config.name} · ${st.n_events} eventos`;
  const sp = st.config.sampling;
  $("p-strategy").value = sp.strategy;
  $("p-n").value = sp.n_per_class;
  $("p-seed").value = sp.seed;
  $("p-wstd").value = sp.window_std;
  renderClassControls();
  applyState(st);
  $("btn-sample").onclick = sample;
  $("btn-export").onclick = exportAnchors;
  $("btn-save").onclick = async () => { await post("/api/save"); markSaved(); toast("Sesión guardada", "ok"); };
  $("btn-prev").onclick = () => goto(S.cursor - 1);
  $("btn-next").onclick = () => goto(S.cursor + 1);
  $("btn-ignore").onclick = () => decide("ignored", null);
  $("btn-skip").onclick = () => decide("skipped", null);
  $("btn-undo").onclick = undo;
  $("p-full").onchange = toggleFullDetector;
  S.fullDetector = $("p-full").checked;
  bindKeys();
  markSaved();
}

init().catch((e) => toast("Error de arranque: " + e.message, "err"));
