/* TrialMatch dashboard. Vanilla JS, no external requests. All data-derived text goes in via textContent. */
"use strict";

// ---------- helpers ----------
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const SVGNS = "http://www.w3.org/2000/svg";

function h(tag, attrs = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "style") Object.assign(el.style, v);
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "text") el.textContent = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) if (kid != null && kid !== false) el.append(kid instanceof Node ? kid : String(kid));
  return el;
}
function s(tag, attrs = {}, ...kids) {
  const el = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null) continue;
    if (k === "text") el.textContent = v; else el.setAttribute(k, v);
  }
  for (const kid of kids.flat()) if (kid) el.append(kid);
  return el;
}
const fmt = (x, d = 3) => (x == null || Number.isNaN(x) ? "n/a" : Number(x).toFixed(d));
const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* storage unavailable */ } },
};
function fmtDuration(sec) {
  if (!isFinite(sec) || sec < 0) return "";
  if (sec < 60) return `${Math.round(sec)}s`;
  const m = Math.round(sec / 60);
  return m < 60 ? `${m} min` : `${Math.floor(m / 60)} h ${m % 60} min`;
}

const LABELS = {
  likely_eligible: { text: "Likely eligible", cls: "good", icon: "✓" },
  possibly_eligible: { text: "Needs review", cls: "warning", icon: "?" },
  likely_ineligible: { text: "Likely ineligible", cls: "critical", icon: "✕" },
};
const OUTCOMES = {
  met: { text: "Met", cls: "good", icon: "✓" },
  not_met: { text: "Not met", cls: "critical", icon: "✕" },
  unknown: { text: "Unknown", cls: "neutral", icon: "?" },
};
const SYSTEM_NAMES = {
  bm25: "BM25", single_call: "Single LLM call", multiagent: "Multi-agent", no_critic: "Multi-agent, no critic",
  multiagent_rag: "Multi-agent + RAG", local_parse: "Multi-agent, local parser",
};
const METRIC_NAMES = { "ndcg@10": "nDCG@10", "p@5": "P@5", elig_acc: "Eligible-vs-excluded accuracy" };
const badge = (def) => h("span", { class: `badge ${def.cls}` }, h("i", { text: def.icon }), def.text);

// ---------- tooltip ----------
const tip = $("#tooltip");
function showTip(ev, value, sub) {
  tip.replaceChildren(h("div", { class: "tv", text: value }), ...(sub || []).map((t) => h("div", { class: "ts", text: t })));
  tip.hidden = false;
  const r = tip.getBoundingClientRect();
  let x = ev.clientX + 14, y = ev.clientY + 14;
  if (x + r.width > innerWidth - 8) x = ev.clientX - r.width - 14;
  if (y + r.height > innerHeight - 8) y = ev.clientY - r.height - 14;
  tip.style.left = `${x}px`; tip.style.top = `${y}px`;
}
const hideTip = () => { tip.hidden = true; };
function tipOn(el, valueFn, subFn) {
  el.addEventListener("pointermove", (e) => showTip(e, valueFn(), subFn ? subFn() : []));
  el.addEventListener("pointerleave", hideTip);
  el.addEventListener("focus", () => { const r = el.getBoundingClientRect(); showTip({ clientX: r.right, clientY: r.top }, valueFn(), subFn ? subFn() : []); });
  el.addEventListener("blur", hideTip);
}

// ---------- theme & tabs ----------
function applyTheme(t) { if (t) document.documentElement.dataset.theme = t; }
applyTheme(store.get("tm-theme"));
$("#theme-btn").addEventListener("click", () => {
  const cur = document.documentElement.dataset.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  const next = cur === "dark" ? "light" : "dark";
  applyTheme(next); store.set("tm-theme", next);
});
function goto(view) {
  $$(".tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.view === view)));
  $$(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${view}`));
  store.set("tm-view", view);
  if (view === "eval" && !evalState.loaded) loadEvalRuns();
  if (view === "about") drawArch();
}
$$(".tabs button").forEach((b) => b.addEventListener("click", () => goto(b.dataset.view)));
document.addEventListener("click", (e) => {
  const a = e.target.closest("[data-goto]");
  if (a) { e.preventDefault(); goto(a.dataset.goto); }
});

// ---------- state ----------
const state = {
  meta: null, running: false, abort: null,
  candidates: [], matches: new Map(), order: null, current: null,
  profile: null, demoFail: {}, trialTimes: [], startedAt: 0, source: "live", title: "",
  filter: "all", open: new Set(), noteForRun: "",
};

// ---------- note editor & evidence view ----------
const noteEl = $("#note"), noteRead = $("#note-read");
function updateCount() {
  const n = noteEl.value.trim() ? noteEl.value.trim().split(/\s+/).length : 0;
  $("#note-count").textContent = `${n} word${n === 1 ? "" : "s"}`;
}
noteEl.addEventListener("input", () => {
  updateCount();
  if (!state.running && noteEl.value !== state.noteForRun) setNoteView("edit", true);
});
function setNoteView(mode, disableRead = false) {
  if (disableRead) $("#noteview-read").disabled = true;
  $$("[data-noteview]").forEach((b) => b.classList.toggle("on", b.dataset.noteview === mode));
  noteEl.hidden = mode !== "edit";
  noteRead.hidden = mode !== "read";
}
$$("[data-noteview]").forEach((b) => b.addEventListener("click", () => !b.disabled && setNoteView(b.dataset.noteview)));

/* Collect evidence spans (profile facts + criterion evidence) and render the note with <mark>s. */
function evidenceSpans() {
  const spans = [];
  const p = state.profile;
  if (p) {
    for (const g of ["conditions", "medications", "procedures", "negatives", "other"])
      (p[g] || []).forEach((f, i) => f.evidence && spans.push({ text: f.evidence, key: `p-${g}-${i}` }));
    (p.labs || []).forEach((l, i) => l.evidence && spans.push({ text: l.evidence, key: `p-labs-${i}` }));
  }
  for (const m of state.matches.values())
    m.results.forEach((r, i) => r.evidence && spans.push({ text: r.evidence, key: `c-${m.nct_id}-${i}` }));
  return spans;
}
function renderNoteRead() {
  const text = state.noteForRun;
  const lower = text.toLowerCase();
  const ranges = [];
  for (const sp of evidenceSpans()) {
    const needle = sp.text.trim().toLowerCase();
    if (needle.length < 2) continue;
    const at = lower.indexOf(needle);
    if (at >= 0) ranges.push({ a: at, b: at + needle.length, keys: [sp.key] });
  }
  ranges.sort((x, y) => x.a - y.a || y.b - x.b);
  const merged = [];
  for (const r of ranges) {
    const last = merged[merged.length - 1];
    if (last && r.a <= last.b) { last.b = Math.max(last.b, r.b); last.keys.push(...r.keys); }
    else merged.push({ ...r, keys: [...r.keys] });
  }
  const frag = [];
  let pos = 0;
  for (const r of merged) {
    if (r.a > pos) frag.push(document.createTextNode(text.slice(pos, r.a)));
    const mk = h("mark", { text: text.slice(r.a, r.b) });
    mk.dataset.keys = ` ${r.keys.join(" ")} `;
    frag.push(mk);
    pos = r.b;
  }
  frag.push(document.createTextNode(text.slice(pos)));
  noteRead.replaceChildren(...frag);
  $("#noteview-read").disabled = merged.length === 0 && !text;
}
function highlightKey(key, on, scroll = false) {
  const marks = $$(`mark`, noteRead).filter((m) => m.dataset.keys.includes(` ${key} `));
  marks.forEach((m) => m.classList.toggle("hot", on));
  if (on && scroll && marks[0]) {
    setNoteView("read");
    marks[0].scrollIntoView({ block: "nearest", behavior: "smooth" });
  }
  return marks.length > 0;
}

// ---------- profile ----------
function renderProfile() {
  const p = state.profile, box = $("#profile");
  $("#profile-card").hidden = !p;
  if (!p) return;
  const groups = [];
  const basics = [];
  if (p.age != null) basics.push(h("span", { class: "chip" }, "Age ", h("b", { text: String(p.age) })));
  if (p.sex) basics.push(h("span", { class: "chip", text: p.sex[0].toUpperCase() + p.sex.slice(1) }));
  if (basics.length) groups.push(["Demographics", basics]);
  const factChips = (list, g, neg = false) => list.map((f, i) => {
    const c = h("span", { class: `chip${neg ? " neg" : ""}`, tabindex: "0", text: f.text });
    hoverLink(c, `p-${g}-${i}`);
    return c;
  });
  const add = (title, g, neg) => { const l = p[g] || []; if (l.length) groups.push([title, factChips(l, g, neg)]); };
  add("Conditions", "conditions"); add("Medications", "medications"); add("Procedures", "procedures");
  if ((p.labs || []).length) groups.push(["Labs", p.labs.map((l, i) => {
    const c = h("span", { class: "chip", tabindex: "0" }, `${l.name} `, h("b", { text: `${l.value}${l.unit ? " " + l.unit : ""}` }));
    hoverLink(c, `p-labs-${i}`);
    return c;
  })]);
  add("Negatives", "negatives", true); add("Other", "other");
  box.replaceChildren(...groups.map(([t, chips]) => h("div", { class: "pgroup" }, h("h4", { text: t }), h("div", { class: "chips" }, chips))));
  if (!groups.length) box.replaceChildren(h("p", { class: "small muted", text: "No facts extracted." }));
}
function hoverLink(el, key) {
  el.addEventListener("pointerenter", () => { if (highlightKey(key, true)) setNoteView("read"); });
  el.addEventListener("pointerleave", () => highlightKey(key, false));
  el.addEventListener("focus", () => highlightKey(key, true, true));
  el.addEventListener("blur", () => highlightKey(key, false));
}

// ---------- pipeline ----------
const NODES = ["profiler", "prefilter", "rule_checker", "eligibility_checker", "critic", "scorer"];
function setPipeline(active, doneUpTo = null) {
  $$("#pipeline li").forEach((li) => {
    const i = NODES.indexOf(li.dataset.node), a = NODES.indexOf(active);
    li.classList.remove("active", "done", "skipped");
    if (active === "__done__") li.classList.add("done");
    else if (i === a) li.classList.add("active");
    else if (i < a || (doneUpTo != null && i <= NODES.indexOf(doneUpTo))) li.classList.add("done");
  });
}
let logN = 0;
function log(agent, msg) {
  const t = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  const li = h("li", {}, h("span", { class: "t", text: t }), h("span", { class: "a", text: agent }), h("span", { text: msg }));
  const ol = $("#log");
  ol.append(li);
  ol.scrollTop = ol.scrollHeight;
  $("#log-count").textContent = `(${++logN})`;
}
function setStatus(t) { $("#run-status").textContent = t; }
function updateProgress() {
  const n = state.candidates.length, done = state.matches.size;
  $("#progress").hidden = !n;
  if (!n) return;
  $("#progress-fill").style.width = `${(100 * done) / n}%`;
  $("#progress-text").textContent = state.running
    ? `Trial ${Math.min(done + 1, n)} of ${n}${state.current ? " · " + state.current : ""}`
    : `${done} of ${n} trials screened`;
  let eta = "";
  if (state.running && state.trialTimes.length) {
    const avg = state.trialTimes.reduce((a, b) => a + b, 0) / state.trialTimes.length;
    eta = `about ${fmtDuration(avg * (n - done))} left`;
  } else if (state.running) eta = `elapsed ${fmtDuration((Date.now() - state.startedAt) / 1000)}`;
  $("#progress-eta").textContent = eta;
}
setInterval(() => state.running && updateProgress(), 1000);

// ---------- results ----------
function counts(m) {
  const c = { met: 0, not_met: 0, unknown: 0 };
  m.results.forEach((r) => c[r.outcome]++);
  return c;
}
function ring(score) {
  const r = 18, C = 2 * Math.PI * r;
  return s("svg", { class: "ring", viewBox: "0 0 44 44", "aria-hidden": "true" },
    s("circle", { class: "bg", cx: 22, cy: 22, r }),
    s("circle", { class: "fg", cx: 22, cy: 22, r, "stroke-dasharray": C, "stroke-dashoffset": C * (1 - Math.max(0, Math.min(1, score))), transform: "rotate(-90 22 22)" }));
}
function cbar(c) {
  const tot = c.met + c.not_met + c.unknown || 1;
  const bar = h("span", { class: "cbar", role: "img", "aria-label": `${c.met} met, ${c.not_met} not met, ${c.unknown} unknown` });
  [["m", c.met], ["n", c.not_met], ["u", c.unknown]].forEach(([k, v]) => v && bar.append(h("span", { class: k, style: { flex: `${v / tot} 0 0` } })));
  return bar;
}
function ckey(c) {
  return h("span", { class: "ckey" },
    h("span", { class: "m" }, h("i"), `${c.met} met`),
    h("span", { class: "n" }, h("i"), `${c.not_met} not met`),
    h("span", { class: "u" }, h("i"), `${c.unknown} unknown`));
}

function pendingCard(t, idx) {
  const running = state.running && state.current === t.nct_id;
  return h("article", { class: `trial pending${running ? " running" : ""}` },
    h("div", { class: "trial-top" },
      h("div", { class: "rank", text: "·" }),
      h("div", {},
        h("div", { class: "trial-title", text: t.title }),
        h("div", { class: "trial-meta" }, h("span", { class: "nct", text: t.nct_id }),
          running ? h("span", { class: "badge neutral" }, h("span", { class: "spinner" }), "Screening…")
                  : h("span", { class: "badge neutral", text: state.running ? `Queued · #${idx + 1} by relevance` : "Not screened" }))),
      h("div")));
}

function trialCard(m, rank) {
  const def = LABELS[m.label], c = counts(m), isOpen = state.open.has(m.nct_id);
  const card = h("article", { class: `trial${isOpen ? " open" : ""}` });
  const top = h("div", { class: "trial-top", role: "button", tabindex: "0", "aria-expanded": String(isOpen) },
    h("div", { class: "rank", text: String(rank) }),
    h("div", {},
      h("div", { class: "trial-title", text: m.title }),
      h("div", { class: "trial-meta" }, badge(def), h("span", { class: "nct", text: m.nct_id }), cbar(c), ckey(c))),
    h("div", { class: "score" },
      h("div", { class: "score-num" }, h("div", { class: "v", text: fmt(m.final_score, 2) }), h("div", { class: "l", text: "match score" })),
      ring(m.final_score),
      s("svg", { class: "chev", viewBox: "0 0 16 16", "aria-hidden": "true" }, s("path", { d: "M4 6l4 4 4-4" }))));
  tipOn(top.querySelector(".score"), () => `Score ${fmt(m.final_score, 2)}`,
    () => [`Eligibility ${fmt(m.eligibility_score, 2)} · relevance ${fmt(m.relevance_score, 2)}`]);
  const toggle = () => {
    if (state.open.has(m.nct_id)) state.open.delete(m.nct_id); else state.open.add(m.nct_id);
    card.replaceWith(trialCard(m, rank));
  };
  top.addEventListener("click", toggle);
  top.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(); } });
  card.append(top);
  if (isOpen) card.append(trialBody(m));
  return card;
}

function trialBody(m) {
  const body = h("div", { class: "trial-body" });
  body.append(h("p", { class: "explain", text: m.explanation }));
  if (m.plain_summary) body.append(h("p", { class: "psum", text: m.plain_summary }));
  let filt = "all";
  const list = h("div");
  const seg = h("div", { class: "seg", role: "group", "aria-label": "Filter criteria" });
  const c = counts(m);
  [["all", `All ${m.results.length}`], ["not_met", `Not met ${c.not_met}`], ["unknown", `Unknown ${c.unknown}`], ["met", `Met ${c.met}`]]
    .forEach(([k, t]) => seg.append(h("button", { class: k === filt ? "on" : "", text: t, onclick: (e) => {
      filt = k; $$("button", seg).forEach((b) => b.classList.toggle("on", b === e.currentTarget)); draw();
    } })));
  body.append(h("div", { class: "crit-tools" }, h("span", { class: "small muted", text: "Criteria, as parsed from the registry" }), seg));
  body.append(list);
  function draw() {
    const groups = { inclusion: [], exclusion: [] };
    m.results.forEach((r, i) => (filt === "all" || r.outcome === filt) && groups[r.criterion.kind].push([r, i]));
    list.replaceChildren(...["inclusion", "exclusion"].filter((k) => groups[k].length).map((k) =>
      h("div", { class: "crit-group" }, h("h4", { text: `${k === "inclusion" ? "Inclusion" : "Exclusion"} criteria · ${groups[k].length}` }),
        groups[k].map(([r, i]) => critRow(m, r, i)))));
    if (!list.children.length) list.append(h("p", { class: "small muted", text: "No criteria in this group." }));
  }
  draw();
  body.append(h("div", { class: "trial-foot" },
    h("a", { class: "btn small", href: `https://clinicaltrials.gov/study/${encodeURIComponent(m.nct_id)}`, target: "_blank", rel: "noopener noreferrer", text: "View on ClinicalTrials.gov ↗" })));
  return body;
}

const SRC_NAMES = { rule: "rule", llm: "LLM", critic: "critic", grounding: "grounding", cap: "cap" };
const SRC_HELP = {
  rule: "Decided by a numeric rule, no LLM", llm: "Judged by the local eligibility checker",
  critic: "Changed by the critic after re-reading the note", grounding: "LLM answer rejected: it did not quote the note",
  cap: "Not checked: criterion limit reached",
};
function critRow(m, r, i) {
  const key = `c-${m.nct_id}-${i}`;
  const textBox = h("div", {}, h("div", { class: "crit-text", text: r.criterion.text }));
  if (r.evidence) {
    const q = h("q", { tabindex: "0", text: r.evidence, title: "Show in note" });
    q.addEventListener("click", () => { if (!highlightKey(key, true, true)) q.title = "Not found verbatim in the note"; setTimeout(() => highlightKey(key, false), 2200); });
    q.addEventListener("keydown", (e) => e.key === "Enter" && q.click());
    textBox.append(h("div", { class: "crit-ev" }, "Evidence: ", q));
  }
  if (r.note) textBox.append(h("div", { class: "crit-note", text: r.note }));
  const src = h("span", { class: `src ${r.source}`, text: SRC_NAMES[r.source] || r.source, tabindex: "0" });
  tipOn(src, () => SRC_NAMES[r.source] || r.source, () => [SRC_HELP[r.source] || ""]);
  return h("div", { class: "crit" }, h("div", {}, badge(OUTCOMES[r.outcome])), textBox, src);
}

function renderResults() {
  const box = $("#results");
  const any = state.candidates.length || state.matches.size;
  $("#empty-state").hidden = !!any;
  $("#results-head").hidden = !any;
  if (!any) { box.replaceChildren(); return; }
  const ms = [...state.matches.values()];
  const byLabel = { likely_eligible: 0, possibly_eligible: 0, likely_ineligible: 0 };
  ms.forEach((m) => byLabel[m.label]++);
  $$("#label-filter button").forEach((b) => {
    const n = b.dataset.f === "all" ? ms.length : byLabel[b.dataset.f];
    b.querySelector("span").textContent = String(n);
    b.classList.toggle("on", b.dataset.f === state.filter);
  });
  $("#results-title").textContent = state.title || "Results";
  const ranked = state.order ? state.order.map((id) => state.matches.get(id)).filter(Boolean) : rankLive(ms);
  const cards = [];
  const strip = h("div", { class: "summary-strip" },
    ["likely_eligible", "possibly_eligible", "likely_ineligible"].map((k) =>
      h("div", { class: "sum-tile" }, h("div", { class: "v", text: String(byLabel[k]) }), h("div", { class: "l" }, badge(LABELS[k])))));
  cards.push(strip);
  ranked.forEach((m, i) => (state.filter === "all" || m.label === state.filter) && cards.push(trialCard(m, i + 1)));
  if (state.filter === "all")
    state.candidates.forEach((t, i) => !state.matches.has(t.nct_id) && cards.push(pendingCard(t, i)));
  if (cards.length === 1) cards.push(h("p", { class: "muted", text: state.running ? "No trials with this label yet." : "No trials with this label." }));
  box.replaceChildren(...cards);
}
const LABEL_ORDER = { likely_eligible: 0, possibly_eligible: 1, likely_ineligible: 2 };
function rankLive(ms) {  // until the server sends the final order: same idea as scorer.rank_key
  return [...ms].sort((a, b) => LABEL_ORDER[a.label] - LABEL_ORDER[b.label] || b.final_score - a.final_score);
}
$$("#label-filter button").forEach((b) => b.addEventListener("click", () => { state.filter = b.dataset.f; renderResults(); }));
$("#export-btn").addEventListener("click", () => {
  const ms = state.order ? state.order.map((id) => state.matches.get(id)).filter(Boolean) : rankLive([...state.matches.values()]);
  const blob = new Blob([JSON.stringify({ note: state.noteForRun, profile: state.profile, matches: ms }, null, 1)], { type: "application/json" });
  const a = h("a", { href: URL.createObjectURL(blob), download: "trial_matches.json" });
  document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
});

// ---------- running a screening (SSE over POST) ----------
function resetRun(title, source) {
  Object.assign(state, { candidates: [], matches: new Map(), order: null, current: null, profile: null, demoFail: {},
    trialTimes: [], startedAt: Date.now(), source, title, filter: "all", open: new Set(), noteForRun: noteEl.value });
  $("#log").replaceChildren(); logN = 0; $("#log-count").textContent = "";
  $("#form-msg").textContent = "";
  renderProfile(); renderResults(); renderNoteRead(); setPipeline(null);
}
function setRunning(on) {
  state.running = on;
  $("#run-btn").disabled = on;
  $("#run-btn").lastChild.textContent = on ? "Screening…" : "Find trials";
  $("#cancel-btn").hidden = !on;
  noteEl.readOnly = on;
}

async function runScreening() {
  const note = noteEl.value.trim();
  if (note.length < 30) { $("#form-msg").textContent = "Please enter a longer patient description (at least a sentence or two)."; return; }
  resetRun("Screening results", "live");
  setRunning(true);
  setStatus("Starting…");
  $("#log-wrap").open = true;
  const ctrl = new AbortController();
  state.abort = ctrl;
  try {
    const res = await fetch("/api/match", {
      method: "POST", headers: { "Content-Type": "application/json" }, signal: ctrl.signal,
      body: JSON.stringify({ note, snapshot: $("#snapshot").value, top_k: +$("#topk").value,
        critic: $("#critic").checked, rag: $("#rag").checked }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error || `Server error ${res.status}`);
    }
    const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += value;
      let idx;
      while ((idx = buf.indexOf("\n\n")) >= 0) {
        const chunk = buf.slice(0, idx); buf = buf.slice(idx + 2);
        let ev = "message", data = "";
        for (const line of chunk.split("\n")) {
          if (line.startsWith("event: ")) ev = line.slice(7);
          else if (line.startsWith("data: ")) data += line.slice(6);
        }
        if (data) onEvent(ev, JSON.parse(data));
      }
    }
  } catch (e) {
    if (e.name === "AbortError") { setStatus("Stopped"); log("dashboard", "stopped by user; finished trials are kept"); }
    else { $("#form-msg").textContent = e.message; setStatus("Failed"); log("error", e.message); }
  } finally {
    setRunning(false);
    state.current = null;
    updateProgress(); renderResults(); renderNoteRead();
  }
}
function onEvent(ev, d) {
  switch (ev) {
    case "stage":
      setStatus(d.msg); log("supervisor", d.msg);
      if (d.stage === "profiler") setPipeline("profiler");
      break;
    case "candidates":
      state.candidates = d.trials;
      log("prefilter", `${d.trials.length} candidate trials retrieved${d.rag ? " (hybrid RAG)" : " (BM25)"}`);
      if (d.warning) log("prefilter", d.warning);
      renderResults(); updateProgress();
      break;
    case "profile": {
      state.profile = d.profile; state.demoFail = d.demo_fail;
      const p = d.profile, nf = ["conditions", "medications", "procedures", "labs", "negatives", "other"].reduce((a, g) => a + (p[g] || []).length, 0);
      log("profiler", `age ${p.age ?? "?"}, sex ${p.sex ?? "?"}, ${nf} facts (${fmtDuration(d.elapsed)})`);
      log("prefilter", `${Object.keys(d.demo_fail).length} trials fail age/sex rules`);
      setPipeline("rule_checker", "prefilter");
      renderProfile(); renderNoteRead(); setNoteView("read");
      break;
    }
    case "trial_start":
      state.current = d.nct_id; setStatus(`Screening trial ${d.i} of ${d.n}`);
      setPipeline("rule_checker"); renderResults(); updateProgress();
      break;
    case "node": {
      log(d.node.replace("_", " "), `[${d.i}] ${d.nct_id}: ${d.msg}`);
      const next = { rule_checker: "eligibility_checker", eligibility_checker: "critic", critic: "scorer" }[d.node];
      if (next) setPipeline(next, d.node);
      break;
    }
    case "match":
      state.matches.set(d.match.nct_id, d.match); state.trialTimes.push(d.seconds);
      setPipeline("scorer", "scorer");
      renderResults(); updateProgress(); renderNoteRead();
      break;
    case "done":
      state.order = d.order; setPipeline("__done__");
      setStatus(`Done · ${state.matches.size} trials in ${fmtDuration(d.elapsed)}`);
      log("supervisor", `ranked ${d.order.length} trials`);
      break;
    case "cancelled":
      setStatus(`Stopped after ${d.done} trials`);
      break;
    case "error":
      $("#form-msg").textContent = d.msg; setStatus("Failed"); log("error", d.msg);
      break;
  }
}
$("#run-btn").addEventListener("click", runScreening);
$("#cancel-btn").addEventListener("click", () => state.abort && state.abort.abort());
noteEl.addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey) && !state.running) runScreening(); });

// ---------- replay a saved evaluation result ----------
async function openReplay(name) {
  if (state.running) return;
  const res = await fetch(`/api/eval/detail/${encodeURIComponent(name)}`);
  if (!res.ok) return;
  const d = await res.json();
  noteEl.value = d.note; updateCount();
  resetRun(`Saved run · ${SYSTEM_NAMES[d.system] || d.system} · TREC topic ${d.topic}`, "replay");
  d.matches.forEach((m) => state.matches.set(m.nct_id, m));
  state.order = d.matches.map((m) => m.nct_id);
  $("#results-sub").textContent = "Loaded from results/. No model calls were made. The profile is not saved with evaluation runs.";
  setPipeline(null); setStatus("Showing saved results");
  renderResults(); renderNoteRead(); setNoteView("read");
  goto("screen");
  window.scrollTo({ top: 0, behavior: "smooth" });
}

// ---------- meta / init ----------
async function init() {
  const view = store.get("tm-view");
  try {
    const meta = await (await fetch("/api/meta")).json();
    state.meta = meta;
    $("#model-name").textContent = meta.model;
    $("#snapshot").replaceChildren(...meta.snapshots.map((sn) => h("option", { value: sn.key, text: `${sn.label} · ${sn.count}` })));
    $("#example").replaceChildren(h("option", { value: "", text: "Choose…" }), ...meta.examples.map((e) => h("option", { value: e.id, text: e.label })));
    if (!meta.snapshots.length) $("#form-msg").textContent = "No trial snapshot found. Run: python -m scripts.download_data --demo …";
    if (!meta.rag_index) { $("#rag").checked = false; $("#rag").disabled = true; $("#rag-hint").textContent = "Index not built. Run: python -m scripts.build_index"; }
    noteEl.value = meta.examples[0] ? meta.examples[0].text : "";
    updateCount();
    if (meta.busy) $("#form-msg").textContent = "Another screening is running on this machine right now.";
  } catch (e) {
    $("#form-msg").textContent = "Could not reach the dashboard server.";
  }
  // deep links: #eval, #about, #replay=multiagent_4 (open a saved run, optionally &open=all)
  const hash = new URLSearchParams(location.hash.slice(1));
  if (hash.has("replay")) {
    await openReplay(hash.get("replay"));
    if (hash.get("open") === "all") { state.matches.forEach((_, k) => state.open.add(k)); renderResults(); }
    else if (state.order && state.order[0]) { state.open.add(state.order[0]); renderResults(); }
  } else if (hash.has("eval") || hash.has("about") || hash.has("screen")) goto([...hash.keys()][0]);
  else if (view && view !== "screen") goto(view);
}
$("#example").addEventListener("change", (e) => {
  const ex = state.meta.examples.find((x) => x.id === e.target.value);
  if (ex && !state.running) { noteEl.value = ex.text; updateCount(); setNoteView("edit", true); }
});
$("#topk").addEventListener("input", (e) => {
  $("#topk-val").textContent = e.target.value;
  const n = +e.target.value;
  $("#topk-hint").textContent = `Roughly ${fmtDuration(n * 60)}–${fmtDuration(n * 180)} on a laptop CPU`;
});
init();

// =====================================================================
// Evaluation view
// =====================================================================
const evalState = { loaded: false, data: null, topic: null, hover: null };

async function loadEvalRuns() {
  evalState.loaded = true;
  const { runs } = await (await fetch("/api/eval/runs")).json();
  const sel = $("#run-select");
  sel.replaceChildren(...runs.map((r) => h("option", { value: r.id, text: r.label })));
  const saved = store.get("tm-run");
  if (saved && runs.some((r) => r.id === saved)) sel.value = saved;
  if (runs.length) loadRun(sel.value);
  else $("#eval-sub").textContent = "No evaluation results yet. Run: python -m scripts.run_eval";
}
$("#run-select").addEventListener("change", (e) => { store.set("tm-run", e.target.value); loadRun(e.target.value); });
let resizeT;
addEventListener("resize", () => { clearTimeout(resizeT); resizeT = setTimeout(() => evalState.data && $("#view-eval").classList.contains("active") && drawMultiples(), 150); });
$("#metric-select").addEventListener("change", () => { drawHeat(); drawRankings(); });
$("#topic-select").addEventListener("change", (e) => { evalState.topic = e.target.value; drawRankings(); });

async function loadRun(id) {
  const d = await (await fetch(`/api/eval/run/${encodeURIComponent(id)}`)).json();
  evalState.data = d;
  const pairs = d.per_topic.filter((p) => p.system === d.systems[0]).reduce((a, p) => a + p.ranking.length, 0);
  $("#eval-sub").textContent = `${d.topics.length} patient${d.topics.length === 1 ? "" : "s"} · ${pairs} patient–trial pairs · TREC 2021`;
  $("#topic-select").replaceChildren(...d.topics.map((t) => h("option", { value: t, text: `Topic ${t}` })));
  evalState.topic = d.topics[0];
  drawTiles(); drawMultiples(); drawHeat(); drawRankings(); drawSig();
}

function drawTiles() {
  const d = evalState.data;
  const best = (m) => {
    const c = d.systems.filter((x) => d.summary[x][m] != null);
    if (!c.length) return null;
    const top = Math.max(...c.map((x) => d.summary[x][m]));
    return { v: top, who: c.filter((x) => d.summary[x][m] === top) };
  };
  const tiles = [h("div", { class: "tile" }, h("div", { class: "l", text: "Patients evaluated" }),
    h("div", { class: "v", text: String(d.topics.length) }), h("div", { class: "s", text: `${d.systems.length} systems on the same trials` }))];
  for (const m of ["ndcg@10", "p@5", "elig_acc"]) {
    const b = best(m);
    tiles.push(h("div", { class: "tile" }, h("div", { class: "l", text: `Best ${METRIC_NAMES[m].replace("Eligible-vs-excluded accuracy", "eligibility accuracy")}` }),
      h("div", { class: "v", text: b ? fmt(b.v) : "n/a" }), h("div", { class: "s", text: b ? b.who.map((x) => SYSTEM_NAMES[x] || x).join(", ") : "" })));
  }
  $("#tiles").replaceChildren(...tiles);
}

function drawMultiples() {
  const d = evalState.data, box = $("#chart-metrics");
  box.replaceChildren();
  const W = Math.max(300, box.clientWidth || 520), rowH = 28, left = Math.min(168, W * 0.36), right = 48, top = 6;
  for (const m of ["ndcg@10", "p@5", "elig_acc"]) {
    const sys = d.systems;
    const H = top + sys.length * rowH + 22;
    const x = (v) => left + v * (W - left - right);
    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img", "aria-label": `${METRIC_NAMES[m]} by system` });
    for (const t of [0, 0.25, 0.5, 0.75, 1]) {
      svg.append(s("line", { class: t === 0 ? "base" : "gridl", x1: x(t), x2: x(t), y1: top, y2: H - 18 }));
      svg.append(s("text", { class: "tick", x: x(t), y: H - 4, "text-anchor": "middle", text: t === 0 ? "0" : t.toFixed(2).replace(/0$/, "") }));
    }
    sys.forEach((name, i) => {
      const v = d.summary[name][m], y = top + i * rowH, cy = y + rowH / 2;
      const g = s("g", { class: "row", tabindex: "0" });
      g.append(s("text", { class: "lbl", x: left - 10, y: cy + 4, "text-anchor": "end", text: SYSTEM_NAMES[name] || name }));
      if (v == null) {
        g.append(s("text", { class: "na", x: left + 6, y: cy + 4, text: "not applicable" }));
      } else {
        const bw = Math.max(x(v) - left, 2), bh = 16, by = cy - bh / 2;
        // bar with a 4px rounded data-end, square at the baseline
        const r = Math.min(4, bw);
        g.append(s("path", { class: "bar", d: `M${left},${by}h${bw - r}a${r},${r} 0 0 1 ${r},${r}v${bh - 2 * r}a${r},${r} 0 0 1 -${r},${r}h-${bw - r}z` }));
        g.append(s("text", { class: "val", x: x(v) + 6, y: cy + 4, text: fmt(v) }));
      }
      g.append(s("rect", { class: "hit", x: 0, y, width: W, height: rowH }));
      const pts = d.per_topic.filter((p) => p.system === name);
      tipOn(g, () => (v == null ? "n/a" : fmt(v)), () => [
        `${SYSTEM_NAMES[name] || name} · ${METRIC_NAMES[m]}, mean`,
        ...pts.map((p) => `Topic ${p.topic}: ${fmt(p[m])}`)]);
      svg.append(g);
    });
    box.append(h("div", { class: "mchart" }, h("h3", { text: METRIC_NAMES[m] }), svg));
  }
}

function drawHeat() {
  const d = evalState.data, m = $("#metric-select").value;
  $("#heat-sub").textContent = `${METRIC_NAMES[m]} per patient; darker is higher`;
  const get = (sys, t) => d.per_topic.find((p) => p.system === sys && p.topic === t);
  const thead = h("thead", {}, h("tr", {}, h("th", { text: "Topic" }), d.systems.map((x) => h("th", { class: "num", text: SYSTEM_NAMES[x] || x }))));
  const tbody = h("tbody", {}, d.topics.map((t) => h("tr", {}, h("td", { text: t }), d.systems.map((x) => {
    const v = get(x, t)?.[m];
    const td = h("td", { class: "cell", text: fmt(v, 2) });
    if (v != null) {
      td.style.background = `color-mix(in srgb, var(--series-1) ${Math.round(8 + v * 52)}%, var(--surface))`;
      if (v > 0.62) td.style.color = "#fff";
    } else td.style.color = "var(--muted)";
    return td;
  }))));
  const tfoot = h("tfoot", {}, h("tr", {}, h("td", { text: "Mean" }), d.systems.map((x) => h("td", { class: "cell", text: fmt(d.summary[x][m]) }))));
  $("#heat").replaceChildren(h("table", { class: "tbl" }, thead, tbody, tfoot));
}

function drawRankings() {
  const d = evalState.data, t = evalState.topic, m = $("#metric-select").value;
  if (!d || !t) return;
  const QI = { 2: "✓", 1: "✕", 0: "·" };
  const QN = { 2: "eligible", 1: "excluded", 0: "not relevant" };
  const cols = d.systems.map((x) => {
    const p = d.per_topic.find((q) => q.system === x && q.topic === t);
    if (!p) return null;
    const ul = h("ol", { class: "rlist" });
    p.ranking.forEach((r, i) => {
      const li = h("li", { class: `q${r.qrel}${i === 10 ? " cut" : ""}`, tabindex: "0", "data-nct": r.nct_id },
        h("span", { class: "n", text: String(i + 1) }), h("span", { class: "ic", text: QI[r.qrel] }), r.nct_id);
      li.addEventListener("pointerenter", () => $$(`.rlist li[data-nct="${r.nct_id}"]`).forEach((e) => e.classList.add("hl")));
      li.addEventListener("pointerleave", () => $$(".rlist li.hl").forEach((e) => e.classList.remove("hl")));
      tipOn(li, () => r.nct_id, () => [d.titles[r.nct_id] || "", `Ground truth: ${QN[r.qrel]}`,
        r.label ? `Predicted: ${(LABELS[r.label]?.text) || r.label.replace("_", " ")}` : ""].filter(Boolean));
      ul.append(li);
    });
    const name = `${x}_${t}`;
    return h("div", { class: "rcol" },
      h("h4", { text: SYSTEM_NAMES[x] || x }),
      h("div", { class: "rm", text: `${METRIC_NAMES[m].replace("Eligible-vs-excluded accuracy", "accuracy")} ${fmt(p[m])}` }),
      ul,
      d.detail.includes(name) ? h("button", { class: "btn small", text: "Open criteria view", onclick: () => openReplay(name) }) : null);
  }).filter(Boolean);
  $("#rankings").replaceChildren(...cols);
}

function drawSig() {
  const md = evalState.data.significance;
  $("#sig-card").hidden = !md;
  if (!md) return;
  const out = [];
  const lines = md.split("\n");
  for (let i = 0; i < lines.length; i++) {
    const L = lines[i];
    if (L.startsWith("|")) {
      const rows = [];
      while (i < lines.length && lines[i].startsWith("|")) rows.push(lines[i++]);
      i--;
      const cells = (r) => r.split("|").slice(1, -1).map((c) => c.trim());
      const head = cells(rows[0]), body = rows.slice(2).map(cells);
      out.push(h("div", { class: "table-wrap" }, h("table", { class: "tbl" },
        h("thead", {}, h("tr", {}, head.map((c) => h("th", { text: c })))),
        h("tbody", {}, body.map((r) => h("tr", {}, r.map((c) => h("td", { text: c, class: /^[-+]?\d|^\[/.test(c) ? "num" : (c === "no clear difference" ? "" : c.includes("better") ? "sig-verdict yes" : "") }))))))));
    } else if (L.startsWith("# ")) {
      continue;
    } else if (L.startsWith("- ")) {
      out.push(h("div", { class: "small", text: "• " + L.slice(2).replace(/\*\*/g, "") }));
    } else if (L.trim()) {
      out.push(h("p", { class: "small muted", text: L.replace(/\*\*/g, "").replace(/`/g, "") }));
    }
  }
  $("#sig").replaceChildren(...out);
}

// =====================================================================
// About: architecture diagram
// =====================================================================
let archDrawn = false;
function drawArch() {
  if (archDrawn) return;
  archDrawn = true;
  const W = 900, H = 330;
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Architecture: local tier processes patient notes, public tier parses only registry text" });
  svg.append(s("defs", {}, s("marker", { id: "ah", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" }, s("path", { d: "M0,0L10,5L0,10z" }))));
  svg.append(s("rect", { class: "zone-local", x: 10, y: 10, width: 610, height: 310, rx: 14 }));
  svg.append(s("text", { class: "zl", x: 26, y: 34, text: "This laptop · sees patient data · Ollama" }));
  svg.append(s("rect", { class: "zone-public", x: 680, y: 10, width: 210, height: 310, rx: 14 }));
  svg.append(s("text", { class: "zl", x: 696, y: 34, text: "Cloud API · public text only" }));
  svg.append(s("line", { class: "wall", x1: 650, x2: 650, y1: 20, y2: 310 }));
  const box = (x, y, w, t, sub) => {
    svg.append(s("rect", { class: "box", x, y, width: w, height: 52, rx: 9 }));
    svg.append(s("text", { x: x + w / 2, y: y + 22, "text-anchor": "middle", text: t }));
    svg.append(s("text", { class: "sub", x: x + w / 2, y: y + 39, "text-anchor": "middle", text: sub }));
  };
  const arrow = (x1, y1, x2, y2) => svg.append(s("path", { class: "arrow", d: `M${x1},${y1}L${x2},${y2}` }));
  box(30, 60, 130, "Patient note", "sensitive tier");
  box(200, 60, 130, "Profiler", "local LLM");
  box(370, 60, 220, "Pre-filter + hybrid RAG", "BM25 + embeddings, RRF");
  box(30, 170, 130, "Rule checker", "numeric criteria");
  box(200, 170, 130, "Eligibility checker", "local LLM + KB cards");
  box(370, 170, 100, "Critic", "local LLM");
  box(490, 170, 100, "Scorer", "rules");
  box(200, 255, 260, "Ranked trials + evidence", "every decision quotes the note");
  box(700, 60, 170, "Criteria parser", "registry text → criteria");
  box(700, 170, 170, "Concept cards", "written from trial text");
  arrow(160, 86, 198, 86); arrow(330, 86, 368, 86);
  arrow(480, 112, 95, 168);
  arrow(160, 196, 198, 196); arrow(330, 196, 368, 196); arrow(470, 196, 488, 196);
  arrow(540, 222, 460, 268);
  arrow(785, 112, 785, 168);
  arrow(698, 86, 592, 86);
  arrow(698, 196, 592, 210);
  svg.append(s("text", { class: "sub", x: 655, y: 300, "text-anchor": "middle", text: "no patient data crosses" }));
  $("#arch").replaceChildren(svg);
}
