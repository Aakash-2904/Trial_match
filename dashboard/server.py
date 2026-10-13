"""Dashboard server: a local web UI for the matcher and the evaluation results.

  python -m dashboard.server            # then open http://127.0.0.1:8501

Binds to localhost only and makes no outbound requests: the page is static files served from
here, and patient notes go only to the local model, exactly as in app.py. Read-only with respect
to the evaluation: it never writes to results/ and imports the agents without changing them.
"""
import argparse
import asyncio
import csv
import json
import math
import threading
import time
from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from config import DATA_DIR, LOCAL_MODEL, RESULTS_DIR
from metrics import eligible_vs_excluded_accuracy, ndcg_at_k, precision_at_k

STATIC = Path(__file__).resolve().parent / "static"
SYSTEMS = ["bm25", "single_call", "multiagent", "no_critic", "multiagent_rag", "local_parse"]
SNAPSHOTS = {"demo": ("Recruiting trials (demo)", "demo_trials.json"),
             "trec": ("TREC 2021 eval trials", "trials.json")}
EXAMPLE = ("A 62-year-old woman presents with fatigue and a 6-month history of worsening shortness of breath. "
           "She was diagnosed with type 2 diabetes 10 years ago and takes metformin 1000 mg twice daily. "
           "Recent labs show HbA1c 8.4% and eGFR 52 mL/min. Echocardiogram shows an ejection fraction of 35%. "
           "She had a myocardial infarction 3 years ago. She has no history of cancer and is not pregnant.")
DEV_TOPICS = ["4", "5"]          # the dev patients; safe to offer as examples

_snapshots = {}                  # snapshot key -> (trials, parsed, index, retriever, kb), loaded lazily
_run_lock = threading.Lock()     # one screening at a time: the CPU and the model are shared


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_snapshot(key: str):
    """Same loading rules as app.py: API parse overrides local; anything unparsed is parsed locally."""
    if key in _snapshots:
        return _snapshots[key]
    from agents import criteria_parser
    from agents.prefilter import TrialIndex
    from rag.hybrid import TRIAL_STORE, HybridRetriever
    from rag.knowledge import KB_STORE, KnowledgeBase
    from rag.vector_store import VectorStore
    from schemas import ParsedTrial, Trial

    trials = [Trial(**v) for v in _load_json(DATA_DIR / SNAPSHOTS[key][1]).values()]
    parsed = {}
    for name in ("parsed_local.json", "parsed_api.json"):
        p = DATA_DIR / name
        if p.exists():
            parsed.update({k: ParsedTrial(**v) for k, v in _load_json(p).items()})
    for t in trials:
        parsed.setdefault(t.nct_id, criteria_parser.parse_locally(t))
    retriever = kb = None
    if VectorStore.exists(TRIAL_STORE):
        store = VectorStore.load(TRIAL_STORE)
        if all(t.nct_id in store.ids for t in trials):
            retriever = HybridRetriever(trials, parsed, store)
    if VectorStore.exists(KB_STORE):
        kb = KnowledgeBase(VectorStore.load(KB_STORE))
    _snapshots[key] = (trials, parsed, TrialIndex(trials, parsed), retriever, kb)
    return _snapshots[key]


# ---------- screening (streams agent events) ----------
def _screen(note_text: str, snapshot: str, top_k: int, use_critic: bool, use_rag: bool, emit, cancelled):
    """Mirrors supervisor.match_patient, but emits structured events (profile, per-trial nodes, matches)."""
    from agents import scorer
    from schemas import PatientNote
    from supervisor import PATIENT_GRAPH, TRIAL_GRAPH, _describe

    t0 = time.time()
    emit("stage", {"stage": "loading", "msg": "Loading trial snapshot and indexes"})
    trials, parsed, index, retriever, kb = load_snapshot(snapshot)
    rag_note = None
    if use_rag and retriever is None:
        rag_note = "RAG index not built for this snapshot; using BM25 retrieval."
    retriever = retriever if use_rag else None
    kb = kb if use_rag else None
    candidates = (retriever or index).search(note_text, top_k)
    emit("candidates", {"trials": [{"nct_id": t.nct_id, "title": t.title, "conditions": t.conditions[:4]}
                                   for t in candidates], "rag": retriever is not None, "warning": rag_note})

    note = PatientNote(patient_id="dashboard", text=note_text)
    emit("stage", {"stage": "profiler", "msg": "Profiler is reading the note"})
    ps = PATIENT_GRAPH.invoke({"note": note, "trials": candidates, "parsed": parsed, "retriever": retriever})
    emit("profile", {"profile": ps["profile"].model_dump(), "demo_fail": ps["demo_fail"],
                     "elapsed": time.time() - t0})

    matches = []
    for i, t in enumerate(candidates, 1):
        if cancelled.is_set():
            emit("cancelled", {"done": len(matches)})
            return
        emit("trial_start", {"i": i, "n": len(candidates), "nct_id": t.nct_id, "title": t.title})
        ts = time.time()
        init = {"note": note, "profile": ps["profile"], "trial": t, "parsed": parsed[t.nct_id],
                "relevance": ps["relevance"].get(t.nct_id, 0.0), "demo_fail": ps["demo_fail"].get(t.nct_id),
                "use_critic": use_critic, "kb": kb}
        for update in TRIAL_GRAPH.stream(init, stream_mode="updates"):
            for node, out in update.items():
                emit("node", {"i": i, "nct_id": t.nct_id, "node": node, "msg": _describe(node, out)})
                if node == "scorer":
                    matches.append(out["match"])
                    emit("match", {"i": i, "match": out["match"].model_dump(), "seconds": time.time() - ts})
    matches.sort(key=scorer.rank_key)
    emit("done", {"order": [m.nct_id for m in matches], "elapsed": time.time() - t0})


async def match_stream(request: Request):
    body = await request.json()
    note_text = str(body.get("note", "")).strip()
    snapshot = body.get("snapshot", "demo")
    if len(note_text) < 30:
        return JSONResponse({"error": "Please enter a longer patient description."}, status_code=400)
    if snapshot not in SNAPSHOTS or not (DATA_DIR / SNAPSHOTS[snapshot][1]).exists():
        return JSONResponse({"error": "Unknown trial snapshot."}, status_code=400)
    if not _run_lock.acquire(blocking=False):
        return JSONResponse({"error": "A screening is already running. Wait for it or cancel it."}, status_code=409)

    top_k = max(1, min(int(body.get("top_k", 8)), 30))
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    cancelled = threading.Event()

    def emit(kind, data):
        loop.call_soon_threadsafe(queue.put_nowait, (kind, data))

    def worker():
        try:
            _screen(note_text, snapshot, top_k, bool(body.get("critic", True)), bool(body.get("rag", True)),
                    emit, cancelled)
        except Exception as e:                       # surface the failure in the UI instead of hanging
            emit("error", {"msg": f"{type(e).__name__}: {e}"})
        finally:
            _run_lock.release()
            emit("_end", None)

    threading.Thread(target=worker, daemon=True).start()

    async def events():
        try:
            while True:
                try:
                    kind, data = await asyncio.wait_for(queue.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                if kind == "_end":
                    break
                yield f"event: {kind}\ndata: {json.dumps(data)}\n\n"
        finally:
            cancelled.set()                          # client went away: stop after the current trial

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ---------- metadata ----------
async def meta(request: Request):
    from rag.hybrid import TRIAL_STORE
    from rag.knowledge import KB_STORE
    from rag.vector_store import VectorStore

    topics = _load_json(DATA_DIR / "topics2021.json") if (DATA_DIR / "topics2021.json").exists() else {}
    examples = [{"id": "example", "label": "Heart failure + type 2 diabetes (synthetic)", "text": EXAMPLE}]
    examples += [{"id": f"trec-{t}", "label": f"TREC 2021 topic {t} (dev)", "text": topics[t]}
                 for t in DEV_TOPICS if t in topics]
    snaps = []
    for key, (label, fname) in SNAPSHOTS.items():
        p = DATA_DIR / fname
        if p.exists():
            snaps.append({"key": key, "label": label, "count": len(_load_json(p))})
    return JSONResponse({"model": LOCAL_MODEL, "snapshots": snaps, "examples": examples,
                         "rag_index": VectorStore.exists(TRIAL_STORE), "kb_index": VectorStore.exists(KB_STORE),
                         "busy": _run_lock.locked()})


# ---------- evaluation results (read-only) ----------
# (id, label, rankings csv, significance md, whether results/<system>_<topic>.json belong to this run)
# Newest first; only runs whose csv exists are listed. The per-trial JSON files are overwritten by each
# run, so only the latest run of a topic can link to them.
RUNS = [
    ("heldout", "Held-out test set", "rankings_heldout.csv", "significance_heldout.md", True),
    ("dev_after", "Dev set, after fixes", "rankings_dev_after.csv", None, True),
    ("dev_before", "Dev set, before fixes", "rankings_batch_B_all5_p2.csv", "significance.md", False),
]


def _per_topic(csv_path: Path):
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    by = {}
    for r in rows:
        by.setdefault((r["system"], r["topic"]), []).append(r)
    out = []
    for (sysname, topic), rs in by.items():
        rs.sort(key=lambda r: int(r["rank"]))
        ranked = [r["nct_id"] for r in rs]
        gains = {r["nct_id"]: int(r["qrel"]) for r in rs}
        labels = {r["nct_id"]: r["label"] for r in rs}
        pred = None
        if sysname == "single_call":
            pred = {n: l == "eligible" for n, l in labels.items()}
        elif sysname != "bm25":
            pred = {n: l != "likely_ineligible" for n, l in labels.items()}
        acc = eligible_vs_excluded_accuracy(pred, gains) if pred else None
        out.append({"system": sysname, "topic": topic,
                    "ndcg@10": ndcg_at_k(ranked, gains, 10), "p@5": precision_at_k(ranked, gains, 5),
                    "elig_acc": None if acc is None or math.isnan(acc) else acc,
                    "ranking": [{"nct_id": r["nct_id"], "qrel": int(r["qrel"]), "label": r["label"]} for r in rs]})
    return out


async def eval_runs(request: Request):
    runs = [{"id": i, "label": l, "file": f} for i, l, f, *_ in RUNS if (RESULTS_DIR / f).exists()]
    return JSONResponse({"runs": runs})


async def eval_run(request: Request):
    rid = request.path_params["run"]
    entry = next((r for r in RUNS if r[0] == rid), None)
    if not entry or not (RESULTS_DIR / entry[2]).exists():
        return JSONResponse({"error": "unknown run"}, status_code=404)
    per = _per_topic(RESULTS_DIR / entry[2])
    systems = [s for s in SYSTEMS if any(p["system"] == s for p in per)]
    topics = sorted({p["topic"] for p in per}, key=int)
    summary = {}
    for s in systems:
        vals = [p for p in per if p["system"] == s]
        summary[s] = {}
        for m in ("ndcg@10", "p@5", "elig_acc"):
            xs = [p[m] for p in vals if p[m] is not None]
            summary[s][m] = round(sum(xs) / len(xs), 4) if xs else None
    trials = _load_json(DATA_DIR / "trials.json") if (DATA_DIR / "trials.json").exists() else {}
    titles = {p_["nct_id"]: trials.get(p_["nct_id"], {}).get("title", "") for p in per for p_ in p["ranking"]}
    detail = sorted(f.stem for f in RESULTS_DIR.glob("*_*.json")
                    if f.stem.rsplit("_", 1)[0] in SYSTEMS and f.stem.rsplit("_", 1)[1].isdigit()) if entry[4] else []
    sig = RESULTS_DIR / entry[3] if entry[3] else None
    for p in per:
        for m in ("ndcg@10", "p@5", "elig_acc"):
            p[m] = None if p[m] is None else round(p[m], 4)
    return JSONResponse({"run": rid, "systems": systems, "topics": topics, "summary": summary,
                         "per_topic": per, "titles": titles, "detail": detail,
                         "significance": sig.read_text(encoding="utf-8") if sig and sig.exists() else None})


async def eval_detail(request: Request):
    name = request.path_params["name"]
    sysname, _, topic = name.rpartition("_")
    if sysname not in SYSTEMS or not topic.isdigit():
        return JSONResponse({"error": "bad name"}, status_code=400)
    p = RESULTS_DIR / f"{sysname}_{topic}.json"
    if not p.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    topics = _load_json(DATA_DIR / "topics2021.json")
    return JSONResponse({"system": sysname, "topic": topic, "note": topics.get(topic, ""), "matches": _load_json(p)})


async def index(request: Request):
    return FileResponse(STATIC / "index.html")


app = Starlette(routes=[
    Route("/", index),
    Route("/api/meta", meta),
    Route("/api/match", match_stream, methods=["POST"]),
    Route("/api/eval/runs", eval_runs),
    Route("/api/eval/run/{run}", eval_run),
    Route("/api/eval/detail/{name}", eval_detail),
    Mount("/static", StaticFiles(directory=STATIC), name="static"),
])


if __name__ == "__main__":
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8501)
    args = ap.parse_args()
    print(f"TrialMatch dashboard: http://127.0.0.1:{args.port}")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
