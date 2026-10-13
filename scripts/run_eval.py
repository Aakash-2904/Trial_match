"""Step 3: run all systems on the TREC eval set and compute metrics. Fully local: no API calls here.

  python -m scripts.run_eval                     # all systems
  python -m scripts.run_eval --systems bm25 multiagent no_critic
  python -m scripts.run_eval --patients 3        # quick smoke test

Every LLM call is cached, so you can stop and restart at any time. The no_critic ablation
reuses the cached checker outputs, so it costs almost nothing extra.
"""
import argparse
import csv
import json
import statistics
import time

from baselines import bm25_scores, single_call
from config import DATA_DIR, RESULTS_DIR
from metrics import eligible_vs_excluded_accuracy, ndcg_at_k, precision_at_k
from schemas import ParsedTrial, PatientNote, Trial
from rag.hybrid import TRIAL_STORE, HybridRetriever
from rag.knowledge import KB_STORE, KnowledgeBase
from rag.vector_store import VectorStore
from supervisor import match_patient

ALL = ["bm25", "single_call", "multiagent", "no_critic", "local_parse", "multiagent_rag"]


def load(name):
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


def select_topics(eval_set, exclude=(), patients=None):
    """Eval-set order minus excluded topic IDs, optionally cut to the first `patients`."""
    unknown = set(exclude) - set(eval_set)
    if unknown:
        raise SystemExit(f"--exclude: not in the eval set: {sorted(unknown)}")
    ids = [t for t in eval_set if t not in set(exclude)]
    return ids[:patients] if patients else ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=ALL, choices=ALL)
    ap.add_argument("--patients", type=int, default=None)
    ap.add_argument("--exclude", nargs="+", default=[], help="topic IDs to skip (e.g. the dev set: 4 5)")
    args = ap.parse_args()

    topics, qrels, eval_set = load("topics2021.json"), load("qrels2021.json"), load("eval_set.json")
    trials = {k: Trial(**v) for k, v in load("trials.json").items()}
    p_local = {k: ParsedTrial(**v) for k, v in load("parsed_local.json").items()}
    api_path = DATA_DIR / "parsed_api.json"
    p_api = {k: ParsedTrial(**v) for k, v in load("parsed_api.json").items()} if api_path.exists() else p_local
    if not api_path.exists():
        print("note: parsed_api.json not found, 'multiagent' will use the local parser")

    # multiagent first: no_critic then reuses its cached checker calls
    systems = [s for s in ["bm25", "multiagent", "no_critic", "multiagent_rag", "local_parse", "single_call"]
               if s in args.systems]
    retriever = kb = None
    if "multiagent_rag" in systems:
        if not (VectorStore.exists(TRIAL_STORE) and VectorStore.exists(KB_STORE)):
            raise SystemExit("RAG indexes missing: run python -m scripts.build_index first")
        retriever = HybridRetriever(list(trials.values()), p_api, VectorStore.load(TRIAL_STORE))
        kb = KnowledgeBase(VectorStore.load(KB_STORE))
    topic_ids = select_topics(eval_set, args.exclude, args.patients)
    RESULTS_DIR.mkdir(exist_ok=True)

    per_topic = {s: [] for s in systems}
    rows = []
    for ti, tid in enumerate(topic_ids, 1):
        note = PatientNote(patient_id=f"trec2021-{tid}", text=topics[tid])
        cands = [trials[n] for n in eval_set[tid]]
        gains = {n: qrels[tid].get(n, 0) for n in eval_set[tid]}
        print(f"\n=== patient {tid} ({ti}/{len(topic_ids)}): {len(cands)} trials ===")

        for sysname in systems:
            t0 = time.time()
            if sysname == "bm25":
                s = bm25_scores(note, cands, p_api)
                ranked = sorted(s, key=s.get, reverse=True)
                pred, labels = None, {n: "" for n in ranked}
            elif sysname == "single_call":
                outs = {t.nct_id: single_call(note, t) for t in cands}
                ranked = sorted(outs, key=lambda n: outs[n].score, reverse=True)
                pred = {n: o.label == "eligible" for n, o in outs.items()}
                labels = {n: o.label for n, o in outs.items()}
            else:
                parsed = p_local if sysname == "local_parse" else p_api
                rag_on = sysname == "multiagent_rag"
                matches = match_patient(note, cands, parsed, use_critic=(sysname != "no_critic"),
                                        retriever=retriever if rag_on else None, kb=kb if rag_on else None,
                                        on_event=lambda a, m: print("   ", m) if a in ("profiler", "scorer") else None)
                ranked = [m.nct_id for m in matches]
                pred = {m.nct_id: m.label != "likely_ineligible" for m in matches}
                labels = {m.nct_id: m.label for m in matches}
                (RESULTS_DIR / f"{sysname}_{tid}.json").write_text(
                    json.dumps([m.model_dump() for m in matches], indent=1), encoding="utf-8")

            res = {"ndcg@10": ndcg_at_k(ranked, gains, 10), "p@5": precision_at_k(ranked, gains, 5),
                   "elig_acc": eligible_vs_excluded_accuracy(pred, gains) if pred else float("nan")}
            per_topic[sysname].append(res)
            for rank, n in enumerate(ranked, 1):
                rows.append({"system": sysname, "topic": tid, "nct_id": n, "rank": rank,
                             "qrel": gains.get(n, 0), "label": labels.get(n, "")})
            print(f"  {sysname:12s} nDCG@10={res['ndcg@10']:.3f}  P@5={res['p@5']:.2f}  "
                  f"elig_acc={res['elig_acc']:.2f}  ({time.time() - t0:.0f}s)")

    summary = {}
    for s, lst in per_topic.items():
        summary[s] = {m: round(statistics.fmean(v for v in (r[m] for r in lst) if v == v), 4)
                      if any(r[m] == r[m] for r in lst) else None for m in ("ndcg@10", "p@5", "elig_acc")}
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    with open(RESULTS_DIR / "rankings.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["system", "topic", "nct_id", "rank", "qrel", "label"])
        w.writeheader()
        w.writerows(rows)

    lines = ["| System | nDCG@10 | P@5 | Eligible-vs-excluded acc. |", "|---|---|---|---|"]
    for s, m in summary.items():
        fmt = lambda x: "n/a" if x is None else f"{x:.3f}"
        lines.append(f"| {s} | {fmt(m['ndcg@10'])} | {fmt(m['p@5'])} | {fmt(m['elig_acc'])} |")
    (RESULTS_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
