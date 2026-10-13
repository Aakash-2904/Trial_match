"""Paired significance tests on per-patient nDCG@10 for the key system comparisons.

  python -m scripts.significance                                   # results/rankings.csv + rankings_batch_*.csv
  python -m scripts.significance results/rankings_batch_A_core_p3.csv

Each comparison uses only the patients both systems were run on. Reports the mean difference, a paired
bootstrap 95% CI (resampling patients), an exact two-sided Wilcoxon signed-rank p-value and win/tie/loss.
Writes results/significance.md. No scipy needed: with this few patients the exact test is instant.
"""
import csv
import itertools
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from config import RESULTS_DIR, SEED
from metrics import ndcg_at_k

PAIRS = [("multiagent", "single_call"), ("multiagent", "no_critic"),
         ("multiagent_rag", "multiagent"), ("multiagent", "bm25")]
N_BOOT = 10000
ALPHA = 0.05
TIE = 1e-9


def load_rankings(paths):
    """{(system, topic): [nct ids in rank order]} and {topic: {nct: qrel}}, merged over several files."""
    ranked, gains = {}, defaultdict(dict)
    for path in paths:
        rows = defaultdict(list)
        for r in csv.DictReader(open(path, encoding="utf-8")):
            rows[(r["system"], r["topic"])].append((int(r["rank"]), r["nct_id"]))
            gains[r["topic"]][r["nct_id"]] = int(r["qrel"])
        for key, lst in rows.items():
            order = [n for _, n in sorted(lst)]
            if key in ranked and ranked[key] != order:
                print(f"warning: {key} differs between files, keeping the first one ({path} ignored)")
            ranked.setdefault(key, order)
    return ranked, gains


def wilcoxon_exact(diffs):
    """Two-sided exact Wilcoxon signed-rank p-value. Zero differences are dropped; tied |d| get average ranks."""
    d = [x for x in diffs if abs(x) > TIE]
    if not d:
        return 1.0
    order = sorted(range(len(d)), key=lambda i: abs(d[i]))
    ranks = [0.0] * len(d)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and abs(abs(d[order[j + 1]]) - abs(d[order[i]])) <= TIE:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    w_obs = sum(r for r, x in zip(ranks, d) if x > 0)
    center = sum(ranks) / 2
    extreme = total = 0
    for signs in itertools.product((0, 1), repeat=len(d)):
        w = sum(r for r, s in zip(ranks, signs) if s)
        extreme += abs(w - center) >= abs(w_obs - center) - TIE
        total += 1
    return extreme / total


def bootstrap_ci(diffs, rng):
    means = sorted(statistics.fmean(rng.choices(diffs, k=len(diffs))) for _ in range(N_BOOT))
    return means[int(N_BOOT * ALPHA / 2)], means[int(N_BOOT * (1 - ALPHA / 2)) - 1]


def compare(a, b, ranked, gains, rng):
    topics = sorted({t for s, t in ranked if s == a} & {t for s, t in ranked if s == b}, key=int)
    if not topics:
        return None
    diffs = [ndcg_at_k(ranked[(a, t)], gains[t], 10) - ndcg_at_k(ranked[(b, t)], gains[t], 10) for t in topics]
    lo, hi = bootstrap_ci(diffs, rng)
    p = wilcoxon_exact(diffs)
    wins, ties = sum(x > TIE for x in diffs), sum(abs(x) <= TIE for x in diffs)
    if p < ALPHA and (lo > 0 or hi < 0):
        verdict = f"{a} better" if lo > 0 else f"{b} better"
    else:
        verdict = "no clear difference"
    return {"a": a, "b": b, "topics": topics, "diffs": diffs, "mean": statistics.fmean(diffs), "lo": lo, "hi": hi,
            "p": p, "wins": wins, "ties": ties, "losses": len(diffs) - wins - ties, "verdict": verdict}


def main():
    paths = [Path(p) for p in sys.argv[1:]] or \
        [p for p in [RESULTS_DIR / "rankings.csv", *sorted(RESULTS_DIR.glob("rankings_batch_*.csv"))] if p.exists()]
    ranked, gains = load_rankings(paths)
    rng = random.Random(SEED)

    lines = ["# Paired significance tests (per-patient nDCG@10)", "",
             f"Inputs: {', '.join(p.name for p in paths)}. Difference = first system minus second. "
             f"95% CI: paired bootstrap over patients ({N_BOOT} resamples, seed {SEED}). "
             "p: exact two-sided Wilcoxon signed-rank (zero differences dropped). "
             f"A difference counts only if p < {ALPHA} and the CI excludes 0; otherwise \"no clear difference\".", "",
             "| Comparison | Patients | Mean diff | 95% CI | Wilcoxon p | Win/tie/loss | Verdict |",
             "|---|---|---|---|---|---|---|"]
    details, min_p_note = [], set()
    for a, b in PAIRS:
        r = compare(a, b, ranked, gains, rng)
        if r is None:
            lines.append(f"| {a} vs {b} | 0 | n/a | n/a | n/a | n/a | not run on a common patient |")
            continue
        n = len(r["topics"])
        lines.append(f"| {a} vs {b} | {n} | {r['mean']:+.3f} | [{r['lo']:+.3f}, {r['hi']:+.3f}] | {r['p']:.3f} | "
                     f"{r['wins']}/{r['ties']}/{r['losses']} | {r['verdict']} |")
        details.append(f"- {a} vs {b}: " + ", ".join(f"patient {t} {d:+.3f}" for t, d in zip(r["topics"], r["diffs"])))
        min_p_note.add(n)
    lines += ["", "Per-patient differences:", *details, ""]
    for n in sorted(min_p_note):
        lines.append(f"With {n} patients the smallest possible two-sided Wilcoxon p is {2 / 2 ** n:.3f}, so "
                     f"{'no' if 2 / 2 ** n >= ALPHA else 'only a consistent'} difference can reach p < {ALPHA}. "
                     "The bootstrap CI with this few patients only spans the observed differences, not real uncertainty.")
    out = "\n".join(lines) + "\n"
    (RESULTS_DIR / "significance.md").write_text(out, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
