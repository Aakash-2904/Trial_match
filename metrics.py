"""Ranking and classification metrics. Gains follow TREC qrels: 2 eligible, 1 excluded, 0 not relevant."""
import math
from typing import Dict, List


def ndcg_at_k(ranked: List[str], gains: Dict[str, int], k: int = 10) -> float:
    dcg = sum(gains.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted(gains.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(ranked: List[str], gains: Dict[str, int], k: int = 5) -> float:
    top = ranked[:k]
    return sum(gains.get(d, 0) == 2 for d in top) / max(len(top), 1)


def eligible_vs_excluded_accuracy(predicted_eligible: Dict[str, bool], gains: Dict[str, int]) -> float:
    """Among trials the patient is topically relevant to (qrel 1 or 2), how often eligibility is right."""
    pairs = [(predicted_eligible[d], g == 2) for d, g in gains.items() if g in (1, 2) and d in predicted_eligible]
    return sum(p == t for p, t in pairs) / len(pairs) if pairs else float("nan")
