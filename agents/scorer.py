"""Scorer and ranker (no LLM). Combines criterion outcomes and relevance into a label, score and explanation."""
from typing import List

from agents.groups import group_results
from schemas import CriterionResult, ParsedTrial, Trial, TrialMatch

TIERS = {"likely_eligible": 0, "possibly_eligible": 1, "likely_ineligible": 2}


def rank_key(m: TrialMatch):
    """Eligibility tier first; relevance only orders trials within a tier."""
    return TIERS[m.label], -m.relevance_score


def score(trial: Trial, parsed: ParsedTrial, results: List[CriterionResult], relevance: float) -> TrialMatch:
    results = group_results(results)              # either/or groups count as one criterion
    incl = [r for r in results if r.criterion.kind == "inclusion"]
    excl = [r for r in results if r.criterion.kind == "exclusion"]
    fails = [r for r in results if r.outcome == "not_met"]
    unknown = [r for r in results if r.outcome == "unknown"]

    if fails:
        label, elig = "likely_ineligible", 0.0
    else:
        n = max(len(incl), 1)
        met = sum(r.outcome == "met" for r in incl)
        unk = sum(r.outcome == "unknown" for r in incl)
        elig = (met + 0.5 * unk) / n
        excl_unknown = sum(r.outcome == "unknown" for r in excl) / max(len(excl), 1)
        elig *= 1 - 0.3 * excl_unknown
        label = "likely_eligible" if elig >= 0.75 else "possibly_eligible"

    met_incl = sum(r.outcome == "met" for r in incl)
    parts = [f"{met_incl}/{len(incl)} inclusion criteria met"]
    if fails:
        parts.append("excluded by: " + "; ".join(r.criterion.text[:80] for r in fails[:2]))
    if unknown:
        parts.append(f"{len(unknown)} criteria need review")

    return TrialMatch(
        nct_id=trial.nct_id, title=trial.title, plain_summary=parsed.plain_summary, label=label,
        eligibility_score=round(elig, 3), relevance_score=round(relevance, 3),
        final_score=round(0.5 * relevance + 0.5 * elig, 4),
        explanation=". ".join(parts) + ".", results=results)
