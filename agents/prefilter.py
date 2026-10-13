"""Pre-filter agent (no LLM). BM25 relevance + hard age/sex checks from structured registry fields."""
import re
from typing import Dict, List, Optional

from rank_bm25 import BM25Okapi

from schemas import ParsedTrial, PatientProfile, Trial

_STOP = set("a an and are as at be by for from has have he her his in is it of on or she that the "
            "this to was were with who patient patients year old history presents".split())


def tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if t not in _STOP and len(t) > 1]


def _doc(trial: Trial, parsed: Optional[ParsedTrial]) -> str:
    extra = " ".join(parsed.search_terms) if parsed else ""
    return " ".join([trial.title, " ".join(trial.conditions), extra, extra, trial.summary, trial.eligibility_text])


class TrialIndex:
    def __init__(self, trials: List[Trial], parsed: Dict[str, ParsedTrial]):
        self.trials = trials
        self.bm25 = BM25Okapi([tokenize(_doc(t, parsed.get(t.nct_id))) for t in trials])

    def scores(self, query: str) -> Dict[str, float]:
        raw = self.bm25.get_scores(tokenize(query))
        return {t.nct_id: float(s) for t, s in zip(self.trials, raw)}

    def search(self, query: str, k: int) -> List[Trial]:
        s = self.scores(query)
        return sorted(self.trials, key=lambda t: s[t.nct_id], reverse=True)[:k]


def relevance_scores(query: str, trials: List[Trial], parsed: Dict[str, ParsedTrial]) -> Dict[str, float]:
    """BM25 among the candidate set, normalised to 0..1."""
    s = TrialIndex(trials, parsed).scores(query)
    top = max(s.values(), default=0.0)
    return {k: (max(v, 0.0) / top if top > 0 else 0.0) for k, v in s.items()}


def demographic_fail(profile: PatientProfile, trial: Trial) -> Optional[str]:
    if profile.age is not None:
        if trial.min_age is not None and profile.age < trial.min_age:
            return f"Trial minimum age is {trial.min_age:g}; patient is {profile.age:g}"
        if trial.max_age is not None and profile.age > trial.max_age:
            return f"Trial maximum age is {trial.max_age:g}; patient is {profile.age:g}"
    if profile.sex and trial.sex in ("MALE", "FEMALE") and profile.sex.upper() != trial.sex:
        return f"Trial enrolls {trial.sex.lower()} participants only"
    return None
