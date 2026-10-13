"""Critic agent (LOCAL tier). Re-reads the ORIGINAL note and challenges every LLM decision that excludes the patient."""
from typing import List

import llm_local
from schemas import CriterionResult, CriticOutput, PatientNote
from utils import grounded, outcome_for

SYSTEM = """You are a skeptical clinical reviewer. Another agent concluded that each criterion below makes
the patient INELIGIBLE. Re-read the ORIGINAL NOTE and check each conclusion.
- verdict "confirm": the note clearly supports the conclusion.
- verdict "overturn": the conclusion is wrong. Set "corrected" to the value of the criterion statement that
  the note actually supports: "yes" or "no" whenever the note settles it. Example: the note says the heart
  attack was 3 years ago, so "myocardial infarction within the last 3 months" is "no", not "unknown".
  Use "unknown" only when the note says nothing that answers the statement.
- "evidence": copy the exact words from the ORIGINAL NOTE that your decision rests on, or null.
- "reason": one short sentence.
Return one decision per id."""


def needs_review(results: List[CriterionResult]) -> List[CriterionResult]:
    return [r for r in results if r.outcome == "not_met" and r.source == "llm"]


def run(results: List[CriterionResult], note: PatientNote) -> List[CriterionResult]:
    targets = needs_review(results)
    if not targets:
        return results
    lines = [f"{r.criterion.id} [{r.criterion.kind}] criterion: {r.criterion.text} | "
             f"concluded statement_true={r.statement_true} | evidence: {r.evidence or 'none'}" for r in targets]
    # Note goes FIRST for prefix caching across this patient's trials
    user = f"ORIGINAL NOTE:\n{note.text}\n\nCONCLUSIONS TO REVIEW:\n" + "\n".join(lines)
    out: CriticOutput = llm_local.chat_json(SYSTEM, user, CriticOutput, "critic")
    by_id = {d.id.strip().upper(): d for d in out.decisions}

    revised = []
    for r in results:
        d = by_id.get(r.criterion.id.upper()) if r in targets else None
        if d and d.verdict == "overturn" and d.corrected != r.statement_true:
            corrected, ev, reason = d.corrected, d.evidence, d.reason
            if corrected != "unknown" and not grounded(ev, note.text):   # a yes/no needs a quote from the note
                corrected, reason = "unknown", f"{d.reason} (critic evidence not found in note)"
            r = r.model_copy(update={"statement_true": corrected, "outcome": outcome_for(r.criterion.kind, corrected),
                                     "evidence": ev if corrected != "unknown" else None,
                                     "source": "critic", "note": reason})
        elif d:
            r = r.model_copy(update={"note": f"critic confirmed: {d.reason}"})
        revised.append(r)
    return revised
