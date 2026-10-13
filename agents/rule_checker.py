"""Rule checker agent (no LLM). Evaluates numeric criteria directly against the profile."""
import operator
from typing import List, Optional, Tuple

from schemas import Criterion, CriterionResult, PatientProfile
from utils import canon, outcome_for

OPS = {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge, "==": operator.eq}


def _result(c: Criterion, truth: str, evidence: Optional[str], note: Optional[str] = None) -> CriterionResult:
    return CriterionResult(criterion=c, statement_true=truth, outcome=outcome_for(c.kind, truth),
                           evidence=evidence, source="rule", note=note)


def check_numeric(c: Criterion, profile: PatientProfile) -> Optional[CriterionResult]:
    if c.check is None:
        return None
    field = canon(c.check.field)
    if field == "age":
        if profile.age is None:
            return _result(c, "unknown", None, "age not stated in note")
        return _result(c, "yes" if OPS[c.check.op](profile.age, c.check.value) else "no",
                       None, f"patient age {profile.age:g}")
    lab = next((l for l in profile.labs if canon(l.name) == field), None)
    if lab is None:
        return _result(c, "unknown", None, f"{c.check.field} not reported in note")
    if c.check.unit and lab.unit and canon(c.check.unit) != canon(lab.unit):
        return None                      # unit mismatch: let the LLM judge it
    truth = "yes" if OPS[c.check.op](lab.value, c.check.value) else "no"
    return _result(c, truth, lab.evidence)


def run(criteria: List[Criterion], profile: PatientProfile) -> Tuple[List[CriterionResult], List[Criterion]]:
    results, pending = [], []
    for c in criteria:
        r = check_numeric(c, profile)
        (results.append(r) if r else pending.append(c))
    return results, pending


def demographic_result(reason: str) -> CriterionResult:
    c = Criterion(id="D1", kind="inclusion", text="Age and sex requirements", category="age")
    return CriterionResult(criterion=c, statement_true="no", outcome="not_met", source="rule", note=reason)
