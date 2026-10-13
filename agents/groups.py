"""Either/or criterion groups ("at least one of the following:", "any of the following:", ...), scored as one.

Two parse shapes are recognised:
- rule/API parses put the heading in front of each option: "Presence of any of the following: <option>"
  (consecutive criteria with the same kind and the same option-introducing prefix);
- the local parser keeps the heading as its own criterion ending with ":"; the options are the following
  criteria of the same kind, up to the next heading that ends with ":".

A group of n options that needs k of them (k=1 unless the heading says "at least two" etc.):
the group statement is true if >= k options are true, false only if fewer than k could still be true, else unknown.
"""
import re
from typing import List, Optional

from schemas import Criterion, CriterionResult
from utils import outcome_for

_NUM = r"(\d+|one|two|three|four|five)"
_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
_TIME = r"(?!\s*(?:\w+\s)?(?:years?|months?|weeks?|days?|hours?|times?|doses?|cycles?|lines?|mg|%))"
OPTIONS = re.compile(rf"\bat least {_NUM}\b{_TIME}|\bany (?:one )?of the following\b|\bone or more of\b|"
                     rf"\bone of the following\b|\bone additional\b|\beither\b", re.I)
_K = re.compile(rf"\bat least {_NUM}\b{_TIME}", re.I)


def needed(heading: str) -> int:
    m = _K.search(heading)
    if not m:
        return 1
    g = m.group(1).lower()
    return int(g) if g.isdigit() else _WORDS[g]


def _prefix(text: str) -> Optional[str]:
    if ": " not in text:
        return None
    head = text.split(": ", 1)[0]
    return head if OPTIONS.search(head) else None


def _is_heading(text: str) -> bool:
    return text.rstrip().endswith(":") and bool(OPTIONS.search(text))


def combine(heading: str, members: List[CriterionResult]) -> CriterionResult:
    k, n = needed(heading), len(members)
    yes = [m for m in members if m.statement_true == "yes"]
    no = [m for m in members if m.statement_true == "no"]
    unknown = n - len(yes) - len(no)
    if len(yes) >= k:
        truth, deciding = "yes", yes
    elif k > n:                                    # options lost in parsing: never conclude "no"
        truth, deciding = "unknown", []
    elif len(yes) + unknown < k:
        truth, deciding = "no", no
    else:
        truth, deciding = "unknown", []
    first, last = members[0].criterion, members[-1].criterion
    options = [m.criterion.text.split(": ", 1)[1] if m.criterion.text.startswith(heading + ": ") else m.criterion.text
               for m in members]
    crit = Criterion(id=f"{first.id}-{last.id}" if n > 1 else first.id, kind=first.kind,
                     text=f"{heading.rstrip(':')}: " + " | ".join(options), category=first.category)
    evidence = "; ".join(m.evidence for m in deciding if m.evidence)[:500] or None
    return CriterionResult(criterion=crit, statement_true=truth, outcome=outcome_for(first.kind, truth),
                           evidence=evidence, source=(deciding or members)[0].source,
                           note=f"either/or group: {len(yes)} of {n} options true, {len(no)} false, {k} needed")


def group_results(results: List[CriterionResult]) -> List[CriterionResult]:
    out, i = [], 0
    while i < len(results):
        r = results[i]
        kind, text = r.criterion.kind, r.criterion.text
        prefix = _prefix(text)
        if prefix:                                 # shape 1: shared "heading: option" prefix
            j = i + 1
            while j < len(results) and results[j].criterion.kind == kind and _prefix(results[j].criterion.text) == prefix:
                j += 1
            out.append(combine(prefix, results[i:j]))
            i = j
            continue
        if _is_heading(text):                      # shape 2: heading criterion, options follow
            j = i + 1
            while j < len(results) and results[j].criterion.kind == kind and not results[j].criterion.text.rstrip().endswith(":"):
                j += 1
            if j > i + 1:
                out.append(combine(text.rstrip().rstrip(":"), results[i + 1:j]))
                i = j
                continue
        out.append(r)
        i += 1
    return out
