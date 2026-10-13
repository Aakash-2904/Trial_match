"""Criteria parser agent (PUBLIC tier). Turns eligibility text into atomic, checkable criteria.

API mode uses the cloud model on public trial text. Local mode is a rule-based
fallback (and the ablation that shows how much the API parser contributes).
"""
import re

import llm_public
from config import CANONICAL_FIELDS
from schemas import Criterion, ParsedTrial, Trial, TrialParseOutput

INSTRUCTION = f"""You convert clinical trial eligibility criteria into atomic, checkable criteria.
Rules:
- Split compound criteria into separate atomic criteria. Keep each "text" faithful to the original wording.
- "kind" is "inclusion" or "exclusion" according to the section the criterion appears in.
- category is one of: age, sex, lab, condition, medication, procedure, temporal, other.
- For numeric thresholds (age, labs, scores, vitals) fill "check" with one of these canonical field
  names when applicable: {", ".join(CANONICAL_FIELDS)}. The check must express the criterion text
  literally, e.g. exclusion "eGFR < 30 mL/min" -> field "eGFR", op "<", value 30, unit "mL/min".
  For a range like "age 18-75" create two criteria (age >= 18 and age <= 75).
- If there is no numeric threshold, set "check" to null.
- Use ids I1, I2, ... for inclusion and E1, E2, ... for exclusion.
- "search_terms": 5 to 15 synonyms or abbreviations for the trial's conditions and interventions.
- "plain_summary": two sentences a patient could understand.
- "concepts": up to 8 cards for medical terms in the criteria that need background knowledge to judge,
  such as drug classes, disease groups or staging terms. Each card: "term", a one-sentence "definition",
  and "includes" listing common members (e.g. anti-PD-1 therapy includes pembrolizumab, nivolumab)."""

_BULLET = re.compile(r"^\s*(?:[-*\u2022\u00b7]|\d+[.)]|[a-z][.)])\s*", re.I)
_HEADING = re.compile(r"^(key\s+)?(inclusion|exclusion)\s+criteria\s*:?$", re.I)


def _sections(text: str):
    low = text.lower()
    i, e = low.find("inclusion criteria"), low.find("exclusion criteria")
    if e == -1:
        return text, ""
    start = i if 0 <= i < e else 0
    return text[start:e], text[e:]


def _items(block: str):
    out = []
    for line in block.splitlines():
        s = _BULLET.sub("", line).strip()
        if len(s) < 4 or _HEADING.match(s):
            continue
        out.append(s)
    return out


def parse_locally(trial: Trial) -> ParsedTrial:
    incl, excl = _sections(trial.eligibility_text or "")
    criteria = [Criterion(id=f"I{n}", kind="inclusion", text=t) for n, t in enumerate(_items(incl), 1)]
    criteria += [Criterion(id=f"E{n}", kind="exclusion", text=t) for n, t in enumerate(_items(excl), 1)]
    return ParsedTrial(nct_id=trial.nct_id, criteria=criteria,
                       plain_summary=(trial.summary or "")[:300], parser="local")


def parse_with_api(trial: Trial) -> ParsedTrial:
    out: TrialParseOutput = llm_public.generate_json(trial.public(), INSTRUCTION, TrialParseOutput, "parse_api")
    return ParsedTrial(nct_id=trial.nct_id, criteria=out.criteria, search_terms=out.search_terms,
                       plain_summary=out.plain_summary, concepts=out.concepts, parser="api")


def parse(trial: Trial, mode: str = "auto") -> ParsedTrial:
    if mode == "local" or (mode == "auto" and not llm_public.available()):
        return parse_locally(trial)
    try:
        return parse_with_api(trial)
    except Exception as e:
        print(f"  [parser] API failed for {trial.nct_id} ({e}); using local parser")
        return parse_locally(trial)
