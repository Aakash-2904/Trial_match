"""Patient profiler agent (LOCAL tier). Free-text note -> structured, evidence-grounded profile."""
import re

import llm_local
from config import CANONICAL_FIELDS
from schemas import PatientNote, PatientProfile
from utils import NEGATION, canon, grounded, negates, norm, numbers_in, supports

SYSTEM = f"""You are a clinical information extraction agent. Read the WHOLE patient note, including the
first sentences about the current illness, and return a structured profile as JSON.
Where each item goes:
- "conditions": EVERY disease, diagnosis, syndrome or past medical history item, including the CURRENT main
  diagnosis (for example a cancer found during this admission). Abbreviations like HTN, DM, CHF count.
- "medications": drugs the patient takes or took.
- "procedures": surgeries, biopsies, devices, transplants, dialysis.
- "labs": numeric test results and measurements, using these names when applicable: {", ".join(CANONICAL_FIELDS)}.
- "negatives": ONLY findings the note explicitly denies with words like "no", "denies", "negative for",
  "without". Never write a negative the note does not state.
- "other": only things that fit nowhere else (smoking, alcohol, pregnancy status, social history).
Rules:
- Record only what the note states. Never infer or guess.
- "evidence" is a short span copied exactly from the note that states the item.
- "age" is in years as a number. "sex" is "male", "female" or null."""


def _fallback_age(text: str):
    m = (re.search(r"(\d{1,3})[- ](?:year|yr)s?[- ]old", text, re.I) or re.search(r"\b(\d{1,3})\s*(?:yo|y/o)\b", text, re.I)
         or re.search(r"\b(\d{1,3})\s?[MF]\b", text))          # clinical shorthand: "74M", "62 F"
    return float(m.group(1)) if m else None


def _fallback_sex(text: str):
    t = f" {text.lower()} "
    if any(w in t for w in (" woman ", " female ", " girl ", " she ", " her ")):
        return "female"
    if any(w in t for w in (" man ", " male ", " boy ", " he ", " his ")):
        return "male"
    m = re.search(r"\b\d{1,3}\s?([MF])\b", text)
    return {"M": "male", "F": "female"}[m.group(1)] if m else None


def profile_patient(note: PatientNote) -> PatientProfile:
    prof: PatientProfile = llm_local.chat_json(SYSTEM, f"PATIENT NOTE:\n{note.text}", PatientProfile, "profile")

    return ground_profile(prof, note)


def ground_profile(prof: PatientProfile, note: PatientNote) -> PatientProfile:
    """Hallucination guard: keep facts supported by the note (fuzzy quote match, exact numbers)."""
    def keep_facts(items, negative=False):
        out = []
        for x in items:
            if grounded(x.evidence, note.text) and supports(x.text, x.evidence):
                fact = x
            elif grounded(x.text, note.text):          # quote garbled but the fact itself is in the note
                fact = x.model_copy(update={"evidence": x.text})
            else:
                continue                               # unsupported: drop (hallucination guard)
            if negative and not negates(fact.text, fact.evidence):
                continue                               # a "negative" must be an explicit denial of this fact
            out.append(fact)
        return out

    def keep_labs(items):
        return [l for l in items if f"{l.value:g}" in numbers_in(note.text)
                and (grounded(l.evidence, note.text) or canon(l.name) in canon(note.text))]

    prof = prof.model_copy(deep=True)
    for field in ("conditions", "medications", "procedures", "other"):   # "No prior syncope" is a denial, not a fact
        items = getattr(prof, field)
        prof.negatives += [x for x in items if NEGATION.search(x.text)]
        setattr(prof, field, [x for x in items if not NEGATION.search(x.text)])
    prof.conditions, prof.medications = keep_facts(prof.conditions), keep_facts(prof.medications)
    prof.procedures, prof.negatives = keep_facts(prof.procedures), keep_facts(prof.negatives, negative=True)
    prof.other, prof.labs = keep_facts(prof.other), keep_labs(prof.labs)
    seen = set()
    prof.negatives = [x for x in prof.negatives if not (norm(x.text) in seen or seen.add(norm(x.text)))]

    if prof.age is None:
        prof.age = _fallback_age(note.text)
    if prof.sex is None:
        prof.sex = _fallback_sex(note.text)
    return prof