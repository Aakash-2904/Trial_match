"""Eligibility checker agent (LOCAL tier). Judges the fuzzy criteria the rule checker could not."""
import json
from typing import List

import llm_local
from config import MAX_FUZZY_CRITERIA
from schemas import CheckerOutput, Criterion, CriterionResult, PatientNote, PatientProfile
from utils import grounded, outcome_for

SYSTEM = """You are a clinical trial eligibility screening agent. For each criterion, decide whether the
criterion STATEMENT is true for this patient.
The PATIENT NOTE is the source of truth. The PROFILE is a structured summary of it that may be incomplete.
- "yes": the note clearly shows the statement is true.
- "no": the note clearly shows the statement is false (explicit denials count).
- "unknown": the note does not say. Prefer "unknown" over guessing.
- "evidence": copy the exact words from the PATIENT NOTE that support your answer, or null.
Judge each statement literally. Do not consider whether it is an inclusion or exclusion criterion.
MEDICAL KNOWLEDGE, if given, helps you interpret terms (for example which drugs belong to a class).
Take facts about the patient ONLY from the note and profile, never from the knowledge section.
Return one judgment per criterion id."""


def compact_profile(profile: PatientProfile) -> str:
    d = {k: v for k, v in profile.model_dump(exclude_none=True).items() if v not in ([], None)}
    return json.dumps(d, ensure_ascii=False, separators=(",", ":"))


def run(pending: List[Criterion], profile: PatientProfile, note: PatientNote, kb=None) -> List[CriterionResult]:
    checked, overflow = pending[:MAX_FUZZY_CRITERIA], pending[MAX_FUZZY_CRITERIA:]
    results: List[CriterionResult] = []
    if checked:
        knowledge = ""
        if kb is not None:                     # knowledge RAG: retrieved cards go AFTER the profile
            cards = kb.for_criteria(checked)
            if cards:
                from rag.knowledge import card_text
                knowledge = ("\n\nMEDICAL KNOWLEDGE (background only: general definitions, NOT facts about this "
                             "patient; never quote it as evidence):\n" + "\n".join(f"- {card_text(c)}" for c in cards))
        # Note + profile go FIRST so Ollama can reuse the cached prefix across this patient's trials
        user = f"PATIENT NOTE:\n{note.text}\n\nPROFILE:\n{compact_profile(profile)}{knowledge}\n\nCRITERIA:\n" + \
               "\n".join(f"{c.id}: {c.text}" for c in checked)
        out: CheckerOutput = llm_local.chat_json(SYSTEM, user, CheckerOutput, "check")
        by_id = {j.id.strip().upper(): j for j in out.judgments}
        for c in checked:
            j = by_id.get(c.id.upper())
            truth, ev, src, note_txt = (j.statement_true, j.evidence, "llm", None) if j else ("unknown", None, "llm", "no judgment returned")
            if truth != "unknown" and not grounded(ev, note.text):   # met and not_met both need a quote from the note
                truth, src, note_txt = "unknown", "grounding", "evidence not found in note" if ev else "no evidence quoted"
            results.append(CriterionResult(criterion=c, statement_true=truth, outcome=outcome_for(c.kind, truth),
                                           evidence=ev, source=src, note=note_txt))
    for c in overflow:
        results.append(CriterionResult(criterion=c, statement_true="unknown", outcome="unknown",
                                       source="cap", note="not checked (criteria limit)"))
    return results