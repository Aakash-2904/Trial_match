"""Baselines for the evaluation: BM25 only, and a single local LLM call per patient-trial pair."""
from typing import Dict, List

import llm_local
from agents.prefilter import TrialIndex
from schemas import ParsedTrial, PatientNote, SingleCallOutput, Trial

SINGLE_SYSTEM = """You are a clinical trial eligibility assistant. Read the patient note and the trial.
Return "label": "eligible" if the patient likely meets the criteria, "excluded" if the patient has the
targeted condition but fails a criterion, or "not_relevant" if the trial does not fit the patient.
Return "score" from 0 to 100 for how good a match it is."""


def bm25_scores(note: PatientNote, trials: List[Trial], parsed: Dict[str, ParsedTrial]) -> Dict[str, float]:
    return TrialIndex(trials, parsed).scores(note.text)


def single_call(note: PatientNote, trial: Trial) -> SingleCallOutput:
    user = (f"PATIENT NOTE:\n{note.text}\n\nTRIAL: {trial.title}\nConditions: {', '.join(trial.conditions)}\n"
            f"Eligibility criteria:\n{(trial.eligibility_text or '')[:6000]}")
    return llm_local.chat_json(SINGLE_SYSTEM, user, SingleCallOutput, "single")
