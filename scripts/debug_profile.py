"""Show exactly what the profiler extracts for one TREC patient: raw model output vs. what survives grounding.

  python -m scripts.debug_profile 4
"""
import json
import sys

import llm_local
from agents.profiler import SYSTEM, ground_profile
from config import DATA_DIR
from schemas import PatientNote, PatientProfile


def show(title, prof):
    print(f"\n--- {title}: age {prof.age}, sex {prof.sex}")
    for field in ("conditions", "medications", "procedures", "negatives", "other"):
        for f in getattr(prof, field):
            print(f"  {field:11s} {f.text!r:45.45s}  evidence: {f.evidence!r}")
    for l in prof.labs:
        print(f"  {'lab':11s} {l.name} = {l.value:g} {l.unit or ''}  evidence: {l.evidence!r}")


def main():
    tid = sys.argv[1] if len(sys.argv) > 1 else next(iter(json.loads((DATA_DIR / "eval_set.json").read_text())))
    topics = json.loads((DATA_DIR / "topics2021.json").read_text(encoding="utf-8"))
    note = PatientNote(patient_id=tid, text=topics[tid])
    print(f"PATIENT {tid}:\n{note.text}")
    raw = llm_local.chat_json(SYSTEM, f"PATIENT NOTE:\n{note.text}", PatientProfile, "profile")
    show("RAW model output", raw)
    show("AFTER grounding (what the agents see)", ground_profile(raw, note))


if __name__ == "__main__":
    main()  