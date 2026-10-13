"""Streamlit demo. Runs fully offline: local model + saved trial snapshot, no API calls.

  streamlit run app.py
"""
import json

import streamlit as st

from agents import criteria_parser
from agents.prefilter import TrialIndex
from config import DATA_DIR, LOCAL_MODEL
from schemas import ParsedTrial, PatientNote, Trial
from rag.hybrid import TRIAL_STORE, HybridRetriever
from rag.knowledge import KB_STORE, KnowledgeBase
from rag.vector_store import VectorStore
from supervisor import match_patient

EXAMPLE = ("A 62-year-old woman presents with fatigue and a 6-month history of worsening shortness of breath. "
           "She was diagnosed with type 2 diabetes 10 years ago and takes metformin 1000 mg twice daily. "
           "Recent labs show HbA1c 8.4% and eGFR 52 mL/min. Echocardiogram shows an ejection fraction of 35%. "
           "She had a myocardial infarction 3 years ago. She has no history of cancer and is not pregnant.")

LABEL = {"likely_eligible": ("Likely eligible", "green"),
         "possibly_eligible": ("Possibly eligible, needs review", "orange"),
         "likely_ineligible": ("Likely ineligible", "red")}
ICON = {"met": "✅ met", "not_met": "❌ not met", "unknown": "❔ unknown"}


@st.cache_resource
def load_snapshot(source: str):
    path = DATA_DIR / ("demo_trials.json" if source == "Recruiting (demo)" else "trials.json")
    trials = [Trial(**v) for v in json.loads(path.read_text(encoding="utf-8")).values()]
    parsed = {}
    for name in ("parsed_local.json", "parsed_api.json"):          # API parse overrides local if present
        p = DATA_DIR / name
        if p.exists():
            parsed.update({k: ParsedTrial(**v) for k, v in json.loads(p.read_text(encoding="utf-8")).items()})
    for t in trials:                                               # anything unparsed: local parse, offline
        parsed.setdefault(t.nct_id, criteria_parser.parse_locally(t))
    retriever = kb = None
    if VectorStore.exists(TRIAL_STORE):
        store = VectorStore.load(TRIAL_STORE)
        if all(t.nct_id in store.ids for t in trials):
            retriever = HybridRetriever(trials, parsed, store)
    if VectorStore.exists(KB_STORE):
        kb = KnowledgeBase(VectorStore.load(KB_STORE))
    return trials, parsed, TrialIndex(trials, parsed), retriever, kb


st.set_page_config(page_title="Local trial matcher", layout="wide")
st.title("Privacy-preserving clinical trial matching")
st.caption(f"Multi-agent screening on this machine with {LOCAL_MODEL}. The patient note never leaves the laptop. "
           "Research prototype, not for clinical use.")

with st.sidebar:
    sources = [s for s, f in (("Recruiting (demo)", "demo_trials.json"), ("TREC eval trials", "trials.json"))
               if (DATA_DIR / f).exists()]
    if not sources:
        st.error("No trial snapshot found. Run: python -m scripts.download_data --demo ...")
        st.stop()
    source = st.selectbox("Trial snapshot", sources)
    top_k = st.slider("Trials to screen", 3, 20, 8, help="Each trial takes roughly a minute on a laptop CPU")
    use_critic = st.toggle("Critic agent", value=True)
    use_rag = st.toggle("RAG (hybrid search + medical knowledge)", value=True)

note_text = st.text_area("Patient note", EXAMPLE, height=160)

if st.button("Find trials", type="primary"):
    if len(note_text.strip()) < 30:
        st.warning("Please enter a longer patient description.")
        st.stop()
    trials, parsed, index, retriever, kb = load_snapshot(source)
    if use_rag and retriever is None:
        st.info("RAG index not built for this snapshot; using BM25. Run: python -m scripts.build_index")
    retriever = retriever if use_rag else None
    kb = kb if use_rag else None
    candidates = (retriever or index).search(note_text, top_k)
    note = PatientNote(patient_id="demo", text=note_text)

    with st.status("Agents working...", expanded=True) as status:
        matches = match_patient(note, candidates, parsed, use_critic=use_critic, retriever=retriever, kb=kb,
                                on_event=lambda agent, msg: status.write(f"**{agent}** · {msg}"))
        status.update(label=f"Screened {len(matches)} trials", state="complete", expanded=False)

    for rank, m in enumerate(matches, 1):
        text, color = LABEL[m.label]
        with st.expander(f"{rank}. {m.nct_id} · {m.title}  —  score {m.final_score:.2f}", expanded=rank <= 2):
            st.markdown(f":{color}[**{text}**]  ·  {m.explanation}")
            if m.plain_summary:
                st.caption(m.plain_summary)
            st.dataframe([{"Outcome": ICON[r.outcome], "Type": r.criterion.kind, "Criterion": r.criterion.text,
                           "Evidence": r.evidence or "", "Decided by": r.source, "Note": r.note or ""}
                          for r in m.results], use_container_width=True, hide_index=True)
            st.link_button("View on ClinicalTrials.gov", f"https://clinicaltrials.gov/study/{m.nct_id}")

    st.download_button("Download results (JSON)", json.dumps([m.model_dump() for m in matches], indent=1),
                       file_name="trial_matches.json", mime="application/json")
