"""Supervisor: orchestrates the agents with two LangGraph graphs.

patient graph : profiler -> prefilter                      (once per patient)
trial graph   : rule_checker -> [eligibility_checker] -> [critic] -> scorer   (per trial, with routing)

Trials are screened one at a time on purpose: on a CPU laptop sequential calls let Ollama
reuse the cached patient prefix, and streaming each step keeps UI updates on the main thread.
"""
from typing import Callable, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from agents import critic, llm_checker, prefilter, profiler, rule_checker, scorer
from schemas import Criterion, CriterionResult, ParsedTrial, PatientNote, PatientProfile, Trial, TrialMatch


# ---------- patient graph ----------
class PatientState(TypedDict, total=False):
    note: PatientNote
    trials: List[Trial]
    parsed: Dict[str, ParsedTrial]
    profile: PatientProfile
    relevance: Dict[str, float]
    demo_fail: Dict[str, str]
    retriever: object        # optional HybridRetriever (RAG)


def _profiler_node(s: PatientState):
    return {"profile": profiler.profile_patient(s["note"])}


def _prefilter_node(s: PatientState):
    if s.get("retriever") is not None:      # hybrid BM25 + dense, fused with RRF
        rel = s["retriever"].scores(s["note"].text, restrict_to=[t.nct_id for t in s["trials"]])
    else:
        rel = prefilter.relevance_scores(s["note"].text, s["trials"], s["parsed"])
    fails = {}
    for t in s["trials"]:
        reason = prefilter.demographic_fail(s["profile"], t)
        if reason:
            fails[t.nct_id] = reason
    return {"relevance": rel, "demo_fail": fails}


def build_patient_graph():
    g = StateGraph(PatientState)
    g.add_node("profiler", _profiler_node)
    g.add_node("prefilter", _prefilter_node)
    g.add_edge(START, "profiler")
    g.add_edge("profiler", "prefilter")
    g.add_edge("prefilter", END)
    return g.compile()


# ---------- trial graph ----------
class TrialState(TypedDict, total=False):
    note: PatientNote
    profile: PatientProfile
    trial: Trial
    parsed: ParsedTrial
    relevance: float
    demo_fail: Optional[str]
    use_critic: bool
    kb: object               # optional KnowledgeBase (RAG)
    results: List[CriterionResult]
    pending: List[Criterion]
    match: TrialMatch


def _rule_node(s: TrialState):
    if s.get("demo_fail"):
        return {"results": [rule_checker.demographic_result(s["demo_fail"])], "pending": []}
    results, pending = rule_checker.run(s["parsed"].criteria, s["profile"])
    return {"results": results, "pending": pending}


def _route_after_rules(s: TrialState) -> str:
    return "eligibility_checker" if s.get("pending") else "scorer"


def _llm_node(s: TrialState):
    return {"results": s["results"] + llm_checker.run(s["pending"], s["profile"], s["note"], kb=s.get("kb"))}


def _route_after_llm(s: TrialState) -> str:
    return "critic" if s.get("use_critic", True) and critic.needs_review(s["results"]) else "scorer"


def _critic_node(s: TrialState):
    return {"results": critic.run(s["results"], s["note"])}


def _scorer_node(s: TrialState):
    return {"match": scorer.score(s["trial"], s["parsed"], s["results"], s["relevance"])}


def build_trial_graph():
    g = StateGraph(TrialState)
    g.add_node("rule_checker", _rule_node)
    g.add_node("eligibility_checker", _llm_node)
    g.add_node("critic", _critic_node)
    g.add_node("scorer", _scorer_node)
    g.add_edge(START, "rule_checker")
    g.add_conditional_edges("rule_checker", _route_after_rules,
                            {"eligibility_checker": "eligibility_checker", "scorer": "scorer"})
    g.add_conditional_edges("eligibility_checker", _route_after_llm, {"critic": "critic", "scorer": "scorer"})
    g.add_edge("critic", "scorer")
    g.add_edge("scorer", END)
    return g.compile()


PATIENT_GRAPH = build_patient_graph()
TRIAL_GRAPH = build_trial_graph()


# ---------- public entry point ----------
def _describe(node: str, out: dict) -> str:
    if node == "rule_checker":
        n_rule = len(out.get("results", []))
        return f"rule checker decided {n_rule} criteria, {len(out.get('pending', []))} left for the LLM"
    if node == "eligibility_checker":
        return "eligibility checker judged the remaining criteria"
    if node == "critic":
        n = sum(r.source == "critic" for r in out.get("results", []))
        return f"critic reviewed exclusions, overturned {n}"
    if node == "scorer":
        m = out["match"]
        return f"scored {m.final_score:.2f} ({m.label.replace('_', ' ')})"
    return node


def match_patient(note: PatientNote, trials: List[Trial], parsed: Dict[str, ParsedTrial],
                  use_critic: bool = True, retriever=None, kb=None,
                  on_event: Optional[Callable[[str, str], None]] = None) -> List[TrialMatch]:
    emit = on_event or (lambda agent, msg: None)
    emit("supervisor", f"screening {len(trials)} candidate trials")

    ps = PATIENT_GRAPH.invoke({"note": note, "trials": trials, "parsed": parsed, "retriever": retriever})
    p = ps["profile"]
    emit("profiler", f"profile: age {p.age}, sex {p.sex}, {len(p.conditions)} conditions, "
                     f"{len(p.medications)} medications, {len(p.procedures)} procedures, {len(p.labs)} labs, "
                     f"{len(p.negatives)} negatives, {len(p.other)} other")
    emit("prefilter", f"{len(ps['demo_fail'])} trials fail age/sex rules")

    matches = []
    for i, t in enumerate(trials, 1):
        init = {"note": note, "profile": p, "trial": t, "parsed": parsed[t.nct_id],
                "relevance": ps["relevance"].get(t.nct_id, 0.0),
                "demo_fail": ps["demo_fail"].get(t.nct_id), "use_critic": use_critic, "kb": kb}
        for update in TRIAL_GRAPH.stream(init, stream_mode="updates"):
            for node, out in update.items():
                emit(node, f"[{i}/{len(trials)}] {t.nct_id}: {_describe(node, out)}")
                if node == "scorer":
                    matches.append(out["match"])
    matches.sort(key=scorer.rank_key)
    return matches