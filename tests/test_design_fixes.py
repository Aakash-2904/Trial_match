"""Unit tests for the design fixes (no Ollama needed: LLM calls are replaced with stubs)."""
import numpy as np

import llm_local
from agents import critic, llm_checker
from agents.groups import group_results, needed
from agents.scorer import rank_key, score
from rag import knowledge
from rag.knowledge import KnowledgeBase, expand_abbreviations
from rag.vector_store import VectorStore
from schemas import (CheckerOutput, ConceptCard, CriterionJudgment, CriterionResult, CriticDecision, CriticOutput,
                     Criterion, ParsedTrial, PatientNote, PatientProfile, Trial, TrialMatch)
from scripts.run_eval import select_topics

NOTE = PatientNote(patient_id="t", text="62-year-old woman with type 2 diabetes. Myocardial infarction 3 years ago. "
                                        "Takes metformin. Denies chest pain.")
PROFILE = PatientProfile(age=62, sex="female")
INC = Criterion(id="I1", kind="inclusion", text="Type 2 diabetes", category="condition")
EXC = Criterion(id="E1", kind="exclusion", text="Myocardial infarction within 3 months", category="condition")


def stub_checker(monkeypatch, judgments, seen=None):
    def fake(system, user, schema, namespace, **kw):
        if seen is not None:
            seen.append(user)
        return CheckerOutput(judgments=[CriterionJudgment(**j) for j in judgments])
    monkeypatch.setattr(llm_local, "chat_json", fake)


def result(c, truth, source="llm", evidence=None):
    from utils import outcome_for
    return CriterionResult(criterion=c, statement_true=truth, outcome=outcome_for(c.kind, truth), evidence=evidence,
                           source=source)


# 1. a not_met judgment needs grounded evidence (checker)
def test_checker_not_met_needs_grounded_evidence(monkeypatch):
    stub_checker(monkeypatch, [{"id": "I1", "statement_true": "no", "evidence": None},
                               {"id": "E1", "statement_true": "yes", "evidence": "myocardial infarction last week"}])
    r = llm_checker.run([INC, EXC], PROFILE, NOTE)
    assert [x.outcome for x in r] == ["unknown", "unknown"]
    assert [x.source for x in r] == ["grounding", "grounding"]

    stub_checker(monkeypatch, [{"id": "E1", "statement_true": "yes", "evidence": "Myocardial infarction 3 years ago"}])
    assert llm_checker.run([EXC], PROFILE, NOTE)[0].outcome == "not_met"      # grounded: kept


# 2. a met judgment needs grounded evidence too
def test_checker_met_needs_grounded_evidence(monkeypatch):
    stub_checker(monkeypatch, [{"id": "I1", "statement_true": "yes", "evidence": None}])
    assert llm_checker.run([INC], PROFILE, NOTE)[0].outcome == "unknown"
    stub_checker(monkeypatch, [{"id": "I1", "statement_true": "yes", "evidence": "type 2 diabetes"}])
    assert llm_checker.run([INC], PROFILE, NOTE)[0].outcome == "met"


def stub_critic(monkeypatch, decisions):
    monkeypatch.setattr(llm_local, "chat_json",
                        lambda *a, **k: CriticOutput(decisions=[CriticDecision(**d) for d in decisions]))


# 1 + 3. critic: overturn to the value its reasoning supports, but only with grounded evidence
def test_critic_overturns_to_supported_value(monkeypatch):
    wrong = result(EXC, "yes", evidence="Myocardial infarction 3 years ago")       # checker: excluded
    stub_critic(monkeypatch, [{"id": "E1", "verdict": "overturn", "corrected": "no",
                               "evidence": "Myocardial infarction 3 years ago", "reason": "MI was 3 years ago"}])
    r = critic.run([wrong], NOTE)[0]
    assert (r.statement_true, r.outcome, r.source) == ("no", "met", "critic")
    assert "3 years ago" in critic.SYSTEM and '"unknown" only when' in critic.SYSTEM


def test_critic_value_without_grounded_evidence_becomes_unknown(monkeypatch):
    wrong = result(EXC, "yes", evidence="Myocardial infarction 3 years ago")
    stub_critic(monkeypatch, [{"id": "E1", "verdict": "overturn", "corrected": "no",
                               "evidence": "no cardiac history at all", "reason": "guess"}])
    r = critic.run([wrong], NOTE)[0]
    assert (r.statement_true, r.outcome) == ("unknown", "unknown")


# 4. ranking: eligibility tier first, relevance only within a tier
def test_rank_by_tier_then_relevance():
    def m(nct, label, rel):
        return TrialMatch(nct_id=nct, title="t", label=label, eligibility_score=0, relevance_score=rel,
                          final_score=0, explanation="", results=[])
    ms = [m("a", "likely_ineligible", 1.0), m("b", "possibly_eligible", 0.2), m("c", "likely_eligible", 0.1),
          m("d", "possibly_eligible", 0.9)]
    assert [x.nct_id for x in sorted(ms, key=rank_key)] == ["c", "d", "b", "a"]


# 5. either/or groups
def opt(i, body, kind="inclusion", head="At least one of the following"):
    return Criterion(id=f"I{i}", kind=kind, text=f"{head}: {body}", category="condition")


def test_any_of_group():
    opts = [opt(1, "HTN"), opt(2, "diabetes"), opt(3, "age over 70")]
    one_yes = group_results([result(opts[0], "no"), result(opts[1], "yes", evidence="type 2 diabetes"),
                             result(opts[2], "unknown")])
    assert len(one_yes) == 1 and one_yes[0].outcome == "met" and one_yes[0].criterion.id == "I1-I3"
    all_no = group_results([result(o, "no") for o in opts])
    assert all_no[0].outcome == "not_met"
    some_unknown = group_results([result(opts[0], "no"), result(opts[1], "unknown"), result(opts[2], "no")])
    assert some_unknown[0].outcome == "unknown"


def test_k_of_n_group_and_non_groups():
    head = "SIRS established by at least two of the following"
    assert needed(head) == 2 and needed("At least one additional risk factor for HF") == 1
    opts = [opt(i, f"sign {i}", head=head) for i in range(1, 4)]
    assert group_results([result(opts[0], "yes", evidence="x"), result(opts[1], "no"),
                          result(opts[2], "unknown")])[0].outcome == "unknown"
    assert group_results([result(opts[0], "yes", evidence="x"), result(opts[1], "no"),
                          result(opts[2], "no")])[0].outcome == "not_met"
    age = Criterion(id="I1", kind="inclusion", text="Age: at least 18 years", category="age")
    plain = [result(age, "yes"), result(INC, "no")]
    assert [r.criterion.id for r in group_results(plain)] == ["I1", "I1"]             # untouched


def test_exclusion_group_and_local_heading_shape():
    ex = [opt(1, "dialysis", kind="exclusion", head="Any of the following"),
          opt(2, "cirrhosis", kind="exclusion", head="Any of the following")]
    assert group_results([result(ex[0], "no"), result(ex[1], "yes", evidence="x")])[0].outcome == "not_met"
    heading = Criterion(id="I2", kind="inclusion", text="Presence of at least one of the following:")
    a, b = Criterion(id="I3", kind="inclusion", text="HTN"), Criterion(id="I4", kind="inclusion", text="CKD")
    nxt = Criterion(id="I5", kind="inclusion", text="Other heading:")
    out = group_results([result(heading, "unknown"), result(a, "no"), result(b, "yes", evidence="x"), result(nxt, "unknown")])
    assert [r.criterion.id for r in out] == ["I3-I4", "I5"] and out[0].outcome == "met"


def test_scorer_uses_groups():
    opts = [opt(1, "HTN"), opt(2, "diabetes")]
    m = score(Trial(nct_id="x", title="t"), ParsedTrial(nct_id="x", criteria=opts),
              [result(opts[0], "no"), result(opts[1], "yes", evidence="type 2 diabetes")], 0.5)
    assert m.label == "likely_eligible"                # before: the "no" option made it likely_ineligible


# 6. knowledge retrieval matches abbreviations
def test_abbreviation_expansion(monkeypatch):
    assert "heart failure" in expand_abbreviations("History of HF or MI")
    assert "myocardial infarction" in expand_abbreviations("History of HF or MI")
    assert expand_abbreviations("family history") == "family history"           # no false 'mi' hit
    card = ConceptCard(term="heart failure", definition="d")
    store = VectorStore(["heart failure"], np.ones((1, 2), np.float32) / np.sqrt(2), [card.model_dump()])
    monkeypatch.setattr(knowledge, "embed", lambda texts, kind: np.array([[1.0, -1.0]] * len(texts)) / np.sqrt(2))
    crit = Criterion(id="E1", kind="exclusion", text="Hospitalised for HF in the last year", category="condition")
    assert [c.term for c in KnowledgeBase(store).for_criteria([crit])] == ["heart failure"]


# 7. RAG prompt labels the knowledge as background, never evidence
def test_knowledge_section_is_labelled(monkeypatch):
    seen = []
    stub_checker(monkeypatch, [{"id": "I1", "statement_true": "unknown"}], seen)

    class KB:
        def for_criteria(self, criteria):
            return [ConceptCard(term="metformin", definition="a biguanide")]
    llm_checker.run([INC], PROFILE, NOTE, kb=KB())
    assert "NOT facts about this patient; never quote it as evidence" in seen[0]
    assert seen[0].index("PATIENT NOTE") < seen[0].index("MEDICAL KNOWLEDGE")


# 8. markdown escapes are cleaned from criterion text
def test_markdown_escapes_cleaned():
    c = Criterion(id="I1", kind="exclusion", text=r"hemoglobin\<9.5 gms/dl, serum creatinine\>1.5mg/100ml")
    assert c.text == "hemoglobin<9.5 gms/dl, serum creatinine>1.5mg/100ml"


# 9. run_eval --exclude
def test_select_topics_exclude():
    ev = {"4": [], "5": [], "12": [], "20": []}
    assert select_topics(ev, ["4", "5"]) == ["12", "20"]
    assert select_topics(ev, ["4"], 2) == ["5", "12"]
    try:
        select_topics(ev, ["99"])
        assert False, "unknown topic should be rejected"
    except SystemExit:
        pass
