"""Fast unit tests for the non-LLM agents (no Ollama needed)."""
from agents import criteria_parser, rule_checker, scorer
from agents.prefilter import demographic_fail
from metrics import ndcg_at_k, precision_at_k
from agents.profiler import ground_profile
from schemas import Criterion, Fact, LabValue, NumericCheck, ParsedTrial, PatientNote, PatientProfile, Trial
from utils import NEGATION, grounded, negates, outcome_for, supports

PROFILE = PatientProfile(age=62, sex="female",
                         labs=[LabValue(name="eGFR", value=52, unit="mL/min", evidence="eGFR 52 mL/min")])


def test_outcome_mapping():
    assert outcome_for("inclusion", "yes") == "met"
    assert outcome_for("exclusion", "yes") == "not_met"
    assert outcome_for("exclusion", "no") == "met"
    assert outcome_for("inclusion", "unknown") == "unknown"


def test_numeric_rule_on_exclusion():
    c = Criterion(id="E1", kind="exclusion", text="eGFR < 30", check=NumericCheck(field="eGFR", op="<", value=30))
    r = rule_checker.check_numeric(c, PROFILE)
    assert r.statement_true == "no" and r.outcome == "met"


def test_missing_lab_is_unknown():
    c = Criterion(id="I1", kind="inclusion", text="HbA1c 7-10%", check=NumericCheck(field="HbA1c", op=">=", value=7))
    assert rule_checker.check_numeric(c, PROFILE).outcome == "unknown"


def test_demographics():
    assert demographic_fail(PROFILE, Trial(nct_id="x", title="t", max_age=50)) is not None
    assert demographic_fail(PROFILE, Trial(nct_id="x", title="t", sex="MALE")) is not None
    assert demographic_fail(PROFILE, Trial(nct_id="x", title="t", min_age=18, max_age=80)) is None


def test_local_parser_splits_sections():
    t = Trial(nct_id="x", title="t", eligibility_text=(
        "Inclusion Criteria:\n\n* Age 18 or older\n* Type 2 diabetes\n\nExclusion Criteria:\n\n* Pregnancy"))
    p = criteria_parser.parse_locally(t)
    assert [c.kind for c in p.criteria] == ["inclusion", "inclusion", "exclusion"]


def test_scorer_excludes_on_failure():
    c = Criterion(id="E1", kind="exclusion", text="eGFR < 60", check=NumericCheck(field="eGFR", op="<", value=60))
    r = rule_checker.check_numeric(c, PROFILE)
    m = scorer.score(Trial(nct_id="x", title="t"), ParsedTrial(nct_id="x", criteria=[c]), [r], 1.0)
    assert m.label == "likely_ineligible"


def test_metrics():
    gains = {"a": 2, "b": 1, "c": 0}
    assert ndcg_at_k(["a", "b", "c"], gains) == 1.0
    assert precision_at_k(["a", "b", "c"], gains, 1) == 1.0


def test_rrf_rewards_agreement():
    from rag.hybrid import rrf
    fused = rrf([["a", "b", "c"], ["b", "a", "c"]])
    assert fused["a"] == fused["b"] > fused["c"]


NOTE = PatientNote(patient_id="t", text="A 58-year-old man admitted [**2137-3-2**] with newly diagnosed pancreatic "
                                         "adenocarcinoma. He has HTN and takes lisinopril 20 mg. Denies chest pain.")


def test_grounded_rejects_placeholders_and_new_numbers():
    assert not grounded("[**2137**]", NOTE.text)
    assert not grounded("[**2137-3-2**] 12", NOTE.text)
    assert not grounded("lisinopril 40 mg", NOTE.text)            # invented number
    assert grounded("takes lisinopril 20 mg", NOTE.text)
    assert grounded("newly diagnosed pancreatic adenocarcinoma", NOTE.text)


def test_supports_and_negation():
    assert supports("pancreatic cancer", "newly diagnosed pancreatic adenocarcinoma")
    assert not supports("diabetes", "takes lisinopril 20 mg")
    assert NEGATION.search("Denies chest pain") and not NEGATION.search("has chest pain")


def test_ground_profile_filters_hallucinations():
    raw = PatientProfile(
        conditions=[Fact(text="pancreatic adenocarcinoma", evidence="newly diagnosed pancreatic adenocarcinoma"),
                    Fact(text="diabetes", evidence="[**2137**]"),                  # placeholder evidence
                    Fact(text="asthma", evidence="takes lisinopril 20 mg")],        # evidence about something else
        negatives=[Fact(text="chest pain", evidence="Denies chest pain"),
                   Fact(text="HTN", evidence="He has HTN")])                        # not an explicit denial
    p = ground_profile(raw, NOTE)
    assert [f.text for f in p.conditions] == ["pancreatic adenocarcinoma"]
    assert [f.text for f in p.negatives] == ["chest pain"]
    assert p.age == 58 and p.sex == "male"


def test_grounded_matches_whole_words_only():
    assert not grounded("AST", "Past Medical History: HTN")
    assert grounded("ICH", "meningitis-->ICH-->seizure")


def test_negation_scope():
    ev = ("Fluid was consistent with Burkitt's lymphoma. Bone marrow biopsy was negative for any involvement "
          "which made this a primary cardiac lymphoma.")
    assert negates("involvement", ev)
    assert not negates("Negative for Burkitt's lymphoma", ev)
    assert negates("chest pain", "Denies chest pain")
    assert negates("night sweats", "no fever, chills, or night sweats")


def test_wilcoxon_exact():
    from scripts.significance import wilcoxon_exact
    assert abs(wilcoxon_exact([1, 2, 3, 4, 5]) - 2 / 32) < 1e-12               # all positive, n=5
    assert abs(wilcoxon_exact([1, -2, 3, 4, 5, 6, 7, 8, 9, 10]) - 6 / 1024) < 1e-12   # W- = 2: subsets {}, {1}, {2}
    assert wilcoxon_exact([0, 0]) == 1.0
    assert wilcoxon_exact([0.1, -0.1]) == 1.0                                   # tied magnitudes, opposite signs


def test_denials_listed_as_conditions_become_negatives():
    note = PatientNote(patient_id="t", text="74M with HTN. No prior syncope. Stroke 10 years ago without residual deficit.")
    raw = PatientProfile(conditions=[Fact(text="HTN", evidence="74M with HTN"),
                                     Fact(text="No prior syncope", evidence="No prior syncope"),
                                     Fact(text="Prior stroke", evidence="Stroke 10 years ago without residual deficit")],
                         negatives=[Fact(text="No prior syncope", evidence="No prior syncope")])
    p = ground_profile(raw, note)
    assert [f.text for f in p.conditions] == ["HTN", "Prior stroke"]
    assert [f.text for f in p.negatives] == ["No prior syncope"]
    assert p.age == 74 and p.sex == "male"
