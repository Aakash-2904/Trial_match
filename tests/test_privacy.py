"""The privacy boundary is enforced in code, and these tests prove it."""
import re
from pathlib import Path

import pytest

import llm_public
from schemas import PatientNote, PublicTrialText, TrialParseOutput

ROOT = Path(__file__).resolve().parents[1]
NOTE = PatientNote(patient_id="p1", text="A 62-year-old woman with type 2 diabetes.")


@pytest.mark.parametrize("payload", [NOTE, NOTE.text, NOTE.model_dump(), ["text"], None])
def test_non_public_payloads_are_rejected(payload):
    with pytest.raises(llm_public.PrivacyViolation):
        llm_public.generate_json(payload, "x", TrialParseOutput, "test")


def test_public_trial_text_passes_the_guard():
    llm_public._guard(PublicTrialText(nct_id="NCT0", title="t"))     # must not raise


def test_only_llm_public_imports_a_cloud_sdk():
    pattern = re.compile(r"^\s*(from\s+google(\.genai)?\s+import|import\s+google\.genai|import\s+openai|"
                         r"from\s+openai|import\s+anthropic|from\s+anthropic)", re.M)
    offenders = [p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*.py")
                 if ".venv" not in p.parts and p.name != "llm_public.py"
                 and pattern.search(p.read_text(encoding="utf-8"))]
    assert offenders == [], f"cloud SDK imported outside llm_public.py: {offenders}"


def test_patient_agents_never_import_llm_public():
    for name in ("profiler.py", "llm_checker.py", "critic.py", "rule_checker.py", "scorer.py", "prefilter.py"):
        src = (ROOT / "agents" / name).read_text(encoding="utf-8")
        assert "llm_public" not in src, f"agents/{name} must not use the public LLM"
    for name in ("embed.py", "hybrid.py", "knowledge.py", "vector_store.py"):
        src = (ROOT / "rag" / name).read_text(encoding="utf-8")
        assert "llm_public" not in src, f"rag/{name} must not use the public LLM"
