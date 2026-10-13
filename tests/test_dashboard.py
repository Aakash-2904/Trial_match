"""Dashboard server tests (no Ollama needed: the agent graphs are replaced with stubs)."""
import json

import pytest
from starlette.testclient import TestClient

import supervisor
from config import RESULTS_DIR
from dashboard import server
from schemas import Fact, ParsedTrial, PatientProfile, Trial, TrialMatch

NOTE = "A 62-year-old woman with type 2 diabetes on metformin and heart failure with reduced ejection fraction."
TRIALS = [Trial(nct_id=f"NCT0000000{i}", title=f"Trial {i}") for i in range(3)]


class _Index:
    def search(self, text, k):
        return TRIALS[:k]


class _PatientGraph:
    def invoke(self, s):
        return {"profile": PatientProfile(age=62, sex="female", conditions=[Fact(text="T2D", evidence="type 2 diabetes")]),
                "relevance": {t.nct_id: 0.5 for t in TRIALS}, "demo_fail": {}}


class _TrialGraph:
    def stream(self, init, stream_mode):
        t = init["trial"]
        yield {"rule_checker": {"results": [], "pending": []}}
        yield {"scorer": {"match": TrialMatch(nct_id=t.nct_id, title=t.title, label="possibly_eligible",
                                             eligibility_score=0.5, relevance_score=0.5, final_score=0.5,
                                             explanation="x", results=[])}}


@pytest.fixture
def client(monkeypatch):
    parsed = {t.nct_id: ParsedTrial(nct_id=t.nct_id, criteria=[]) for t in TRIALS}
    monkeypatch.setattr(server, "load_snapshot", lambda key: (TRIALS, parsed, _Index(), None, None))
    monkeypatch.setattr(supervisor, "PATIENT_GRAPH", _PatientGraph())
    monkeypatch.setattr(supervisor, "TRIAL_GRAPH", _TrialGraph())
    return TestClient(server.app)


def _events(client, body):
    kinds, data = [], []
    with client.stream("POST", "/api/match", json=body) as r:
        assert r.status_code == 200
        for line in r.iter_lines():
            if line.startswith("event: "):
                kinds.append(line[7:])
            elif line.startswith("data: "):
                data.append(json.loads(line[6:]))
    return kinds, data


def test_match_streams_profile_matches_and_order(client):
    kinds, data = _events(client, {"note": NOTE, "snapshot": "demo", "top_k": 3, "rag": False})
    assert kinds[0] == "stage" and kinds[-1] == "done"
    assert kinds.count("match") == 3 and "profile" in kinds and "candidates" in kinds
    assert len(data[-1]["order"]) == 3
    assert not server._run_lock.locked()          # released after the run


def test_match_rejects_short_note_and_unknown_snapshot(client):
    assert client.post("/api/match", json={"note": "short"}).status_code == 400
    assert client.post("/api/match", json={"note": NOTE, "snapshot": "nope"}).status_code == 400


def test_match_refuses_concurrent_runs(client):
    assert server._run_lock.acquire(blocking=False)
    try:
        assert client.post("/api/match", json={"note": NOTE, "snapshot": "demo"}).status_code == 409
    finally:
        server._run_lock.release()


def test_eval_metrics_match_run_eval_summary(client):
    """The dashboard recomputes metrics from rankings CSVs; they must agree with run_eval's summary files."""
    pairs = [("dev_after", "summary_dev_after.json"), ("dev_before", "summary_batch_B_all5_p2.json"),
             ("heldout", "summary_heldout.json")]
    checked = 0
    for run, summary_file in pairs:
        if not (RESULTS_DIR / summary_file).exists():
            continue
        r = client.get(f"/api/eval/run/{run}")
        if r.status_code == 404:
            continue
        got = r.json()["summary"]
        want = json.loads((RESULTS_DIR / summary_file).read_text(encoding="utf-8"))
        for sysname, metrics in want.items():
            for m, v in metrics.items():
                assert (got[sysname][m] is None) == (v is None), (run, sysname, m)
                if v is not None:
                    assert got[sysname][m] == pytest.approx(v, abs=1e-3), (run, sysname, m)
        checked += 1
    if not checked:
        pytest.skip("no evaluation results present")


def test_detail_rejects_path_tricks(client):
    assert client.get("/api/eval/detail/..%2Fconfig_1").status_code in (400, 404)
    assert client.get("/api/eval/detail/notasystem_4").status_code == 400
