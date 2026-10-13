"""Step 1 (one-time, needs internet): download public data and save it locally.

  python -m scripts.download_data
  python -m scripts.download_data --demo "heart failure" "type 2 diabetes" "breast cancer"

Saves to data/: topics2021.json, qrels2021.json, eval_set.json, trials.json, and optionally demo_trials.json.
Note: TREC 2021 was judged on an April 2021 registry snapshot; the current ClinicalTrials.gov text can
differ for some trials. Record this as a limitation in the report.
"""
import argparse
import json
import random
import re
import time
import xml.etree.ElementTree as ET
from collections import defaultdict

import requests
from tqdm import tqdm

from config import (CTGOV_SEARCH_URL, CTGOV_STUDY_URL, DATA_DIR, EVAL_PATIENTS, EVAL_TRIALS_PER_PATIENT,
                    QRELS_URL, SEED, TOPICS_URL)

HEADERS = {"User-Agent": "trialmatch-research/0.1 (student project)"}


def parse_age(s):
    if not s or s.upper() == "N/A":
        return None
    m = re.match(r"([\d.]+)\s*(year|month|week|day)", s, re.I)
    if not m:
        return None
    v, unit = float(m.group(1)), m.group(2).lower()
    return {"year": v, "month": v / 12, "week": v / 52, "day": v / 365}[unit]


def study_to_trial(d: dict) -> dict:
    ps = d.get("protocolSection", {})
    ident, el = ps.get("identificationModule", {}), ps.get("eligibilityModule", {})
    return {
        "nct_id": ident.get("nctId"),
        "title": ident.get("briefTitle") or ident.get("officialTitle", ""),
        "conditions": ps.get("conditionsModule", {}).get("conditions", []),
        "summary": ps.get("descriptionModule", {}).get("briefSummary", ""),
        "eligibility_text": el.get("eligibilityCriteria", ""),
        "min_age": parse_age(el.get("minimumAge")),
        "max_age": parse_age(el.get("maximumAge")),
        "sex": el.get("sex", "ALL"),
        "status": ps.get("statusModule", {}).get("overallStatus", ""),
    }


def fetch_trial(nct: str):
    r = requests.get(CTGOV_STUDY_URL.format(nct=nct), headers=HEADERS, timeout=30)
    if r.status_code != 200:
        return None
    t = study_to_trial(r.json())
    return t if t["eligibility_text"] else None


def download_trec():
    topics_xml = requests.get(TOPICS_URL, headers=HEADERS, timeout=60).content  # bytes: XML has an encoding header
    topics = {t.get("number"): (t.text or "").strip() for t in ET.fromstring(topics_xml).iter("topic")}
    qrels = defaultdict(dict)
    for line in requests.get(QRELS_URL, headers=HEADERS, timeout=60).text.splitlines():
        parts = line.split()
        if len(parts) == 4:
            qrels[parts[0]][parts[2]] = int(parts[3])
    (DATA_DIR / "topics2021.json").write_text(json.dumps(topics, indent=1), encoding="utf-8")
    (DATA_DIR / "qrels2021.json").write_text(json.dumps(qrels), encoding="utf-8")
    print(f"topics: {len(topics)}, judged topics: {len(qrels)}")
    return topics, qrels


def build_eval_set(qrels):
    """Pick patients with both eligible and excluded trials; stratify trials across the 3 labels."""
    rng = random.Random(SEED)
    good = [t for t, q in qrels.items()
            if sum(v == 2 for v in q.values()) >= 2 and sum(v == 1 for v in q.values()) >= 2]
    topics = sorted(rng.sample(good, min(EVAL_PATIENTS, len(good))), key=int)
    eval_set, per = {}, max(EVAL_TRIALS_PER_PATIENT // 3, 1)
    for t in topics:
        by = {g: [n for n, v in qrels[t].items() if v == g] for g in (2, 1, 0)}
        chosen = []
        for g in (2, 1, 0):
            rng.shuffle(by[g])
            chosen += by[g][:per]
        leftovers = [n for g in (2, 1, 0) for n in by[g][per:]]
        chosen += leftovers[: EVAL_TRIALS_PER_PATIENT - len(chosen)]
        eval_set[t] = chosen
    return eval_set


def fetch_demo(conditions, per_condition=40):
    out = {}
    for cond in conditions:
        params = {"query.cond": cond, "query.locn": "Massachusetts", "filter.overallStatus": "RECRUITING",
                  "pageSize": per_condition}
        r = requests.get(CTGOV_SEARCH_URL, params=params, headers=HEADERS, timeout=60)
        r.raise_for_status()
        for s in r.json().get("studies", []):
            t = study_to_trial(s)
            if t["nct_id"] and t["eligibility_text"]:
                out[t["nct_id"]] = t
        print(f"demo: {cond!r} -> {len(out)} trials so far")
        time.sleep(0.5)
    (DATA_DIR / "demo_trials.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", nargs="*", default=None, help="conditions to fetch recruiting trials for the demo")
    args = ap.parse_args()
    DATA_DIR.mkdir(exist_ok=True)

    _, qrels = download_trec()
    eval_set = build_eval_set(qrels)

    trials_path = DATA_DIR / "trials.json"
    trials = json.loads(trials_path.read_text(encoding="utf-8")) if trials_path.exists() else {}
    needed = sorted({n for ids in eval_set.values() for n in ids} - set(trials))
    for nct in tqdm(needed, desc="fetching trials"):
        t = fetch_trial(nct)
        if t:
            trials[nct] = t
        time.sleep(0.2)
    trials_path.write_text(json.dumps(trials, indent=1), encoding="utf-8")

    eval_set = {t: [n for n in ids if n in trials] for t, ids in eval_set.items()}
    (DATA_DIR / "eval_set.json").write_text(json.dumps(eval_set, indent=1), encoding="utf-8")
    print(f"eval set: {len(eval_set)} patients, {sum(map(len, eval_set.values()))} pairs, {len(trials)} trials saved")

    if args.demo:
        fetch_demo(args.demo)


if __name__ == "__main__":
    main()
