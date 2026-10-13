"""Rule-parse the demo trials that have no API parse yet and merge them into data/parsed_api.json.

  python -m scripts.parse_demo_rules            # writes parsed_api.json (backup: parsed_api.backup.json)
  python -m scripts.parse_demo_rules --dry-run  # only report what would change

Uses the deterministic rules in data/batches/smart_parse.py (criteria splitting, inclusion/exclusion, conservative
numeric checks, categories), the same rules behind the existing API parses. Search terms, plain summaries and
concept cards are not rule-derived, so the trial's current values are kept. Existing "api" entries are never
touched. Only public trial text is used.
"""
import argparse
import importlib.util
import json
import shutil

from config import DATA_DIR
from schemas import ParsedTrial


# Hand review of the rule-emitted numeric checks (the eval-trial checks were reviewed the same way). A wrong check
# makes the rule checker confidently wrong, and a check decides the WHOLE criterion, so these are dropped and the
# criterion goes to the LLM. IDs refer to smart_parse output before collapsing.
WEIGHT_CHANGE = "a change in weight, not body weight: weight_kg > 5 would exclude every adult"
CONDITIONAL = "applies only to a subgroup/cohort or is one of several alternatives, not a requirement for everyone"
PARTIAL = "the check covers only part of the criterion; the rest (diagnosis/procedure) would be ignored"
REVIEW_DROP = {
    ("NCT06677060", "I5"): CONDITIONAL + " ('at least one additional risk factor: Age >= 70')",
    ("NCT06677060", "I7"): CONDITIONAL + " ('at least one additional risk factor: eGFR < 60')",
    ("NCT03480633", "I9"): CONDITIONAL + " (HFpEF LVEF >= 50% vs HFrEF LVEF < 50%)",
    ("NCT05873634", "I10"): CONDITIONAL + " (LVEF >= 50% only for the HFpEF deep-phenotyping cohort)",
    ("NCT05873634", "I17"): CONDITIONAL + " (LVEF >= 50% only for the non-HFpEF deep-phenotyping cohort)",
    ("NCT07192952", "I3"): CONDITIONAL + " ('For newly enrolled infants <6 months': a condition, not an age limit)",
    ("NCT07662135", "I4"): CONDITIONAL + " (HbA1c range only for the SGLT2i-naive group)",
    ("NCT06525259", "I2"): CONDITIONAL + " (age 8-13 for girls, 9-15 for boys)",
    ("NCT06525259", "I3"): CONDITIONAL + " (age 8-13 for girls, 9-15 for boys)",
    ("NCT06112418", "I2"): CONDITIONAL + " (men > 55, women > 65)",
    ("NCT07654374", "E4"): WEIGHT_CHANGE,
    ("NCT07351045", "E2"): WEIGHT_CHANGE,
    ("NCT05799222", "E4"): WEIGHT_CHANGE + " ('weight loss of greater than 5 kg')",
    ("NCT05503797", "I1"): "wrong number: 'age >= 8 ... weighing >= 25 kg' was parsed as age >= 25",
    ("NCT05503797", "I8"): "wrong number: 'age >= 8 ... weighing >= 25 kg' was parsed as age >= 25",
    ("NCT05503797", "I20"): "wrong number: 'age >= 8 ... weighing >= 25 kg' was parsed as age >= 25",
    ("NCT05503797", "I27"): "wrong number: 'age >= 8 ... weighing >= 25 kg' was parsed as age >= 25",
    ("NCT07215715", "E17"): PARTIAL + " (HbA1c >= 9% *with target organ damage*)",
    ("NCT04986579", "I1"): PARTIAL + " (metastatic invasive breast cancer)",
    ("NCT05501704", "I1"): PARTIAL + " (invasive breast cancer, no surgical resection)",
    ("NCT06451757", "I2"): PARTIAL + " (multi-system primary mitochondrial disease)",
    ("NCT07559903", "I1"): PARTIAL + " (undergoing PCI for de novo CAD using DCB)",
}


def apply_review(nct, criteria, categorize):
    """Drop reviewed checks, collapse range pairs that lost both checks, renumber IDs per kind."""
    out, n = [], {"inclusion": 0, "exclusion": 0}
    for c in criteria:
        if (nct, c["id"]) in REVIEW_DROP:
            c = {**c, "check": None, "category": categorize(c["text"], []), "_dropped": True}
            if out and out[-1].get("_dropped") and (out[-1]["kind"], out[-1]["text"]) == (c["kind"], c["text"]):
                continue                             # second half of a split range: same text, now no check
        out.append(c)
    for c in out:
        c.pop("_dropped", None)
        n[c["kind"]] += 1
        c["id"] = f"{'I' if c['kind'] == 'inclusion' else 'E'}{n[c['kind']]}"
    return out


def load_smart_parse():
    spec = importlib.util.spec_from_file_location("smart_parse", DATA_DIR / "batches" / "smart_parse.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    sp = load_smart_parse()
    api_path = DATA_DIR / "parsed_api.json"
    before = api_path.read_text(encoding="utf-8")
    api = json.loads(before)
    demo = json.loads((DATA_DIR / "demo_trials.json").read_text(encoding="utf-8"))

    missing = [k for k in REVIEW_DROP if not any(c["id"] == k[1] and c["check"] for c in sp.parse(demo[k[0]]))]
    assert not missing, f"review entries no longer match a rule check: {missing}"

    updated, skipped, checks = {}, [], []
    for nct, trial in demo.items():
        old = api.get(nct, {})
        if old.get("parser") == "api":
            continue
        criteria = apply_review(nct, sp.parse(trial), sp.categorize)
        if not criteria:
            skipped.append(nct)                      # nothing the rules could split: keep the local parse
            continue
        entry = ParsedTrial(nct_id=nct, criteria=criteria, search_terms=old.get("search_terms", []),
                            plain_summary=old.get("plain_summary", ""), concepts=old.get("concepts", []),
                            parser="api").model_dump()
        updated[nct] = entry
        checks += [(nct, c["id"], c["check"], c["text"]) for c in entry["criteria"] if c["check"]]

    n_crit = sum(len(e["criteria"]) for e in updated.values())
    print(f"demo trials: {len(demo)}; rule-parsed: {len(updated)} ({n_crit} criteria, {len(checks)} numeric checks); "
          f"skipped (no criteria found): {len(skipped)} {skipped}; checks dropped in review: {len(REVIEW_DROP)}")
    for nct, cid, chk, text in checks:
        print(f"  {nct} {cid}: {chk['field']} {chk['op']} {chk['value']:g} {chk['unit'] or ''}  <- {text[:110]!r}")
    if args.dry_run:
        return

    backup = DATA_DIR / "parsed_api.backup.json"
    if not backup.exists():                          # keep the first, pre-merge version
        shutil.copyfile(api_path, backup)
    merged = {**api, **updated}
    api_path.write_text(json.dumps(merged, indent=1), encoding="utf-8")

    after = json.loads(api_path.read_text(encoding="utf-8"))
    untouched = [k for k in api if k not in updated]
    assert all(after[k] == api[k] for k in untouched), "an existing entry changed"
    print(f"wrote {api_path.name}: {len(after)} entries, {sum(v['parser'] == 'api' for v in after.values())} with "
          f"parser 'api'; {len(untouched)} existing entries unchanged; backup: {backup.name}")


if __name__ == "__main__":
    main()
