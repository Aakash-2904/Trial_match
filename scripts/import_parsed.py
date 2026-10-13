"""Import LLM answers (data/batches/out_*.json) into data/parsed_api.json, with schema validation.

  python -m scripts.import_parsed

Handles markdown fences around the JSON and the UTF-16 files that PowerShell's '>' creates.
Trials that are missing or invalid keep their local parse; re-export and retry just those.
"""
import json
import re

from config import DATA_DIR
from schemas import ParsedTrial, TrialParseOutput


def read_text(path) -> str:
    raw = path.read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    return raw.decode("utf-8-sig")


def extract_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*|\s*```\s*$", "", text.strip(), flags=re.M)
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1])


def main():
    batch_dir = DATA_DIR / "batches"
    local = json.loads((DATA_DIR / "parsed_local.json").read_text(encoding="utf-8"))
    api_path = DATA_DIR / "parsed_api.json"
    merged = json.loads(api_path.read_text(encoding="utf-8")) if api_path.exists() else {}
    merged = {**local, **merged}

    ok, bad = 0, []
    for f in sorted(batch_dir.glob("out_*.json")):
        try:
            data = extract_json(read_text(f))
        except Exception as e:
            print(f"{f.name}: could not read JSON ({e}); re-run that batch")
            continue
        for nct, value in data.items():
            try:
                out = TrialParseOutput.model_validate(value)
                if not out.criteria:
                    raise ValueError("no criteria")
                merged[nct] = ParsedTrial(nct_id=nct, criteria=out.criteria, search_terms=out.search_terms,
                                          plain_summary=out.plain_summary, concepts=out.concepts,
                                          parser="api").model_dump()
                ok += 1
            except Exception as e:
                bad.append(nct)
                print(f"  {f.name} {nct}: invalid ({str(e)[:120]})")

    api_path.write_text(json.dumps(merged, indent=1), encoding="utf-8")
    eval_ids = set(json.loads((DATA_DIR / "trials.json").read_text(encoding="utf-8")))
    n_eval_api = sum(1 for k in eval_ids if merged.get(k, {}).get("parser") == "api")
    print(f"\nimported {ok} trials ({len(bad)} invalid)")
    print(f"evaluation trials with API parse: {n_eval_api}/{len(eval_ids)}")
    if n_eval_api < len(eval_ids):
        print("for the rest: run python -m scripts.export_batches --eval-only again (it exports only what is missing)")


if __name__ == "__main__":
    main()
