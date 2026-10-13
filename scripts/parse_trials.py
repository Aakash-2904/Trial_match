"""Step 2: parse every saved trial's criteria. API parse uses PUBLIC trial text only; local parse is the fallback.

  python -m scripts.parse_trials            # API if GEMINI_API_KEY is set, plus local parse
  python -m scripts.parse_trials --local-only
"""
import argparse
import json

from tqdm import tqdm

import llm_public
from agents import criteria_parser
from config import DATA_DIR
from schemas import Trial


def load_trials():
    trials = {}
    for name in ("trials.json", "demo_trials.json"):
        p = DATA_DIR / name
        if p.exists():
            trials.update(json.loads(p.read_text(encoding="utf-8")))
    return {k: Trial(**v) for k, v in trials.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-only", action="store_true")
    args = ap.parse_args()
    trials = load_trials()

    local = {k: criteria_parser.parse_locally(t).model_dump() for k, t in trials.items()}
    (DATA_DIR / "parsed_local.json").write_text(json.dumps(local, indent=1), encoding="utf-8")
    print(f"local parse: {len(local)} trials")

    if args.local_only or not llm_public.available():
        print("skipping API parse (no key or --local-only)")
        return
    api = {}
    for k, t in tqdm(trials.items(), desc="API parse (public trial text only)"):
        api[k] = criteria_parser.parse(t, mode="api").model_dump()
    (DATA_DIR / "parsed_api.json").write_text(json.dumps(api, indent=1), encoding="utf-8")
    n_api = sum(v["parser"] == "api" for v in api.values())
    print(f"API parse: {n_api}/{len(api)} trials (rest fell back to local)")


if __name__ == "__main__":
    main()
