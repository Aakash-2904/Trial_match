"""Export PUBLIC trial text into batch prompt files for parsing with any LLM (Gemini CLI, Gemini web, Claude).

  python -m scripts.export_batches --eval-only          # ~120 evaluation trials, 8 per file
  python -m scripts.export_batches --size 5             # smaller batches if answers get cut off

Writes data/batches/batch_01.md, batch_02.md, ...  Only trial registry text is exported:
never share topics2021.json (patient descriptions) or anything in results/.
"""
import argparse
import json

from agents.criteria_parser import INSTRUCTION
from config import DATA_DIR
from schemas import Trial, TrialParseOutput

OUTPUT_RULES = """
OUTPUT FORMAT (important):
Return ONE JSON object and nothing else: no markdown, no explanation.
Its keys are the nct_id values below. Each value follows this schema:
{schema}

Example shape:
{{"NCT01234567": {{"criteria": [...], "search_terms": [...], "plain_summary": "...", "concepts": [...]}},
  "NCT07654321": {{ ... }}}}
Include every trial below, even if its criteria are short.
"""


def load(name):
    p = DATA_DIR / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--size", type=int, default=8)
    args = ap.parse_args()

    eval_trials, demo_trials = load("trials.json"), load("demo_trials.json")
    pool = {**demo_trials, **eval_trials} if not args.eval_only else eval_trials
    order = list(eval_trials) + [k for k in pool if k not in eval_trials]
    done = load("parsed_api.json")
    todo = [k for k in order if k in pool and done.get(k, {}).get("parser") != "api"]

    out_dir = DATA_DIR / "batches"
    out_dir.mkdir(exist_ok=True)
    for old in out_dir.glob("batch_*.md"):
        old.unlink()

    schema = json.dumps(TrialParseOutput.model_json_schema(), separators=(",", ":"))
    header = INSTRUCTION + "\n" + OUTPUT_RULES.format(schema=schema)
    n_files = 0
    for start in range(0, len(todo), args.size):
        chunk = todo[start:start + args.size]
        trials = [Trial(**pool[k]).public().model_dump() for k in chunk]   # PUBLIC fields only
        n_files += 1
        (out_dir / f"batch_{n_files:02d}.md").write_text(
            header + "\nTRIALS:\n" + json.dumps(trials, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(todo)} trials -> {n_files} batch files in {out_dir}")
    print("Save each answer as out_XX.json next to its batch file, then run: python -m scripts.import_parsed")


if __name__ == "__main__":
    main()
