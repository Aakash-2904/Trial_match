"""Step 2b: build the local RAG indexes (needs Ollama running, no internet).

  ollama pull nomic-embed-text
  python -m scripts.build_index
"""
import json

from config import DATA_DIR
from rag.hybrid import build_trial_store
from rag.knowledge import build_knowledge_store, collect_cards
from schemas import ParsedTrial, Trial


def main():
    trials = {}
    for name in ("trials.json", "demo_trials.json"):
        p = DATA_DIR / name
        if p.exists():
            trials.update(json.loads(p.read_text(encoding="utf-8")))
    trials = [Trial(**v) for v in trials.values()]

    parsed = {}
    for name in ("parsed_local.json", "parsed_api.json"):
        p = DATA_DIR / name
        if p.exists():
            parsed.update({k: ParsedTrial(**v) for k, v in json.loads(p.read_text(encoding="utf-8")).items()})

    store = build_trial_store(trials, parsed)
    print(f"trial index: {len(store.ids)} trials embedded")
    kb = build_knowledge_store(parsed)
    print(f"knowledge base: {len(kb.ids)} concept cards (from {len(collect_cards(parsed))} unique terms)")
    if not kb.ids:
        print("no concept cards found: run scripts.parse_trials with a GEMINI_API_KEY to generate them")


if __name__ == "__main__":
    main()
