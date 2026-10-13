# Privacy-preserving multi-agent clinical trial matching

Given a free-text patient note, a team of agents finds matching clinical trials and explains,
criterion by criterion, why the patient is or is not eligible. Everything that touches patient data
runs locally through Ollama. A cloud API is used only to parse public trial registry text, and that
boundary is enforced in code (`llm_public.py` + `tests/test_privacy.py`).

Research prototype. Not for clinical use.

## Agents

| Agent | Tier | What it does |
|---|---|---|
| Patient profiler | local LLM | Note -> structured profile, every fact grounded in a quoted span |
| Criteria parser | cloud API (public text) | Eligibility text -> atomic criteria with numeric checks |
| Pre-filter | rules | BM25 relevance, age/sex from registry fields |
| Rule checker | rules | Evaluates numeric criteria (labs, age) with no LLM |
| Eligibility checker | local LLM | Judges the remaining fuzzy criteria |
| Critic | local LLM | Re-reads the note and challenges every exclusion |
| Scorer | rules | Label, score, explanation |

`supervisor.py` wires them together with LangGraph.

## RAG (all retrieval runs locally)

- **Hybrid trial retrieval** (`rag/hybrid.py`): BM25 + `nomic-embed-text` embeddings, fused with
  Reciprocal Rank Fusion, so "heart attack" still finds "myocardial infarction" trials.
- **Knowledge RAG** (`rag/knowledge.py`): concept cards (term, definition, members) written by the API
  from public trial text only, embedded locally, and retrieved per criterion for the eligibility checker.
  Cards are model-generated: audit a sample by hand before reporting results.

## Run it (Windows, Command Prompt)

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
ollama pull qwen2.5:3b
copy .env.example .env          (then add GEMINI_API_KEY if you want API parsing)
python -m pytest                (privacy and logic tests, no LLM needed)

python -m scripts.download_data --demo "heart failure" "type 2 diabetes" "breast cancer"
python -m scripts.parse_trials
ollama pull nomic-embed-text
python -m scripts.build_index
python -m scripts.run_eval --patients 2      (smoke test, ~20 min)
python -m scripts.run_eval                   (full run, several hours; leave overnight)
python -m dashboard.server                  (web dashboard: http://127.0.0.1:8501)
streamlit run app.py                         (original Streamlit demo, still works)
```

## Dashboard

`python -m dashboard.server` serves a local web UI (Starlette + plain HTML/JS, no build step, no CDN):

- **Screen a patient**: live agent pipeline with progress and ETA, ranked trial cards with a criteria
  breakdown, per-criterion evidence that highlights the exact span in the note, the extracted profile
  (hover a fact to see its source), filters, and JSON export.
- **Evaluation**: system comparison charts, a per-patient table, side-by-side rankings coloured by the TREC
  ground truth, significance tests, and a link from any saved run into the criteria view (no model calls).
- **How it works**: the privacy tiers and agents.

Privacy: it binds to 127.0.0.1 only, the page loads nothing from the internet (strict CSP), and patient
notes go only to the local model, as in `app.py`. One screening runs at a time. Deep links: `#eval`,
`#about`, `#replay=multiagent_4`.

## Evaluation

TREC 2021 Clinical Trials topics and qrels (2 eligible, 1 excluded, 0 not relevant).
Systems: BM25, single local LLM call, multi-agent, multi-agent without critic, multi-agent with RAG, multi-agent with the
local (non-API) parser. Metrics: nDCG@10, P@5, eligible-vs-excluded accuracy.
Results go to `results/summary.md` and `results/rankings.csv`.

## Known limitations

- TREC 2021 was judged on an April 2021 registry snapshot; trial text fetched today can differ.
- Small sample (default 10 patients x 12 trials) because inference runs on a laptop CPU.
- A 3B local model is weaker than cloud models; the study measures how much the agent design closes that gap.
