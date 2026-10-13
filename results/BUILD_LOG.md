# Build log

Code changes and decisions, one line each with the reason. Rules followed throughout: no prompt, threshold or
code tuning against qrels; eval set, metrics and `data/parsed_api.json` untouched; patient text only goes to local
Ollama models.

## Phase 1: profiler grounding (2026-10-10)

- `utils.py` `grounded()`: strip `[**...**]` placeholders and require at least 3 letters of real text; numbers must appear in the source. Reason: hallucinated facts were "grounded" by de-identification tags.
- `utils.py` `grounded()`: exact match now uses word boundaries. Reason: `AST` was accepted because the note contains "p**ast** medical history".
- `utils.py` `supports()` + `NEGATION`: fact text and evidence must share a meaningful word. Reason: the model attached unrelated evidence lines to facts.
- `utils.py` `negates()`: a negative is kept only if a fact word is within 6 words of a negation cue in the same clause (NegEx-style scope). Reason: "Negative for Burkitt's lymphoma" passed because a *different* phrase in the span was negated; it contradicted the patient's main diagnosis.
- `agents/profiler.py`: new prompt (main diagnosis goes in `conditions`, negatives only for explicit denials) and `ground_profile()`. Reason: the main diagnosis was missed.
- `agents/llm_checker.py`: note before profile, note is the source of truth, and evidence is quoted from the note. Reason: the profile can be incomplete.
- `supervisor.py`: the profile summary line counts procedures, negatives and other.
- `scripts/debug_profile.py`, `tests/test_logic.py`: debug script and 7 grounding tests.

Phase 1 changes were checked only by reading patient 4's note and the raw model output, never qrels.

## Phase 2: smoke test (2026-10-10)

- First run: patient 4 completed for all systems with non-empty profiles. Patient 5 hung for 13+ min in the profiler call: `llama-server` was busy, but no output came back.
- `llm_local.py` + `config.py`: added `num_predict` cap (`LOCAL_MAX_TOKENS`, default 2048). If a response is cut off at the cap, retry once from the original prompt with `repeat_penalty=1.3`, and print a log line. Reason: greedy constrained-JSON decoding can repeat list items forever (patient 4's raw profile already had about 30 repeated lab names in `other`). Calls under the cap give the same output, and the cache key ignores options, so cached answers stay valid.
- `agents/profiler.py` `ground_profile()`: facts in conditions/medications/procedures/other whose own text is a negation ("No prior syncope") move to `negatives`, which are deduplicated. Reason: patient 5's profile listed denials as conditions, so the checkers could read them as diagnoses. The fact text is checked, not the evidence, so "Prior stroke ... without residual deficit" stays a condition.
- `agents/profiler.py` `_fallback_age/_fallback_sex`: accept clinical shorthand "74M" / "62 F". Reason: patient 5's note starts "74M"; the model returned no age, and the fallback missed it.
- `tests/test_logic.py`: test for the two fixes above.
- Re-ran the smoke test with the same command. It passed (exit 0); both profiles are non-empty (patient 4: 8 conditions; patient 5: age 74, male, 22 conditions). One checker call hit the token cap and recovered on retry. Table saved as `summary_smoke_p2.md`.

## Phase 3: evaluation, time-boxed (2026-10-10)

- Decision (user request): no full 10-patient run. Measured cost on this CPU is about 17–37 min per patient for `multiagent`, so a full run would take many hours. The time budget is 2 hours, ending about 18:25.
- `results/run_batches.sh`: runs batches in sequence, logs to `results/eval_log.txt`, and copies `summary.md/json` and `rankings.csv` to `*_batch_<name>.*` after each batch. Reason: a stopped run still leaves complete tables. No code changes in the evaluation itself.
- Batch order, chosen for coverage of the report's comparisons, not by looking at labels: (B) all 5 systems on the first 2 eval patients, where only `multiagent_rag` is new work; then (A) the 4 core systems on the first 3 patients. `local_parse` is skipped for lack of time.

- Batch B (all 5 systems, patients 4 and 5) finished 16:28–17:15 and was saved as `*_batch_B_all5_p2.*`. One checker call hit the token cap and recovered.
- Batch A (core systems, adding patient 12) was stopped at about 17:45 because system memory was critically low (0.4 GB of 7.9 GB free). It reached patient 12's `multiagent` trial 9 of 12. The shell was killed, but the Python eval process survived as an orphan; I stopped it too, as the stop intended. Not restarted, per the stop notice. Patient 12 has no metrics; its 9 finished trials are cached.
- The report uses Batch B only.

## Significance tests (user request, 2026-10-10)

- `scripts/significance.py`: per-patient nDCG@10 differences for 4 pairs, paired bootstrap 95% CI (10,000 resamples, seed 42), exact two-sided Wilcoxon signed-rank p, win/tie/loss → `results/significance.md`. Reason: the user asked for paired tests. It merges `rankings.csv` with `rankings_batch_*.csv` because the RAG pair and the core pairs come from different batches, and each pair uses only patients both systems ran on.
- Exact Wilcoxon written in plain Python (enumerates all sign patterns). Reason: scipy isn't installed; it's exact and instant for n ≤ ~20, and avoids a new dependency.
- Verdict rule: "X better" only if p < 0.05 and the CI excludes 0; otherwise "no clear difference". Note: with n=2 the smallest possible p is 0.5 and with n=3 it is 0.25, so nothing can reach significance at this sample size.
- `tests/test_logic.py`: Wilcoxon checked against hand-computed exact p-values. Checked the script's mean differences against `summary.json` from the smoke run; they match.

## Phase 4: write-up (2026-10-10)

- `results/REPORT.md`: results table (Batch B), per-patient table, significance table, interpretation of the 3 comparisons, 3 error-analysis cases, limitations. `results/significance.md` was regenerated from `rankings_batch_B_all5_p2.csv`.
- Error-analysis cases were picked from the results JSON with qrels, for the write-up only; nothing was changed because of them. Two design issues found are reported, not fixed (fixing them now would mean tuning on test labels): (1) an LLM "no" without evidence skips grounding and hard-excludes the trial; (2) the 50/50 relevance/eligibility blend keeps high-BM25 ineligible trials near the top.

## Demo-trial rule parse (user request, 2026-10-10)

- `scripts/parse_demo_rules.py`: rule-parses the 119 demo trials with `data/batches/smart_parse.py` and merges them into `data/parsed_api.json` with `parser: "api"`, as the user asked. Reason: demo trials had only the basic local parse, so the rule checker decided 0 criteria for them. Before running, I checked that `smart_parse.parse()` reproduces all 118 existing API parses exactly.
- The 118 existing API entries are byte-identical after the merge (the script asserts this); backup at `data/parsed_api.backup.json`. Demo search terms, plain summaries and concept cards are kept as they were, because the rules don't produce them (PARSING_NOTES.md: those came from an LLM, not rules).
- Hand review of the 128 rule-emitted checks, as was done for the eval trials: 22 dropped (listed with reasons in `REVIEW_DROP`). Those criteria keep their text and go to the LLM. Reasons: weight *change* parsed as body weight (would exclude every adult), "≥8 years ... ≥25 kg" parsed as age ≥ 25, cohort- or sex-specific or one-of thresholds turned into hard requirements, and checks covering only part of a criterion (a check decides the whole criterion in `rule_checker`). 106 checks remain, in 60 of 119 demo trials. `smart_parse.py` itself is unchanged.
- `python -m scripts.build_index`: rebuilt (237 trials, 44 cards). The indexes are identical to before (backup in `data/index_backup_20261010/`), because index text uses title, conditions, search terms and summary, not criteria. Earlier `multiagent_rag` results remain valid.
- Note: the `api` label now covers rule-parsed demo entries that, unlike the eval entries, have no search terms or concept cards.

## Dev/test split (user decision, 2026-10-10)

- Patients 4 and 5 are now the DEVELOPMENT set; all other eval patients (8) are the HELD-OUT TEST set. The fixes below come from code review, dev patients 4-5, and a demo checklist. None comes from test-set results or qrels (`data/qrels2021.json` is now read-denied for me; only `run_eval` reads it).
- Patient 12 was partly run earlier (9 of 12 `multiagent` trials, before the fixes). Its metrics were never computed or looked at. It stays in the test set; its cached checker answers are reused only where prompts are unchanged.

## Phase 1: design fixes (2026-10-10)

Each fix has a unit test in `tests/test_design_fixes.py` (13 tests; 36 in total pass).
1. `agents/llm_checker.py`, `agents/critic.py`: any LLM judgment that makes a criterion not_met must have evidence that passes `grounded()` against the note; otherwise it becomes "unknown" (source `grounding`). The critic schema gets an `evidence` field. Reason: an unsupported "no" hard-excluded dev patient 4's NCT00944567.
2. Same check for "met": every LLM yes/no needs grounded evidence. Reason: a decision without a quote can't be checked either way. Rule-checker results are not affected (they come from the grounded profile).
3. `agents/critic.py`: new prompt. Overturn to the value the note supports, with the MI-3-years-ago example; "unknown" only when the note is silent. A critic yes/no also needs grounded evidence. Reason: the old prompt said "when in doubt, overturn to unknown" (dev patient 5, NCT01685567).
4. `agents/scorer.py` `rank_key`, `supervisor.py`: sort by tier (likely_eligible, possibly_eligible, likely_ineligible), then relevance within a tier. `final_score` is still computed and shown, but no longer orders results. Reason: high-BM25 ineligible trials ranked above eligible ones (dev patient 4, NCT02858661).
5. `agents/groups.py`, applied in `scorer.score`: either/or groups count as one criterion. Generalised to k-of-n because some headings say "at least two of the following" (SIRS); k=1 is exactly the requested rule. Truth is computed on the group statement, so it also works for exclusion groups ("any of the following" → excluded if any option is true). If k exceeds the number of options found (parsing can drop long options), the group is never "no". Two parse shapes are handled: "heading: option" prefixes (rule/API parse) and heading criteria ending in ":" (local parse; options run to the next heading).
6. `rag/knowledge.py` `expand_abbreviations`: CV, HF, MI, HTN, T2DM and CKD get their long forms (whole words only) before card lookup, for both exact and dense matching.
7. `agents/llm_checker.py`: the knowledge section is labelled "background only: general definitions, NOT facts about this patient; never quote it as evidence". The name "MEDICAL KNOWLEDGE" and the system prompt stay the same, so non-RAG checker prompts are byte-identical and their cached answers stay valid. Reason: RAG had more than twice as many grounding rejections as non-RAG on dev (47 vs 19).
8. `schemas.Criterion` validator + `utils.clean_markdown`: drop markdown escapes (`\<`, `\>`, `\*`, ...) from criterion text on every parse path. API parses were already clean (smart_parse strips them). The local parse had 669 escaped criteria. The `single_call` baseline sends raw eligibility text (66 eval trials contain escapes); that text is not criterion text and is left unchanged as specified, so the baseline cache stays valid.
9. `scripts/run_eval.py`: `--exclude` topic IDs, via a testable `select_topics()`; unknown IDs are rejected.

Cache impact: non-RAG checker prompts are unchanged (fixes 1, 2, 4, 5 are post-processing). Critic and RAG prompts changed and re-run.

## Phase 2: demo trials (2026-10-10)

- Already done for the previous request (`scripts/parse_demo_rules.py`, logged above). Re-running it found 0 trials left to parse, and `parsed_api.json` stayed byte-identical (same md5). `build_index` was re-run: 237 trials, 44 cards.

## Phase 3: dev check (2026-10-10)

- Pre-fix dev outputs copied to `results/dev_before/` (Batch B: `summary_batch_B_all5_p2.*`), because `run_eval` overwrites the per-trial JSON files.
- Before starting: no leftover `run_eval`/`run_batches` processes, no Ollama model loaded, 0.8 GB of 7.9 GB RAM free. The remaining Python processes are the user's VS Code Jupyter kernels and Streamlit app (~30 MB in total). I did not stop them, because stopping a kernel loses notebook state and they aren't leftovers from this work. Risk: low RAM may get the background job stopped again.
- Command: `python -m scripts.run_eval --patients 2 --systems bm25 single_call multiagent no_critic multiagent_rag` → `results/dev_log.txt`, then copied to `summary_dev_after.*` / `rankings_dev_after.csv`.
- Completed with exit 0, no crashes, no token-cap hits. No code changes in Phase 3.

Dev results (patients 4 and 5), before → after the Phase 1 fixes:

| System | nDCG@10 | P@5 | Eligible-vs-excluded acc. |
|---|---|---|---|
| bm25 | 0.685 → 0.685 | 0.300 → 0.300 | n/a |
| multiagent | 0.931 → 0.816 | 0.600 → 0.500 | 0.812 → 0.500 |
| no_critic | 0.871 → 0.816 | 0.600 → 0.500 | 0.812 → 0.562 |
| multiagent_rag | 0.916 → 0.954 | 0.700 → 0.500 | 0.625 → 0.500 |
| single_call | 0.967 → 0.967 | 0.700 → 0.700 | 0.812 → 0.812 (unchanged code path, cached) |

Why (dev analysis, no code changes): the grounded-evidence rule (fixes 1-2) turned most LLM decisions into "unknown". On patient 4, 58 of 75 grounding rejections had no evidence quoted at all, and the rest quoted the criterion back or used text not in the note. That is correct behaviour, but the 3B model rarely quotes. Across both dev patients, LLM met/not_met went 69/19 → 9/0 (multiagent), 69/32 → 9/1 (no_critic) and 53/13 → 9/1 (RAG). 19-20 of 24 trials now sit in `possibly_eligible`, so tier-first ranking (fix 4) orders most trials by relevance alone. `multiagent`/`no_critic` therefore fall toward the BM25-prefilter ordering, and `multiagent_rag` rises because its hybrid relevance orders that tier better. Accuracy drops because excluded trials are no longer excluded. The critic changed one decision. Per the instructions, nothing was tuned in response.

## Phase 4: FREEZE (2026-10-10)

- Code frozen before the held-out run. SHA-256 of every project .py file is in `results/FREEZE.txt`. After this point: crash fixes only, each logged here.
- Pre-flight: no leftover eval processes (0 stopped), Ollama models unloaded, 2.5 GB of 7.9 GB RAM free. The user's Jupyter kernels and Streamlit app were left running.
- Started `python -m scripts.run_eval --exclude 4 5 --systems bm25 single_call multiagent no_critic multiagent_rag` → `results/eval_log.txt` (the old batch log was renamed `eval_log_batches_old.txt`). On success, summary and rankings are copied to `summary_heldout.*` / `rankings_heldout.csv`. The test-set metrics are not looked at until the write-up.
- The run was stopped at about 22:33 because system memory was critically low while the session was idle. It was on held-out patient 1 of 8 (topic 12), after `multiagent` trial 11 of 12. Progress from 20:23 to 22:32 was only 2 trials, consistent with the laptop sleeping while the user was away. No orphaned process was left this time. Not restarted, per the stop notice; waiting for the user. All finished calls are cached, so the same command resumes. No code changes.
- 22:43: the user said "continue". Restarted the same command (resumes from the cache), with 0.84 GB RAM free. The killed run's log is kept as `eval_log_run1_killed.txt`. The user then went to sleep and delegated all decisions.

## Dashboard (added during Phase 4, 2026-10-10, outside the frozen code)

- At the user's request: a new web dashboard replacing Streamlit as the main UI, in `dashboard/` (`server.py` plus `static/`), with tests in `tests/test_dashboard.py`. `app.py` is kept.
- It does not change any frozen file: `sha256sum -c` on `FREEZE.txt` passes. It imports the agents and reads `results/` without writing to them. It cannot affect the held-out run.
- Built and tested without the LLM so the eval run kept the CPU: agent graphs stubbed in tests and for screenshots. Its eval metrics are recomputed from the rankings CSVs and checked against the `summary_*.json` files from `run_eval` (test passes, within 1e-3). 41 tests pass.
- A real end-to-end screening through the dashboard is done after the held-out run finishes, so it doesn't compete for the model.

## Phase 4, second stop (2026-10-10 23:29)

- The resumed held-out run was stopped again for critically low memory (0.55 GB free at 23:31, 3.9 GB in Memory Compression). It was on held-out patient 2 of 8 (topic 14), with `multiagent` 12/12 trials done. Topic 12 is complete for all five systems. No orphaned process. I unloaded the Ollama model to free about 0.5 GB.
- Not restarted, per the stop notice, which says to restart only when asked. Resume with the same command: everything finished is cached.
- Dashboard work during this run: no LLM calls. Light Python servers and headless Edge screenshots used ~0.2–0.4 GB for a few minutes, all stopped by 23:00, about 30 minutes before the stop.
- 2026-10-11 05:03: restarted at the user's request, with 2.33 GB RAM free. The previous log is kept as `eval_log_run2_killed.txt`.
- 2026-10-11 06:06: stopped for low memory a third time (0.32 GB free; llama-server 2.0 GB). Topics 12, 14, 15 and 18 are complete; topic 29 (patient 5 of 8) was in progress. No orphaned process. Ollama models unloaded. Not restarted, per the stop notice.
- 2026-10-11: restarted at the user's request ("restart it here again"). The previous log is kept as `eval_log_run3_killed.txt`.
- 2026-10-11 06:16: the run ended when the previous terminal session closed (not a memory stop), on topic 29 after trial 2/12. Log kept as `eval_log_run4_session_end.txt`.
- New memory rules from the user: pre-flight with ≥2.5 GB free required, checks every 15 min during runs, and stop only project leftovers. Pre-flight at 06:22: 1.27 GB free, with no project python processes but the idle Ollama model loaded (2.2 GB). Ran `ollama stop qwen2.5:3b` → 2.87 GB free. Top non-project users: Chrome (~590 MB in 9 processes), VS Code (~200 MB), Defender (~190 MB); none touched. Restarted the same command from the cache.
- 2026-10-12 01:04: the restarted run stopped writing (cause not recorded; no process left, no Ollama model loaded). Topics 12, 14, 15, 18 complete; topic 29 (patient 5 of 8) has bm25, single_call, multiagent and no_critic done and `multiagent_rag` at trial 3/12. Topics 30+ (patients 6-8) not started. All finished calls are cached.
- 2026-10-12 10:44 pre-flight for the resume: 0.93 GB free (below the 2.5 GB rule), no project python processes, no Ollama model loaded. Not started. Largest non-project users: VS Code (~1.0 GB in 3 processes), Chrome (~550 MB), Defender (~260 MB), Explorer (~180 MB), WhatsApp (~175 MB). Waiting for the user.
- 2026-10-12: the user closed apps but free memory stayed at 1.2-2.1 GB (WhatsApp/Phone Link/Chrome kept relaunching). The user chose to run anyway. A detached launch outside VS Code was started and stopped after ~20 s when the user asked to run it inside the terminal instead (log kept as `eval_log_run6_detached_stopped.txt`; the old log is `eval_log_run5_stopped.txt`). The idle `nomic-embed-text` model was unloaded. Restarted the same command as a background job of the terminal session at ~1.8 GB free (below the 2.5 GB rule, by user decision). No code changes.
- 2026-10-12 (first check after restart): 0.76 GB free, below the 800 MB threshold, with qwen2.5:3b loaded (2.2 GB). The only python processes are the active eval, so nothing was stopped. WhatsApp (~140 MB), Phone Link (~180 MB) and CrossDeviceService (~130 MB) were running again although the user had closed them. User informed.
- 2026-10-12 13:03: 0.75 GB free (below 800 MB). WhatsApp and Phone Link now closed by the user; CrossDeviceService (~145 MB) still running. Both Ollama models (qwen2.5:3b and nomic-embed-text) are in use by the active multiagent_rag run, so nothing stopped. Topic 29 multiagent_rag reached trial 4/12.
- 2026-10-12 ~13:15: the user asked about scores mid-run, was reminded of the no-peek rule, and chose to wait. Held-out metrics still not viewed. Topic 29 complete; patients 6-8 remaining.
- 2026-10-12 ~13:55: the eval was stopped for critically low memory (0.25 GB free at 13:57). Topics 12, 14, 15, 18, 29, 32 complete; topic 36 (patient 7/8) has bm25, multiagent, no_critic done and multiagent_rag just started (0 trials). Topic 8/8 not started. No orphaned process. Both Ollama models unloaded → 3.26 GB free. Log kept as `eval_log_run7_memstop.txt`. Not restarted, per the stop notice; waiting for the user.
- 2026-10-12: restarted at the user's request. Pre-flight: 3.53 GB free, no project python processes, no Ollama models loaded. CrossDeviceService (~50 MB) back but small. Same command, resumes from the cache.
- 2026-10-12: 0.59 GB free (below 800 MB) at the start of topic 56 (patient 8/8); Memory Compression 1.6 GB. Only the active eval's python processes are running, so nothing stopped. User informed.
- 2026-10-12 ~14:35: the eval was stopped again for critically low memory. Topics 12-36 (patients 1-7) complete for all systems; topic 56 (patient 8/8) has bm25 done and the next LLM system at trial 9/12. No orphaned process. Ollama models unloaded → 3.4 GB free. Log kept as `eval_log_run8_memstop.txt`. Not restarted, per the stop notice.
- 2026-10-12: restarted at the user's request. Pre-flight: 2.59 GB free, no project python processes, no Ollama models loaded; Chrome (~540 MB) open again. Same command, resumes from the cache.
- 2026-10-12 14:56: 0.76 GB free (below 800 MB) on topic 56; only the active eval's python processes are running, nothing stopped. Chrome (~180 MB+) open. User informed.
- 2026-10-12 15:07: held-out run completed, exit 0, no tracebacks. summary_heldout.{json,md} and rankings_heldout.csv written. FREEZE.txt hashes still match. Ollama models unloaded. Phase 4 done.
- 2026-10-12 15:10: Phase 5 started. `python -m scripts.significance results/rankings_heldout.csv` → `results/significance_heldout.md` (previous significance.md kept as `significance_pre_heldout.md`). No comparison reaches p < 0.05 with 8 patients.

## Phase 5: held-out write-up (2026-10-12)

- `results/REPORT.md` rewritten for the held-out test set: protocol, results, per-patient table, significance, dev vs test, decision-source breakdown, interpretation, 3 error cases (all held-out eligible trials that multiagent labelled likely ineligible), limitations. The old 2-patient pilot report is kept as `REPORT_pilot_2patients.md`.
- Decision counts come from `<system>_<topic>.json`, labels vs qrels from `rankings_heldout.csv`. Analysis only; no code changes, `FREEZE.txt` still matches.
