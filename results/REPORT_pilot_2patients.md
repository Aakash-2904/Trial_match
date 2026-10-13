# TrialMatch evaluation report

Local multi-agent clinical trial matching (LangGraph + Ollama `qwen2.5:3b`, CPU only) on TREC Clinical Trials
2021, run on 2026-10-10. Patient notes went only to local models. Build decisions and every code change are in
`BUILD_LOG.md`.

> **Read this first: only 2 patients (24 patient–trial pairs).** The planned 10-patient evaluation was cut to a
> 2-hour budget (CPU cost: about 20–37 min per patient per LLM system). A third patient (topic 12) stopped at
> trial 9 of 12 when the run was stopped because the machine was critically low on memory, so it has
> no metrics. Every comparison below is **no clear difference** statistically; with 2 patients, no test can
> reach p < 0.05.

## Results

Patients 4 and 5 (the first two of the eval set), 12 candidate trials each, all five systems on the same pairs
(`summary_batch_B_all5_p2.md`, identical to `summary.md`):

| System | nDCG@10 | P@5 | Eligible-vs-excluded acc. |
|---|---|---|---|
| bm25 | 0.685 | 0.300 | n/a |
| multiagent | 0.931 | 0.600 | 0.812 |
| no_critic | 0.871 | 0.600 | 0.812 |
| multiagent_rag | 0.916 | 0.700 | 0.625 |
| single_call | 0.967 | 0.700 | 0.812 |

Per patient (from `eval_log.txt`):

| System | P4 nDCG@10 | P4 P@5 | P4 acc. | P5 nDCG@10 | P5 P@5 | P5 acc. | Time / patient |
|---|---|---|---|---|---|---|---|
| bm25 | 0.557 | 0.20 | n/a | 0.812 | 0.40 | n/a | <1 s |
| multiagent | 0.903 | 0.60 | 0.88 | 0.959 | 0.60 | 0.75 | 17–37 min |
| no_critic | 0.903 | 0.60 | 0.88 | 0.838 | 0.60 | 0.75 | ~0 (reuses cache) |
| multiagent_rag | 0.894 | 0.80 | 0.75 | 0.938 | 0.60 | 0.50 | 21–25 min |
| single_call | 0.989 | 0.60 | 0.75 | 0.945 | 0.80 | 0.88 | ~6 min |

Accuracy is computed only over trials judged relevant (qrel 1 or 2): 8 such trials per patient, 16 in total.
One trial is 0.125 per patient, or 0.0625 overall.

### Significance (`significance.md`, per-patient nDCG@10)

Difference = first system minus second. CI: paired bootstrap over patients (10,000 resamples, seed 42).
p: exact two-sided Wilcoxon signed-rank. A difference counts only if p < 0.05 and the CI excludes 0.

| Comparison | Patients | Mean diff | 95% CI | Wilcoxon p | Win/tie/loss | Verdict |
|---|---|---|---|---|---|---|
| multiagent vs single_call | 2 | −0.035 | [−0.086, +0.015] | 1.000 | 1/0/1 | no clear difference |
| multiagent vs no_critic | 2 | +0.061 | [+0.000, +0.121] | 1.000 | 1/1/0 | no clear difference |
| multiagent_rag vs multiagent | 2 | −0.015 | [−0.021, −0.009] | 0.500 | 0/0/2 | no clear difference |
| multiagent vs bm25 | 2 | +0.246 | [+0.147, +0.346] | 0.500 | 2/0/0 | no clear difference |

With 2 patients the smallest possible two-sided Wilcoxon p is 0.5. The bootstrap "CI" only spans the two
observed differences, so it reflects the observed range, not real uncertainty. Even a 2/0/0 record (vs bm25) is
not evidence of a real difference at this sample size.

## Interpretation

**Multi-agent vs single call: no clear difference.** The single call scored slightly higher on mean nDCG@10
(0.967 vs 0.931) and P@5 (0.70 vs 0.60), with equal accuracy (0.812), and the two split 1–1 by patient. There
is no sign that the multi-agent pipeline beats one well-prompted call to the same 3B model. It costs about
3–6× more CPU time per patient. Its advantage is transparency: per-criterion outcomes, evidence quotes and an
explanation for every trial. It did not improve ranking here.

**Critic vs no critic: no clear difference (identical on patient 4, +0.121 on patient 5).** The critic only
reviews LLM "not met" calls. Across both patients it overturned 13 of 32, all to "unknown". On patient 5 that
moved four trials from likely-ineligible to possibly-eligible. TREC judged two of them eligible and two
excluded. That raised nDCG@10, but accuracy was unchanged (0.75): it gained two eligible trials and lost two
excluded ones. The critic's built-in leniency ("when in doubt, overturn to unknown") is a recall-
for-precision trade, not a correctness check.

**RAG vs no RAG: no clear difference (−0.015, lower on both patients); accuracy 0.625 vs 0.812.** Two effects:
1. Hybrid (BM25 + dense, RRF) relevance scores are compressed into 0.86–1.00 versus 0.18–1.00 for the BM25
   prefilter, so relevance barely separates trials and the eligibility score decides the ranking.
2. With retrieved knowledge in the prompt, 47 checker answers were rejected by the grounding guard for
   evidence not in the note, versus 19 without RAG. This is consistent with the model quoting the knowledge
   cards instead of the note. Rejected answers become "unknown", so RAG produced fewer LLM "not met" calls
   (27 vs 32 before the critic, 13 vs 19 after), and three TREC-"excluded" trials became "possibly eligible". That loss accounts for most
   of the accuracy drop.
P@5 was higher with RAG on patient 4 (0.80 vs 0.60), so the effect is not one-directional.

**Multi-agent vs BM25:** higher on both patients (+0.346, +0.147), but BM25 alone is the weakest baseline,
and with n=2 this is still no clear difference statistically.

## Error analysis

Taken from `multiagent_*.json`, `no_critic_*.json` and `rankings_batch_B_all5_p2.csv`. These cases were used
only for this write-up; none led to code, prompt or threshold changes.

**1. Hard exclusion from an unsupported "no" (patient 4, NCT00944567, TREC: eligible).** A pathology study of
primary mediastinal B-cell lymphoma, for a 44-year-old with primary cardiac Burkitt's lymphoma. The checker
answered "no" to the inclusion criterion "Primary mediastinal diffuse large B-cell lymphoma, CD20 positive…"
with **no evidence**. Grounding only checks evidence when some is given, so the unsupported "no" passed. One
`not_met` makes a trial `likely_ineligible` with eligibility 0. The critic returned no decision for this
criterion, so the "no" stood. The literal
reading is defensible (Burkitt's is not PMBCL), but TREC judged the patient eligible, and the design turns any
single unsupported "no" into a hard exclusion. 19 of 357 criteria were LLM "not met" in the final multiagent
output.

**2. Critic rescue, but the real answer was missed (patient 5, NCT01685567, TREC: eligible).** A carotid-
stenting study for a 74-year-old man whose note reports "80% stenosis of the R ICA and a 90% L ICA stenosis"
and a stroke 10–15 years ago without residual deficit. The checker answered "no" with no evidence to both
alternatives, "Symptomatic: stenosis ≥ 50%…" and "Asymptomatic: stenosis ≥ 70%…", so `no_critic` excluded the
trial. The critic overturned both to "unknown" ("the note does not mention any neurological symptoms within the
prior 180 days"), and the trial became `possibly_eligible`. But the note arguably satisfies the asymptomatic
≥70% criterion, so the right answer was "met", not "unknown". Neither variant reads numeric stenosis values
against criterion thresholds; the rule checker only handles canonical lab fields.

**3. Ranking blends relevance with ineligibility (patient 4, NCT02858661, TREC: not relevant).** A study on
diagnosing meningitis in emergency rooms. The note mentions "viral encephalitis/meningitis" only in past
history, so BM25 relevance is 1.00. The pipeline correctly labeled it `likely_ineligible` (eligibility 0), but
`final_score = 0.5 × relevance + 0.5 × eligibility` = 0.50 still ranked it 4th. It sits above the
TREC-eligible NCT00944567 (0.33, case 1) and all four TREC-excluded trials. Ineligible trials with high lexical
overlap keep up to half the score, which limits how much eligibility reasoning can improve nDCG.

## Limitations

- **Tiny sample.** 2 patients, 24 patient–trial pairs, 16 relevance-judged pairs for accuracy. Nothing is
  statistically significant, and one trial moves accuracy by 6 points. Patient selection was simply the first
  two eval-set topics, not chosen, but two patients cannot represent the 75 TREC 2021 topics. `local_parse` was
  not run.
- **3B model on CPU.** `qwen2.5:3b` (Q4) answered "unknown" for 55% of criteria (198 of 357) and sometimes
  produced repeated JSON. Two calls hit the new 2,048-token cap and were retried with a repeat penalty. Results
  with a larger model may differ in either direction, and greedy decoding means one sample per call.
- **2021 judgments vs current trial text.** Qrels were made in 2021 against the trial text of that time. Trial
  text was later fetched from the current ClinicalTrials.gov API (`scripts/download_data.py`), and criteria may have been amended since, so some
  "disagreements" may be label drift, not model errors.
- **Hybrid rule/LLM criteria parsing.** Criteria were pre-parsed from public trial text (`parsed_api.json`).
  Only canonical numeric lab checks are decided by rules (19 of 357 criteria); everything else goes to the
  local LLM. Free-text thresholds such as "stenosis ≥ 70%" (case 2) are not checked numerically. Trials with
  more than 30 fuzzy criteria have the rest marked "unknown" (20 criteria here).
- **Development used the same patients.** The profiler fixes in Phase 1–2 were developed by reading patients 4
  and 5's notes and raw model output. Qrels were never used, but the two evaluated patients are also the two
  the profiler was debugged on, which may flatter the multi-agent systems.
- **Scoring design.** The 50/50 relevance–eligibility blend (case 3) and "one not_met → ineligible" (case 1)
  are fixed design choices that were not tuned (tuning on these labels would be overfitting).

## Reproduce

```
python -m scripts.run_eval --patients 2 --systems bm25 single_call multiagent no_critic multiagent_rag
python -m scripts.significance results/rankings_batch_B_all5_p2.csv
```
All LLM answers are cached under `.cache/`, so the commands above replay instantly. `results/run_batches.sh`
runs the time-boxed batches.
