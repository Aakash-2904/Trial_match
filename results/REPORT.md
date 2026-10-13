# TrialMatch evaluation report

Local multi-agent clinical trial matching (LangGraph + Ollama `qwen2.5:3b`, CPU only, temperature 0) on TREC
Clinical Trials 2021. Patient notes went only to local models. Every code change and decision is in
`BUILD_LOG.md`. The earlier 2-patient pilot report (before the design fixes) is kept as
`REPORT_pilot_2patients.md`.

> **Summary.** On 8 held-out patients (96 patient–trial pairs), one well-prompted call to the 3B model
> (`single_call`) ranked trials best (nDCG@10 0.882), ahead of multi-agent with RAG (0.831), multi-agent
> (0.777), multi-agent without the critic (0.774) and BM25 (0.712). **No paired comparison is statistically
> clear** (all Wilcoxon p ≥ 0.109, n = 8). The main reason the agent pipeline doesn't win: under the
> grounded-evidence rule, the 3B model rarely quotes the note, so 92% of criteria end up "unknown". Most trials
> then sit in one tier, ranked by relevance alone.

## Protocol

- **Data.** TREC 2021 topics and qrels (2 = eligible, 1 = excluded, 0 = not relevant). 10 eval patients × 12
  candidate trials, balanced to 4/4/4 by qrel. Trial text is the current ClinicalTrials.gov version; criteria
  were parsed beforehand from public text (`data/parsed_api.json`).
- **Dev/test split.** Patients 4 and 5 were development (all design fixes were made by looking at these two
  and by code review). The other 8 (topics 12, 14, 15, 18, 29, 32, 36, 56) are the held-out test set.
- **Freeze.** Before the held-out run, SHA-256 hashes of every project `.py` file were saved in `FREEZE.txt`.
  They still matched after the run (`sha256sum -c`). No code, prompt or threshold changed during the run.
- **Run.** `python -m scripts.run_eval --exclude 4 5 --systems bm25 single_call multiagent no_critic
  multiagent_rag`, 2026-10-10 to 2026-10-12. It was stopped several times for low memory (8 GB laptop),
  and it was resumed with the same command each time. Model answers are cached and decoding is greedy, so a
  resumed run gives the same result as an uninterrupted one. Final run: exit 0, no tracebacks.
- **Metrics.** nDCG@10 and P@5 over the 12 candidates. Eligible-vs-excluded accuracy is computed only over
  trials judged relevant (qrel 1 or 2: 8 per patient, 64 in all), with "likely" or "possibly" eligible
  counted as a prediction of eligible. Always predicting "eligible" scores 0.500.
- **Significance.** Paired over patients on nDCG@10: bootstrap 95% CI (10,000 resamples, seed 42) and exact
  two-sided Wilcoxon signed-rank. The rule, fixed before the run: a difference counts only if p < 0.05 and the
  CI excludes 0.
- **Blinding.** Held-out metrics weren't analysed before the run finished. A few per-patient lines printed
  in the run log were seen while checking progress. The code was already frozen, so they couldn't change
  anything.

## Results

### Held-out test set (8 patients)

| System | nDCG@10 | P@5 | Eligible-vs-excluded acc. | CPU time / patient |
|---|---|---|---|---|
| bm25 | 0.712 | 0.300 | n/a | < 1 s |
| multiagent | 0.777 | 0.375 | 0.547 | ~14–17 min |
| no_critic | 0.774 | 0.400 | 0.562 | ~0 extra (reuses multiagent's checker calls) |
| multiagent_rag | 0.831 | 0.475 | 0.578 | ~5–15 min |
| **single_call** | **0.882** | **0.550** | **0.609** | ~4 min |

Per-patient nDCG@10:

| System | 12 | 14 | 15 | 18 | 29 | 32 | 36 | 56 |
|---|---|---|---|---|---|---|---|---|
| bm25 | 0.833 | 0.856 | 0.607 | 0.684 | 0.692 | 0.485 | 0.704 | 0.839 |
| single_call | 0.877 | 0.954 | 0.672 | **1.000** | **0.993** | **0.971** | 0.629 | **0.961** |
| multiagent | **0.940** | 0.786 | **0.701** | 0.813 | 0.658 | 0.822 | 0.752 | 0.746 |
| no_critic | **0.940** | 0.786 | 0.647 | 0.813 | 0.664 | 0.822 | 0.752 | 0.770 |
| multiagent_rag | 0.861 | 0.920 | 0.661 | 0.889 | 0.754 | 0.928 | **0.862** | 0.769 |

### Significance (`significance_heldout.md`)

| Comparison | Mean diff | 95% CI | Wilcoxon p | Win/tie/loss | Verdict |
|---|---|---|---|---|---|
| multiagent vs single_call | −0.105 | [−0.204, −0.001] | 0.109 | 3/0/5 | no clear difference |
| multiagent vs no_critic | +0.003 | [−0.009, +0.019] | 1.000 | 1/5/2 | no clear difference |
| multiagent_rag vs multiagent | +0.053 | [+0.001, +0.100] | 0.109 | 6/0/2 | no clear difference |
| multiagent vs bm25 | +0.065 | [−0.018, +0.160] | 0.195 | 5/0/3 | no clear difference |

With 8 patients the smallest possible two-sided Wilcoxon p is 0.008, so only a nearly unanimous difference can
pass. Two CIs only just exclude 0 (single_call over multiagent, RAG over multiagent), but their p-values don't
pass, so under the pre-set rule they are not claimed.

### Dev vs test

The order of systems is the same on dev (patients 4–5, after the fixes) and test:
single_call > multiagent_rag > multiagent ≈ no_critic > bm25. On dev the nDCG@10 values were 0.967, 0.954,
0.816, 0.816 and 0.685. Absolute scores are lower on test for every LLM system, as expected after
development on the dev patients.

## What the agents actually decided

Decision sources for all 1,252 criteria across the 96 held-out pairs (from `multiagent_*.json`):

| Source | met | not met | unknown |
|---|---|---|---|
| Rule checker (numeric labs, age, sex) | 32 | 13 | 15 |
| LLM checker, grounded answer | 41 | 6 | 530 |
| LLM answer rejected by the grounding guard | – | – | 506 |
| Over the 30-criteria cap per trial | – | – | 106 |
| Critic overturned | – | – | 3 |

- **92% of criteria are "unknown".** The LLM gave a yes/no for 553 criteria, but 506 of those (91.5%) had no
  evidence quoted from the note, or evidence that wasn't in it. The grounding guard (design fixes 1–2)
  correctly turned these into "unknown". Only 47 grounded LLM decisions remain, alongside 45 rule decisions.
- **No trial was ever "likely eligible".** Labels were 77 possibly eligible and 19 likely ineligible out of 96.
  Ranking is by tier, then by relevance (fix 4), so within the big possibly-eligible tier, multiagent orders
  trials by BM25 prefilter relevance. Its gain over BM25 comes almost entirely from moving 19 trials to the
  bottom. 10 of those 19 were not relevant (correct), 6 were excluded (correct) and 3 were eligible (wrong).
- **Accuracy is barely above "always eligible".** Possibly-eligible counts as eligible, so the multi-agent
  systems score close to the 0.500 majority rate (0.547–0.578). single_call is the only system that labels
  many trials "excluded" (51 of 96), and it gets 39 of 64 relevance-judged pairs right (0.609).

## Interpretation

**Multi-agent vs single call: no clear difference, and the single call is ahead on every metric.** It wins
5 of 8 patients. Several wins are large (patient 29: +0.336, patient 56: +0.214). Reading the whole eligibility
text at once, the 3B model makes a usable overall judgment. Asked one criterion at a time with a quote from the
note required, it rarely meets the evidence bar. The multi-agent pipeline's real advantages are transparency
and safety: per-criterion outcomes, grounded evidence, and no unsupported exclusions. It did not improve
ranking. It costs about 3–4× the CPU time of the single call.

**Critic: no measurable effect.** It overturned only 3 of 1,252 decisions, all "not met" → "unknown", and
5 of 8 patients are identical with and without it. The grounding guard already removes most unsupported
"not met" answers, so there is little left for the critic to review. In the pilot (before the guard) it
overturned 13 of 32.

**RAG: positive trend, not significant (+0.053, 6 of 8 patients).** This reverses the pilot, where RAG was
slightly worse. Two changes explain it. (1) Hybrid BM25 + dense relevance orders the large possibly-eligible
tier better than BM25 alone. That matters a lot now, because relevance decides the order inside a tier.
(2) The knowledge section is labelled "background only, never quote it". RAG still had more grounding
rejections than non-RAG (550 vs 506), but the gap is much smaller than in the pilot (47 vs 19 on 2 patients).
RAG also made fewer grounded "met" decisions (26 vs 41), so the gain comes from retrieval, not from better
eligibility reasoning.

**Multi-agent vs BM25: +0.065, 5 of 8 patients, not significant.** Pushing clearly ineligible trials down helps
on average but costs an eligible trial on some patients (case 1 below).

## Error analysis

Eligible trials (qrel 2) labelled likely ineligible by `multiagent`: 3 of 32. All three are listed. They were
found only for this write-up; nothing was changed because of them.

**1. A cohort option read as a requirement (patient 29, NCT01552811, PET study of islet mass).** The
inclusion criteria list two cohorts: patients with diabetes, or "Adult, healthy controls". The parser split
these into separate inclusion criteria with no "one of the following" heading, so the either/or group rule
(fix 5) didn't apply. The checker answered "no" to "healthy controls" with grounded evidence (the patient is a
diabetic who lost his insulin supply), the critic confirmed it, and the trial was excluded. Each "no" was
correct on its own; the logic combining them was wrong. Cohort alternatives without a heading are a
parsing gap.

**2. A literal match at the wrong time (patient 14, NCT02368639, positive-airway-pressure device).** The
exclusion "Coma, decreased level of consciousness or agitation" was judged met using the note's "agitated then
somnolent at initial presentation". The evidence is real and grounded, but it describes the admission, not
the patient's state at enrollment, and TREC judged the patient eligible. Grounding checks that a quote
exists, not that it applies at the right point in time.

**3. An administrative criterion answered from clinical text (patient 56, NCT01809808, acromegaly outcomes
study).** "Presenting to the PI or one of the sub-investigators for evaluation of acromegaly" was answered "no"
because the patient came to the clinic for follow-up labs after surgery, and the critic confirmed it.
Whether a patient presents to a specific investigator can't be decided from a clinical note. This should have
been "unknown" or skipped as non-clinical.

**Pattern.** All three are grounded "no" decisions, so the guard behaved as designed. The remaining errors
are about how criteria are combined (cohorts), when evidence applies (time) and which criteria are
answerable at all. Better quoting won't fix these.

## Limitations

- **Small sample.** 8 test patients, 96 pairs, 64 relevance-judged pairs for accuracy. One trial moves a
  patient's accuracy by 12.5 points. Effects of about 0.05–0.10 nDCG@10 can't be told apart from noise at this
  size.
- **3B model on CPU.** `qwen2.5:3b` (Q4, greedy) rarely quotes evidence, and that drives the 92% unknown rate. A
  larger model would likely ground more answers, which could change the multi-agent vs single-call
  comparison in either direction. One sample per call; no variance across seeds.
- **Criteria cap.** Trials with more than 30 fuzzy criteria have the rest marked "unknown" (106 criteria, 8.5%).
- **2021 judgments vs current trial text.** The qrels were made against the 2021 registry, but the trial text
  is from today, so some disagreements may be label drift rather than errors.
- **Development on 2 patients.** The fixes were developed on patients 4–5 only. They generalise in direction
  (same system order on test), but the grounding rule wasn't tuned for recall, by design.
- **`local_parse` not evaluated.** It was skipped for CPU time.
- **Operational.** The run needed several resumes on an 8 GB laptop. Caching makes these exact, but timings
  above are rough because of memory paging.

## Reproduce

```
python -m scripts.run_eval --exclude 4 5 --systems bm25 single_call multiagent no_critic multiagent_rag
python -m scripts.significance results/rankings_heldout.csv
```

All model answers are cached under `.cache/`, so both commands replay in seconds. Outputs:
`summary_heldout.{md,json}`, `rankings_heldout.csv`, `significance_heldout.md`, and per-trial decisions in
`<system>_<topic>.json`.
