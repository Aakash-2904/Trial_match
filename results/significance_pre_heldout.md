# Paired significance tests (per-patient nDCG@10)

Inputs: rankings_batch_B_all5_p2.csv. Difference = first system minus second. 95% CI: paired bootstrap over patients (10000 resamples, seed 42). p: exact two-sided Wilcoxon signed-rank (zero differences dropped). A difference counts only if p < 0.05 and the CI excludes 0; otherwise "no clear difference".

| Comparison | Patients | Mean diff | 95% CI | Wilcoxon p | Win/tie/loss | Verdict |
|---|---|---|---|---|---|---|
| multiagent vs single_call | 2 | -0.035 | [-0.086, +0.015] | 1.000 | 1/0/1 | no clear difference |
| multiagent vs no_critic | 2 | +0.061 | [+0.000, +0.121] | 1.000 | 1/1/0 | no clear difference |
| multiagent_rag vs multiagent | 2 | -0.015 | [-0.021, -0.009] | 0.500 | 0/0/2 | no clear difference |
| multiagent vs bm25 | 2 | +0.246 | [+0.147, +0.346] | 0.500 | 2/0/0 | no clear difference |

Per-patient differences:
- multiagent vs single_call: patient 4 -0.086, patient 5 +0.015
- multiagent vs no_critic: patient 4 +0.000, patient 5 +0.121
- multiagent_rag vs multiagent: patient 4 -0.009, patient 5 -0.021
- multiagent vs bm25: patient 4 +0.346, patient 5 +0.147

With 2 patients the smallest possible two-sided Wilcoxon p is 0.500, so no difference can reach p < 0.05. The bootstrap CI with this few patients only spans the observed differences, not real uncertainty.
