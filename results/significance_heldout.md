# Paired significance tests (per-patient nDCG@10)

Inputs: rankings_heldout.csv. Difference = first system minus second. 95% CI: paired bootstrap over patients (10000 resamples, seed 42). p: exact two-sided Wilcoxon signed-rank (zero differences dropped). A difference counts only if p < 0.05 and the CI excludes 0; otherwise "no clear difference".

| Comparison | Patients | Mean diff | 95% CI | Wilcoxon p | Win/tie/loss | Verdict |
|---|---|---|---|---|---|---|
| multiagent vs single_call | 8 | -0.105 | [-0.204, -0.001] | 0.109 | 3/0/5 | no clear difference |
| multiagent vs no_critic | 8 | +0.003 | [-0.009, +0.019] | 1.000 | 1/5/2 | no clear difference |
| multiagent_rag vs multiagent | 8 | +0.053 | [+0.001, +0.100] | 0.109 | 6/0/2 | no clear difference |
| multiagent vs bm25 | 8 | +0.065 | [-0.018, +0.160] | 0.195 | 5/0/3 | no clear difference |

Per-patient differences:
- multiagent vs single_call: patient 12 +0.063, patient 14 -0.168, patient 15 +0.029, patient 18 -0.187, patient 29 -0.336, patient 32 -0.150, patient 36 +0.123, patient 56 -0.214
- multiagent vs no_critic: patient 12 +0.000, patient 14 +0.000, patient 15 +0.054, patient 18 +0.000, patient 29 -0.007, patient 32 +0.000, patient 36 +0.000, patient 56 -0.024
- multiagent_rag vs multiagent: patient 12 -0.079, patient 14 +0.133, patient 15 -0.041, patient 18 +0.075, patient 29 +0.097, patient 32 +0.106, patient 36 +0.110, patient 56 +0.023
- multiagent vs bm25: patient 12 +0.107, patient 14 -0.069, patient 15 +0.095, patient 18 +0.129, patient 29 -0.034, patient 32 +0.337, patient 36 +0.048, patient 56 -0.092

With 8 patients the smallest possible two-sided Wilcoxon p is 0.008, so only a consistent difference can reach p < 0.05. The bootstrap CI with this few patients only spans the observed differences, not real uncertainty.
