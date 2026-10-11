# EDA insight log

Evidence is **computed by notebook 01 §12** and saved to `reports/tables/eda_insight_log.csv`, so the figures always match your run.
This page lists each finding with its consequence. Values marked (ref) were taken from the group's first run of the official file; check that your run reproduces them (notebook 01 §1 prints a checkpoint table).

| # | Theme | Finding | Consequence | Acted on in |
|---|---|---|---|---|
| E1 | Quality | 12 exact duplicates (ref), no empty cells | Drop duplicates, keep order | 02 §3.1 |
| E2 | Quality | `unknown` hides missing answers: default 20.87%, education 4.20%, housing 2.40%, loan 2.40%, job 0.80%, marital 0.19% (ref) | Keep as a category; education -> ordinal + indicator | 02 §3.2 |
| E3 | Quality | `pdays = 999` in 96.32% of rows (ref); agrees with `previous = 0` and `poutcome = nonexistent` | Flag + replace | 02 §3.3 |
| E4 | Leakage | `duration` has the highest correlation with the target (0.41 Pearson, ref) but exists only after the call | Excluded everywhere; leakage demonstrated | 02 §7.1 |
| E5 | Target | 11.27% subscribe (ref): about nine calls per sale | Accuracy meaningless; PR-AUC, lift, budget metrics | 03-06 |
| E6 | History | Previous-campaign successes are a small group with a much higher rate | Keep `poutcome`, `previous`, `pdays` | 02 §6 |
| E7 | Client | Students and retired clients subscribe far more than blue-collar / services; age is U-shaped | Keep job and age; fairness audit | 07 §3 |
| E8 | Campaign | Repeated calls convert worse; `campaign` partly records refusals | Keep + log copy; sensitivity test | 02 §7.2 |
| E9 | Campaign | Month looks strongest but high-rate months are low-volume and belong to different years | Validate chronologically | 02 §5 |
| E10 | Drift | Response rate rises sharply late in the file while interest rates fall | Chronological split, forward-chaining CV | 02 §5, 05 §4 |
| E11 | Macro | `emp.var.rate`, `euribor3m`, `nr.employed` correlate at about 0.9-0.97 (ref) and track the calendar | Excluded by default; tested | 03 §5 |
| E12 | Skew | `campaign` right-skewed; extremes are real behaviour | Keep rows; add `campaign_log` | 02 §4.1 |
