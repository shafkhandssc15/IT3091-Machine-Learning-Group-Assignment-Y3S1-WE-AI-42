# Decision log

Project-level decisions. The detailed preprocessing decisions are in `preprocessing_log.md`. Rules marked "fixed in advance" were written before the result was seen; fill the **Outcome** column after running the notebooks.

| # | Decision | Options | Choice / rule | Evidence | Outcome (fill in after running) |
|---|---|---|---|---|---|
| D1 | Primary lens | response prediction / segmentation / profile analysis | Campaign response prediction | initial submission §5 | as chosen |
| D2 | Secondary lens | none / segmentation | Segment prioritisation, only to operationalise scores | 07 §1 | |
| D3 | `duration` | keep / drop | Drop (post-call) | 02 §7.1 | |
| D4 | `unknown` | drop / impute / own category | Own category | 02 §3.2 | |
| D5 | `pdays = 999` | numeric / NaN / flag | Flag + 0 | 02 §3.3 | |
| D6 | Metrics | accuracy / ROC-AUC / PR-AUC / lift / profit | PR-AUC, ROC-AUC, budget lift / recall / precision, profit | 01 §4 | |
| D7 | Validation | stratified random / chronological | Stratified CV + forward-chaining for selection; chronological hold-out for the final result | 01 §9, 05 §4 | optimism of a random split: |
| D8 | Imbalance | none / weights / SMOTE / SMOTE-Tomek | Simplest unless SMOTE-Tomek improves ranking | 04 | |
| D9 | Final model | LR / RF / HGB / MLP / challengers | Family with the best forward-chaining PR-AUC; challenger only under D13 | 03-05 | |
| D10 | Calibration | none / isotonic / Platt / rolling intercept | Rolling intercept + rank-based policy | 06 §2 | |
| D11 | Data version and macro | `bank-full` / additional with or without macro | Keep B (additional, no macro) unless A beats B and C on forward-chaining AND hold-out lift | 03 §5 | A vs B vs C: |
| D12 | Feature families | baseline / each / all | Adopt on >= 3 of 4 forward-chaining folds and no CV loss | 02 §6 | adopted: |
| D13 | Extra models | keep incumbent / challenger | Replace only with higher mean forward-chaining PR-AUC and >= 3 of 4 folds | 03 §3 | |
| D14 | Operating point | static threshold / break-even / rank-based volume | Rank-based call volume | 06 §5 | called % val vs test: |
| D15 | Explanation | permutation only / add TreeSHAP | Both; trust drivers in both | 07 §2 | rank agreement: |
| D16 | Sensitive features | keep / drop | Decide from the cost table | 07 §3 | |
