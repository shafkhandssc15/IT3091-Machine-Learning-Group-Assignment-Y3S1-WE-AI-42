# Preprocessing log

Each decision: issue -> options -> evidence -> choice. Executable evidence is in `notebooks/02_preprocessing.ipynb`; the section is given in brackets.

| # | Issue | Options considered | Choice | Evidence |
|---|---|---|---|---|
| P1 | 12 exact duplicate rows | keep / drop | Drop later copies, keep first, keep file order (order = time) | 02 §3.1 |
| P2 | `unknown` in job, marital, default, housing, loan | own category / mode imputation / drop rows | Own category unless mode imputation wins >= 3 of 4 forward-chaining folds (rule fixed in advance; verdict printed) | 02 §3.2; EDA 3.4 |
| P3 | `unknown` education | one-hot / ordinal | Ordinal 0-6; `unknown` -> NaN -> training median + missing indicator | 02 §3.2 |
| P4 | `pdays = 999` (96.32% of rows) | leave / impute / flag | Flag `was_contacted_before`; placeholder -> 0 | 02 §3.3; EDA 3.6 |
| P5 | Right-skewed `campaign` | leave / clip / log | Keep + add `campaign_log` | 02 §4.1 |
| P6 | Different numeric scales | none / standardise | Standardise for logistic regression, MLP and SVM only | 02 §4.2 |
| P7 | Drift over time | stratified random / chronological | Chronological train (80%) / pilot (500) / test; forward-chaining CV for selection | 02 §5; EDA §9 |
| P8 | Proposal feature families | baseline / each / all | Adopt a family only if forward-chaining PR-AUC improves in >= 3 of 4 folds and stratified CV does not fall; switch in `ADOPTED_FAMILIES` (`src/config.py`) | 02 §6 |
| P9 | `duration` | keep / drop | Drop: known only after the call | 02 §7.1 |
| P10 | `campaign` reverse causality | keep / drop | Keep with caveat; removal tested | 02 §7.2 |
| P11 | Macro indicators | include / exclude | Exclude by default (`EXCLUDE_MACRO`); tested in 03 §5 | EDA §10; 03 §5 |
| P12 | Leakage control | manual / structural | Every learned step inside one `Pipeline`; asserted in 02 §8 and `tests/test_preprocessing.py` | 02 §8 |

## What is learned from data, and from which rows
| Step | Learned quantity | Fitted on |
|---|---|---|
| Median imputation (numeric + education) | per-column medians | training rows of the current fold |
| Standard scaling | per-column mean and standard deviation | training rows of the current fold |
| One-hot encoding | category lists, rare-level grouping (< 20 rows) | training rows of the current fold |
| SMOTE / SMOTE-Tomek | synthetic points | training rows of the current fold only |
| Everything in `add_features` | nothing (constants only) | n/a |
