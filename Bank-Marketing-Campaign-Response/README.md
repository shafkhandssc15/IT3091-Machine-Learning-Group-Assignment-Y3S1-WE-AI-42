# Bank Marketing Campaign Response

**IT3091 Machine Learning, SLIIT, Group 2026-AI-42**

A decision-support project on the UCI *Bank Marketing* data (`bank-additional-full.csv`, 41,188 phone contacts, 21 columns). The model ranks clients by how likely they are to subscribe to a term deposit, so the call centre can phone the most promising clients first.

## Problem framing

| Question | Answer |
|---|---|
| Decision | Whom should the call centre phone first when it can only call part of the list? |
| Primary lens | Campaign response prediction: a ranked call list. |
| Secondary lens | Segment prioritisation: which client groups deserve the effort. |
| Target | `y` (subscribed to a term deposit), 11.27% positive. |
| Not allowed as input | `duration`. The call length is only known after the call, so it would be leakage. |
| Metrics | PR-AUC, ROC-AUC, and precision, recall and lift at the top 20% of the list. Accuracy is not used. |
| Split | Chronological: oldest 80% train, 500-row pilot, newest 20% held-out future. |
| Business figures | Call cost, subscription value and call budget are **illustrative**. The dataset has no financial data. |

## Team

| Member | ID | Role |
|---|---|---|
| Dharmarathne M.B.H.N | IT24103127 | Problem framing and EDA lead |
| Ekanayake E.M.B | IT24101737 | Preprocessing and feature engineering lead |
| Warakapola A.A.M | IT24103303 | Modelling and comparison lead |
| Mohammed M H S | IT24102308 | Evaluation, recommendation and documentation lead |

> TODO (group): confirm which member holds which role. The roles are listed, but the name-to-role mapping above is a placeholder.

## Results on the real data (seed 42, chronological hold-out)

The notebooks in this repo were executed on `bank-additional-full.csv` and ship with their outputs. The figures below come from those runs.

| Finding | Value |
|---|---|
| Rows after removing 12 exact duplicates | 41,176 |
| Response rate: training / pilot / test period | 6.4% / 6.4% / 32.4% |
| Leakage: ROC-AUC with vs without `duration` | 0.953 vs 0.671 |
| Selected model (forward-chaining PR-AUC) | Tuned logistic regression |
| Hold-out ROC-AUC, whole test period | 0.689 (includes the effect of the period) |
| Client-level skill, ROC-AUC inside time blocks | 0.609, lift about 1.35x at a 20% call budget |
| Stratified random split vs chronological hold-out | Flatters ROC-AUC by about 0.07 and lift by about 1.1x |
| Proposal feature families | None passed the pre-stated adoption rule |
| Macro indicators | Excluded; adding them lowered forward-chaining PR-AUC for both models (hold-out result mixed: logistic regression slightly up, boosting clearly down) |

Read these as an honest result, not a disappointing one: the response rate shifts sharply over time, so skill in ranking clients within a period is modest, and a random split would have hidden that.

## Notebooks (run in order)

| # | Notebook | What it does |
|---|---|---|
| 01 | `01_eda.ipynb` | Understands the data: availability audit, quality scorecard, target, rates by group, history, contact pressure, time drift, correlation, insight log. |
| 02 | `02_preprocessing.ipynb` | Turns each EDA finding into a tested preprocessing decision and builds the final leakage-safe pipeline. |
| 03 | `03_model_comparison.ipynb` | Compares candidate models with stratified and forward-chaining CV; selects the families. |
| 04 | `04_imbalance_and_tuning.ipynb` | Class weights, SMOTE, SMOTE-Tomek and hyper-parameter search on time-ordered folds. |
| 05 | `05_final_evaluation.ipynb` | Hold-out evaluation on the newest 20% of contacts. |
| 06 | `06_call_policy.ipynb` | Recalibration, gain and lift, and the profit-based call policy. |
| 07 | `07_segments_and_responsible_ai.ipynb` | Segments, feature importance, SHAP, fairness audit. |
| 08 | `08_business_interpretation.ipynb` | Recommendation, limitations, decision log, reproducibility, AI-use declaration. |

Notebooks 01 and 02 are the most detailed. Each EDA finding gets an ID (E1 to E12) and each preprocessing decision an ID (P1 to P12), so you can trace a decision back to its evidence.

## Repository layout

```
Bank-Marketing-Campaign-Response/
├── notebooks/        8 numbered notebooks (run 01 -> 08)
├── src/
│   ├── config.py     every setting in one place (seed, budget, split, feature policy)
│   ├── data/         loading, splits, shared data preparation, EDA checks
│   ├── features/     preprocessing pipeline (the only place that transforms features)
│   ├── models/       candidate models and tuning spaces
│   ├── evaluation/   metrics, ranking, workflow diagram
│   └── plotting.py   one figure style for every notebook
├── docs/             project map, workflow, data dictionary, EDA / preprocessing / decision logs,
│                     proposal-vs-implementation, reproducibility notes, report outline
├── tests/            unit tests for preprocessing and leakage safety
├── Dataset/          put bank-additional-full.csv here (see Dataset/README.md)
├── artifacts/        models and scores passed between notebooks (created when you run them)
└── reports/          tables and figures written by the notebooks (created when you run them)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Data: place `bank-additional-full.csv` in `Dataset/`, or set the `BANK_CSV` environment variable to its path. If the file is missing, the loader tries to download it from UCI. Optionally add `bank-full.csv` for the dataset-version check in notebook 03.

Run the notebooks from the `notebooks/` folder, in order. Notebooks 03 to 08 read what earlier ones saved in `artifacts/`, so do not skip ahead.

Tests (they use a synthetic stand-in, not real data):

```bash
python -m unittest discover -s tests -t .
```

## Honest notes on limits

- The data is Portuguese bank telemarketing from 2008 to 2010. Results show the method, not current behaviour.
- The file has no date column. The year is reconstructed from month wraps in the file order (notebook 01 §9). This is a documented assumption.
- Macro indicators (`euribor3m`, `emp.var.rate`, and so on) act as calendar proxies, so they are excluded by default (`EXCLUDE_MACRO` in `src/config.py`). Notebook 03 §5 tests this choice.
- `campaign` (number of calls in the current campaign) is a grey zone. Notebook 02 §7 runs a sensitivity test.
- The synthetic data in `tests/` carries no real information. Never quote a number from it.
- Written readings next to outputs should be checked against your own run before submission.

## Documentation

Start with `docs/project_map.md`. The logs in `docs/` record each decision and the evidence behind it. `docs/final_report_outline.md` maps report sections to notebooks.
