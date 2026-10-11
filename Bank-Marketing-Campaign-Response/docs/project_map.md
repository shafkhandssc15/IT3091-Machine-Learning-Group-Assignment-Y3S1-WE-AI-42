# Project map

| # | Notebook | Question it answers | Reusable code in `src/` | Stage document |
|---|---|---|---|---|
| 01 | `01_eda.ipynb` | What is in the data, can we trust it, who subscribes, does it drift? | `data/loading.py`, `data/eda_checks.py`, `plotting.py` | `eda_insight_log.md`, `data_dictionary.md` |
| 02 | `02_preprocessing.ipynb` | How is the data cleaned, encoded, split and guarded against leakage? | `features/preprocessing.py`, `data/splits.py` | `preprocessing_log.md` |
| 03 | `03_model_comparison.ipynb` | Which model family ranks clients best (and which data version)? | `models/candidates.py`, `evaluation/metrics.py` | `decision_log.md` (D9, D11, D13) |
| 04 | `04_imbalance_and_tuning.ipynb` | Does re-balancing help? Which settings are best? | `models/candidates.py` | `decision_log.md` (D8) |
| 05 | `05_final_evaluation.ipynb` | How does the model do on a later period, and how much would a random split flatter it? | `evaluation/metrics.py` | `decision_log.md` (D7) |
| 06 | `06_call_policy.ipynb` | Whom exactly do we call? Calibration, lift, profit, operating points | `evaluation/metrics.py` | `decision_log.md` (D10, D14) |
| 07 | `07_segments_and_responsible_ai.ipynb` | Segments, explanations, fairness audit | `features/preprocessing.py` | `decision_log.md` (D15, D16) |
| 08 | `08_business_interpretation.ipynb` | Recommendation, limitations, reproducibility, AI-use | - | `decision_log.md`, `final_report_outline.md` |

Notebooks hand results to each other through small files: `artifacts/selection.json` (03 -> 04), `artifacts/tuned_pipeline.joblib` (04 -> 05-07),
`artifacts/test_scores.npz` (05 -> 06), `artifacts/policy_scores.npz` (06 -> 07), and CSV tables in `reports/tables/` (read by 05 and 08).
Run them in order 01 to 08.

Assignment criteria (descriptor section 4) and where the evidence lives:

| Required evidence | Where |
|---|---|
| Problem framing canvas | `README.md` (canvas table) and the initial submission |
| Workflow diagram | `docs/workflow.md`, `docs/figures/workflow_diagram.png` |
| Decision log | `docs/decision_log.md`, `docs/preprocessing_log.md`, notebook 08 §3 |
| EDA insight log | notebook 01 §12, `docs/eda_insight_log.md` |
| Preprocessing / feature log | notebook 02 §9, `docs/preprocessing_log.md`, `reports/tables/final_feature_list.csv` |
| Model / method comparison | notebooks 03-05 |
| Recommendation + limitations | notebook 08 §2 |
| Reproducibility + AI-use | notebook 08 §4, `docs/reproducibility_notes.md` |
