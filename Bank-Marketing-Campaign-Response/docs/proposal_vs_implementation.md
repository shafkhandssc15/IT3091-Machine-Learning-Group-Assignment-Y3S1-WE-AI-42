# Initial submission (16/09/2026) versus implementation

| Proposal item | Delivered in | Status |
|---|---|---|
| Binary target `y` used as given; `duration` excluded (§9-10) | 01 §2, 02 §7.1 | as proposed |
| Dataset `bank-full.csv`, 45,211 x 16 (§6) | Project models `bank-additional-full.csv`; the two are compared in 03 §5 | **deliberate change**, measured (the proposal said the choice would be revisited) |
| Feature engineering: grouped job / education, loan interaction, quarter indicators, `pdays` transformations (§15 step 4) | 02 §6 as a controlled ablation | as proposed, tested |
| Missing values in `contact` / `poutcome` (§19) | In `bank-additional-full.csv` these are not `unknown`; `unknown` handling for other columns in 02 §3.2 | adapted to the data version |
| Models: Dummy, LR, Decision Tree, RF, Gradient Boosting, optional SVM (§11); XGBoost, LightGBM (§15) | 03 §1-3 | as proposed |
| SMOTE-Tomek inside pipelines (§15 step 7) | 04 §2 | as proposed |
| Metrics: recall, precision, F1, ROC-AUC, PR-AUC, confusion matrix (§12) | 03-06; operating points in 06 §5 | as proposed |
| Stratified k-fold CV; test set held out (§13) | Stratified CV in 03-04; **chronological** hold-out in 05 | partly a **deliberate change**: data drift (EDA §9); optimism of a random split measured in 05 §4.2 |
| Threshold tuned on validation (§15 step 9) | 06 §5 | as proposed |
| Interpretability concern for XGBoost / LightGBM (§19) | 07 §2 (permutation importance, TreeSHAP) | addressed |
| Temporal drift, stability, operational feasibility (§16, §19) | 01 §9, 05 §4, 06 §1, fit times in 03 §3 | as proposed |
| Secondary lens: segment prioritisation (§5) | 07 §1 | as proposed |
| Workflow of 10 steps (§15) | `docs/workflow.md` | as proposed |
