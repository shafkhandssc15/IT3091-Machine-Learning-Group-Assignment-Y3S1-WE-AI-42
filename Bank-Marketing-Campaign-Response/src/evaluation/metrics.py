"""Evaluation helpers shared by notebooks 02-08.

The business question is "which clients should be called first?", so ranking metrics (PR-AUC, ROC-AUC) and budget
metrics (precision / recall / lift among the top ``CALL_BUDGET`` share of the list) are reported, never accuracy alone.
"""
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

from src.config import CALL_BUDGET, N_TS_SPLITS, SEED


def rank_order(score, seed=SEED):
    """Indices from best to worst score. Ties are broken RANDOMLY, never by row order (row order is chronological)."""
    return np.lexsort((np.random.RandomState(seed).rand(len(score)), -np.asarray(score)))


def topk_metrics(y_true, score, frac=CALL_BUDGET):
    """(precision, recall, lift) if only the best-scored ``frac`` of clients are contacted."""
    y_true = np.asarray(y_true)
    k = max(1, int(round(frac * len(y_true))))
    hits = y_true[rank_order(score)[:k]].sum()
    precision = hits / k
    return precision, hits / max(y_true.sum(), 1), precision / max(y_true.mean(), 1e-12)


def scores_of(model, X):
    """Probability if the model has one; otherwise the SVM decision value (only the ranking is compared)."""
    return model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X)


def metric_row(y_true, score, frac=CALL_BUDGET):
    p, r, l = topk_metrics(y_true, score, frac)
    return dict(roc_auc=roc_auc_score(y_true, score), pr_auc=average_precision_score(y_true, score),
                precision_at_budget=p, recall_at_budget=r, lift_at_budget=l)


def cv_eval(pipe, X, y, n_splits=5, seed=SEED):
    """Stratified k-fold on the TRAINING period. Same seed -> same folds for every model, so differences are paired."""
    rows = []
    for tr, va in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, y):
        t0 = time.time()
        m = clone(pipe).fit(X.iloc[tr], y[tr])
        fit_s = time.time() - t0
        s = scores_of(m, X.iloc[va])
        row = metric_row(y[va], s)
        row.update(accuracy_at_0_5=accuracy_score(y[va], (s >= 0.5).astype(int)) if hasattr(m, "predict_proba") else np.nan,
                   fit_seconds=fit_s, base_rate=y[va].mean())
        if hasattr(m, "predict_proba"):
            row.update(brier=brier_score_loss(y[va], s), mean_score=s.mean())
        else:
            row.update(brier=np.nan, mean_score=np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def fc_eval(pipe, X, y, n_splits=N_TS_SPLITS):
    """Forward-chaining: each fold trains on the PAST and validates on the next block (how the model is really used)."""
    rows = []
    for tr, va in TimeSeriesSplit(n_splits=n_splits).split(X):
        m = clone(pipe).fit(X.iloc[tr], y[tr])
        row = metric_row(y[va], scores_of(m, X.iloc[va]))
        row["val_base_rate"] = y[va].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def summarise(results):
    """(mean table, 'mean +/- sd' text table) for a dict of per-fold result frames."""
    mean = pd.DataFrame({k: v.mean() for k, v in results.items()}).T
    std = pd.DataFrame({k: v.std() for k, v in results.items()}).T
    return mean, mean.round(3).astype(str) + " +/- " + std.round(3).astype(str)


def decile_table(y_true, score, n_bins=10):
    """Gain / lift table: clients sorted best-first and cut into equal-size bins."""
    y_true = np.asarray(y_true)
    ys = y_true[rank_order(score)]
    dec = np.floor(np.arange(len(ys)) * n_bins / len(ys)).astype(int) + 1
    t = pd.DataFrame({"decile": dec, "y": ys}).groupby("decile").agg(clients=("y", "size"), subscribers=("y", "sum"))
    t["response_rate"] = t["subscribers"] / t["clients"]
    t["lift"] = t["response_rate"] / y_true.mean()
    t["cum_capture"] = t["subscribers"].cumsum() / t["subscribers"].sum()
    return t.round(3)
