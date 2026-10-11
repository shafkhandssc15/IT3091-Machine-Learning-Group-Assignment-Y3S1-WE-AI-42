"""Candidate models, defined once so every notebook compares the same configurations.

All candidates sit behind the SAME preprocessing pipeline (``build_pipeline``) and are scored on the SAME folds.
Hyper-parameters here are fixed, sensible defaults; notebook 04 tunes only the family selected in notebook 03.
"""
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from src.config import SEED
from src.features.preprocessing import build_pipeline

MAIN_FAMILIES = ["Logistic regression", "Random forest", "Hist. gradient boosting"]


def hgb(**kw):
    return HistGradientBoostingClassifier(random_state=SEED, **kw)


def build_main_models():
    """Baseline + four alternatives named in the proposal's strategy (Dummy, LR, RF, boosting) plus an MLP check."""
    return {
        "Dummy (prior)": build_pipeline(DummyClassifier(strategy="prior"), scale=False),
        "Logistic regression": build_pipeline(LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5), scale=True),
        "Random forest": build_pipeline(RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                                               class_weight="balanced_subsample", n_jobs=-1,
                                                               random_state=SEED), scale=False),
        "Hist. gradient boosting": build_pipeline(hgb(learning_rate=0.05, max_iter=300, class_weight="balanced"), scale=False),
        "MLP (64-32)": build_pipeline(MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True,
                                                    max_iter=300, random_state=SEED), scale=True),
    }


def build_challenger_specs(y_train):
    """name -> (estimator, needs_scaling). Decision tree, sklearn gradient boosting, XGBoost, LightGBM and an SVM."""
    import lightgbm as lgb
    import xgboost as xgb
    neg_pos = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    return {
        "Decision tree": (DecisionTreeClassifier(max_depth=6, min_samples_leaf=50, class_weight="balanced",
                                                 random_state=SEED), False),
        "Gradient boosting (sklearn)": (GradientBoostingClassifier(n_estimators=150, learning_rate=0.08, max_depth=3,
                                                                    subsample=0.8, random_state=SEED), False),
        "XGBoost": (xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8,
                                      colsample_bytree=0.8, min_child_weight=5, reg_lambda=5.0, scale_pos_weight=neg_pos,
                                      eval_metric="aucpr", tree_method="hist", n_jobs=-1, random_state=SEED), False),
        "LightGBM": (lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50,
                                        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, class_weight="balanced",
                                        n_jobs=-1, random_state=SEED, verbose=-1), False),
        # exact RBF-SVM is quadratic in the number of rows; Nystroem features + a linear SVM approximate it
        "SVM (RBF approx.)": (Pipeline([("nystroem", Nystroem(gamma=0.02, n_components=300, random_state=SEED)),
                                        ("svm", LinearSVC(C=0.5, class_weight="balanced", dual="auto", max_iter=5000))]),
                              True),
    }


# search spaces for notebook 04 (only the family chosen in notebook 03 is tuned)
def tuning_spaces():
    return {
        "Hist. gradient boosting": ({
            "model__learning_rate": [0.03, 0.05, 0.08, 0.1], "model__max_leaf_nodes": [15, 31, 63],
            "model__min_samples_leaf": [20, 50, 100], "model__l2_regularization": [0.0, 1.0, 10.0],
            "model__max_iter": [200, 400], "model__class_weight": [None, "balanced"]},
            lambda: build_pipeline(hgb(), scale=False)),
        "Random forest": ({
            "model__n_estimators": [200, 400], "model__min_samples_leaf": [3, 5, 10, 20],
            "model__max_features": ["sqrt", 0.3, 0.5], "model__max_depth": [None, 12, 20],
            "model__class_weight": [None, "balanced_subsample"]},
            lambda: build_pipeline(RandomForestClassifier(random_state=SEED, n_jobs=-1), scale=False)),
        "Logistic regression": ({
            "model__C": [0.01, 0.03, 0.1, 0.3, 1, 3], "model__class_weight": [None, "balanced"]},
            lambda: build_pipeline(LogisticRegression(max_iter=3000), scale=True)),
    }
