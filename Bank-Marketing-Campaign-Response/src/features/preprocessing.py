"""Leakage-safe preprocessing for the Bank Marketing task.

Everything that learns something from data (medians, scaling parameters, category lists) lives INSIDE a scikit-learn
``Pipeline`` so it is fitted on a training fold only. The only row-wise, parameter-free steps (re-coding the
``pdays = 999`` placeholder, mapping education to an ordinal, a log copy of ``campaign``) are done by
``add_features``; they use constants, not statistics, so they cannot leak.

Decisions behind every line are recorded in ``docs/preprocessing_log.md`` and demonstrated in notebook 02.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config import ADOPTED_FAMILIES, EDU_ORDER, EXCLUDE_MACRO, MACRO_COLS, SEED

try:                                    # SMOTE variants are optional: only notebook 04 needs them
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_IMB = True
except Exception:                       # pragma: no cover
    SMOTE, ImbPipeline, HAS_IMB = None, None, False

# --------------------------------------------------------------------------- column roles after add_features
NUM_COLS = ["age", "campaign", "campaign_log", "pdays", "previous", "was_contacted_before", "education_ord",
            "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
CAT_COLS = ["job", "marital", "default", "housing", "loan", "contact", "month", "day_of_week", "poutcome"]
MACRO = list(MACRO_COLS)


def add_features(X):
    """Row-wise, statistic-free feature construction (safe to apply before the split)."""
    X = X.copy()
    X["was_contacted_before"] = (X["pdays"] != 999).astype(int)         # pdays == 999 means "never contacted before"
    X["pdays"] = X["pdays"].where(X["pdays"] != 999, 0)                 # placeholder -> 0; the flag keeps the information
    X["campaign_log"] = np.log1p(X["campaign"])                         # tames the long right tail of call counts
    X["education_ord"] = X["education"].map(EDU_ORDER)                  # 'unknown' -> NaN -> imputed + missing indicator
    return X


def active_columns(exclude=(), include_duration=False, exclude_macro=None):
    """The numeric and categorical columns a pipeline will use, after removing ``exclude`` (and the macro block)."""
    exclude_macro = EXCLUDE_MACRO if exclude_macro is None else exclude_macro
    exclude = tuple(exclude) + (tuple(MACRO) if exclude_macro else ())
    num = [c for c in NUM_COLS if c not in exclude] + (["duration"] if include_duration else [])
    cat = [c for c in CAT_COLS if c not in exclude]
    return num, cat


def build_column_transformer(num, cat, scale=True):
    """Median imputation (+ missing indicator) and optional standardisation for numbers; one-hot for categories."""
    num_steps = [("imp", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num_steps.append(("sc", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), list(num)),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False), list(cat)),
    ])


def build_pipeline(model, scale=True, exclude=(), include_duration=False, smote=False, sampler=None, fe=add_features,
                   num=None, cat=None):
    """The one preprocessing + model pipeline used by every notebook.

    scale             standardise numbers (needed for logistic regression / MLP / SVM, harmless but unnecessary for trees)
    exclude           raw or engineered columns to leave out (e.g. the sensitive-attribute test in notebook 07)
    include_duration  ONLY for the leakage demonstration in notebook 02; never for a deployable model
    smote / sampler   resampling INSIDE the pipeline, so it only ever sees the training part of a fold
    fe, num, cat      override the feature function / column lists (used by the feature-family ablation)
    """
    d_num, d_cat = active_columns(exclude, include_duration)
    if ADOPTED_FAMILIES and fe is add_features and num is None and cat is None:     # families adopted in notebook 02
        fe = add_features_v2
        d_num = d_num + [c for g in ADOPTED_FAMILIES for c in FE_GROUPS[g]["num"]]
        d_cat = d_cat + [c for g in ADOPTED_FAMILIES for c in FE_GROUPS[g]["cat"]]
    pre = build_column_transformer(d_num if num is None else num, d_cat if cat is None else cat, scale)
    steps = [("fe", FunctionTransformer(fe)), ("prep", pre)]
    if smote and sampler is None:
        sampler = SMOTE(random_state=SEED)
    if sampler is not None:
        steps.append(("resample", sampler))
    steps.append(("model", model))
    return (ImbPipeline if sampler is not None else Pipeline)(steps)


# --------------------------------------------------------------------------- feature families named in the proposal
JOB_GROUP = {"admin.": "white_collar", "management": "white_collar", "technician": "white_collar",
             "entrepreneur": "white_collar", "self-employed": "white_collar", "blue-collar": "blue_collar_service",
             "services": "blue_collar_service", "housemaid": "blue_collar_service", "retired": "not_working",
             "student": "not_working", "unemployed": "not_working"}
EDU_GROUP = {"illiterate": "basic", "basic.4y": "basic", "basic.6y": "basic", "basic.9y": "basic",
             "high.school": "secondary", "professional.course": "tertiary", "university.degree": "tertiary"}
MONTH_QUARTER = {"jan": 1, "feb": 1, "mar": 1, "apr": 2, "may": 2, "jun": 2, "jul": 3, "aug": 3, "sep": 3,
                 "oct": 4, "nov": 4, "dec": 4}
_YN = {"yes": 1.0, "no": 0.0}


def add_features_v2(X):
    """``add_features`` plus the four feature families promised in the initial submission (section 15, step 4)."""
    X = add_features(X)
    X["job_group"] = X["job"].map(JOB_GROUP).fillna("unknown")
    X["edu_group"] = X["education"].map(EDU_GROUP).fillna("unknown")
    X["debt_load"] = X["housing"].map(_YN) + X["loan"].map(_YN)         # 0, 1 or 2 loans (NaN if either is unknown)
    X["housing_loan"] = X["housing"].astype(str) + "|" + X["loan"].astype(str)
    X["quarter"] = "Q" + X["month"].map(MONTH_QUARTER).astype(str)
    contacted = X["was_contacted_before"] == 1
    X["pdays_recent"] = (contacted & (X["pdays"] <= 6)).astype(int)     # re-contacted within a week
    X["pdays_late"] = (contacted & (X["pdays"] > 6)).astype(int)
    X["pdays_log"] = np.log1p(X["pdays"])
    X["prev_success"] = (X["poutcome"] == "success").astype(int)
    return X


FE_GROUPS = {
    "job/education groups": dict(num=[], cat=["job_group", "edu_group"]),
    "loan interaction": dict(num=["debt_load"], cat=["housing_loan"]),
    "quarter indicators": dict(num=[], cat=["quarter"]),
    "pdays recency + prev. success": dict(num=["pdays_recent", "pdays_late", "pdays_log", "prev_success"], cat=[]),
}


def family_pipeline(model, groups, scale):
    """Baseline columns plus the engineered columns of the chosen ``groups`` (names from ``FE_GROUPS``)."""
    base_num, base_cat = active_columns()
    num = base_num + [c for g in groups for c in FE_GROUPS[g]["num"]]
    cat = base_cat + [c for g in groups for c in FE_GROUPS[g]["cat"]]
    return build_pipeline(model, scale=scale, fe=add_features_v2, num=num, cat=cat)


# --------------------------------------------------------------------------- documentation helper
def feature_log(fitted_pipeline, X_reference):
    """One row per model input column: where it comes from, what was done to it, and what it looks like after
    preprocessing. ``fitted_pipeline`` must already be fitted (so the statistics shown are training-only)."""
    pre = fitted_pipeline.named_steps["prep"]
    names = list(pre.get_feature_names_out())
    Z = fitted_pipeline[:-1].transform(X_reference)
    Z = np.asarray(Z)
    rows = []
    for i, n in enumerate(names):
        kind, rest = n.split("__", 1)
        if kind == "cat":
            src_col = next((c for c in sorted(CAT_COLS + ["job_group", "edu_group", "housing_loan", "quarter"], key=len,
                                              reverse=True) if rest.startswith(c + "_")), rest)
            how = "one-hot indicator"
        elif rest.startswith("missingindicator_"):
            src_col, how = rest.replace("missingindicator_", ""), "missing-value indicator (1 = was 'unknown')"
        else:
            src_col, how = rest, "median-imputed" + (" + standardised" if "sc" in pre.named_transformers_["num"].named_steps else "")
        rows.append(dict(model_input=n, source_column=src_col, treatment=how,
                         mean_after=float(Z[:, i].mean()), std_after=float(Z[:, i].std())))
    return pd.DataFrame(rows)
