"""Reusable checks for the exploratory analysis in notebook 01 (and for the tests).

Every function takes a plain DataFrame and returns a small table, so the notebook can show the evidence and the
tests can assert on it.
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif

from src.config import MONTH_NUM, SEED


# --------------------------------------------------------------------------- basic statistics
def wilson_ci(successes, n, z=1.96):
    """95% Wilson interval for a proportion (well behaved for small groups and rates near 0 or 1)."""
    successes, n = np.asarray(successes, float), np.asarray(n, float)
    p = np.divide(successes, n, out=np.zeros_like(successes), where=n > 0)
    denom = 1 + z ** 2 / np.maximum(n, 1)
    centre = (p + z ** 2 / (2 * np.maximum(n, 1))) / denom
    half = z * np.sqrt(p * (1 - p) / np.maximum(n, 1) + z ** 2 / (4 * np.maximum(n, 1) ** 2)) / denom
    return centre - half, centre + half


def rate_table(df, col, target="y_bin", order=None, min_n=0):
    """Subscription rate per level of ``col`` with volume, 95% interval and lift versus the overall rate."""
    base = df[target].mean()
    g = df.groupby(col, observed=True)[target].agg(contacts="size", subscribers="sum")
    g["share_of_contacts_%"] = 100 * g["contacts"] / len(df)
    g["rate_%"] = 100 * g["subscribers"] / g["contacts"]
    lo, hi = wilson_ci(g["subscribers"], g["contacts"])
    g["ci_low_%"], g["ci_high_%"] = 100 * lo, 100 * hi
    g["lift_vs_overall"] = (g["rate_%"] / 100) / base
    g = g[g["contacts"] >= min_n]
    if order is not None:
        g = g.reindex([o for o in order if o in g.index])
    else:
        g = g.sort_values("rate_%", ascending=False)
    return g.round(2)


def cramers_v(x, y):
    """Bias-corrected Cramer's V between two categorical series, plus the chi-square p-value."""
    ct = pd.crosstab(x, y)
    chi2, p, _, _ = stats.chi2_contingency(ct, correction=False)
    n = ct.values.sum()
    r, k = ct.shape
    phi2 = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    return float(np.sqrt(phi2 / max(min(kc - 1, rc - 1), 1e-12))), float(p)


def signal_table(df, categorical, numeric, target="y_bin", flag=("duration",)):
    """Rank every raw column by how strongly it is associated with the target.

    Categorical -> Cramer's V and chi-square p-value; numeric -> Mann-Whitney effect size (AUC-based, 0.5 = no signal)
    and p-value. Mutual information (nats) is added for both so the two kinds can be compared on one scale.
    """
    rows = []
    y = df[target].values
    for c in categorical:
        v, p = cramers_v(df[c], df[target])
        rows.append(dict(column=c, kind="categorical", effect_size=v, effect_name="Cramer's V", p_value=p))
    for c in numeric:
        a, b = df.loc[y == 1, c].values, df.loc[y == 0, c].values
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        auc = u / (len(a) * len(b))
        rows.append(dict(column=c, kind="numeric", effect_size=abs(auc - 0.5) * 2, effect_name="|2*AUC-1|", p_value=p))
    t = pd.DataFrame(rows).set_index("column")
    X = pd.DataFrame({c: pd.factorize(df[c])[0] for c in categorical})
    for c in numeric:
        X[c] = pd.qcut(df[c].rank(method="first"), 10, labels=False)         # 10 equal-frequency bins -> discrete MI
    discrete = np.array([True] * X.shape[1])
    t["mutual_info"] = pd.Series(mutual_info_classif(X, y, discrete_features=discrete, random_state=SEED), index=X.columns)
    t["flag"] = ["post-call: leakage" if c in flag else "" for c in t.index]
    return t.sort_values("mutual_info", ascending=False)


# --------------------------------------------------------------------------- data-quality checks
def unknown_table(df, categorical):
    """Share of the hidden-missing label 'unknown' per categorical column."""
    rows = []
    for c in categorical:
        n = int((df[c] == "unknown").sum())
        rows.append(dict(column=c, unknown_rows=n, unknown_pct=100 * n / len(df)))
    return pd.DataFrame(rows).set_index("column").sort_values("unknown_pct", ascending=False).round(2)


def iqr_outlier_table(df, columns):
    """Skewness, tail and IQR-fence outlier counts for numeric columns (outliers are CHECKED, not automatically removed)."""
    rows = []
    for c in columns:
        s = df[c]
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((s < lo) | (s > hi)).sum())
        rows.append(dict(column=c, min=s.min(), median=s.median(), p99=s.quantile(0.99), max=s.max(),
                         skew=s.skew(), outliers_iqr=n_out, outliers_pct=100 * n_out / len(s)))
    return pd.DataFrame(rows).set_index("column").round(2)


def vif_table(df, columns):
    """Variance inflation factor per column (VIF > 10 is the usual 'severely collinear' rule of thumb)."""
    X = df[columns].astype(float).values
    X = (X - X.mean(0)) / X.std(0)
    rows = []
    for i, c in enumerate(columns):
        others = np.delete(X, i, axis=1)
        beta, *_ = np.linalg.lstsq(others, X[:, i], rcond=None)
        resid = X[:, i] - others @ beta
        r2 = 1 - resid.var() / X[:, i].var()
        rows.append(dict(column=c, vif=1 / max(1 - r2, 1e-9)))
    return pd.DataFrame(rows).set_index("column").sort_values("vif", ascending=False).round(1)


def history_consistency(df):
    """The three previous-campaign columns describe the same fact. Do they agree row by row?"""
    never_pdays = df["pdays"] == 999
    never_prev = df["previous"] == 0
    never_pout = df["poutcome"] == "nonexistent"
    return pd.DataFrame({
        "pdays == 999": [never_pdays.sum()], "previous == 0": [never_prev.sum()], "poutcome == nonexistent": [never_pout.sum()],
        "all three agree": [int(((never_pdays == never_prev) & (never_prev == never_pout)).sum())],
        "pdays != 999 but previous == 0": [int((~never_pdays & never_prev).sum())],
        "pdays == 999 but previous > 0": [int((never_pdays & ~never_prev).sum())],
    }).T.rename(columns={0: "rows"})


# --------------------------------------------------------------------------- time reconstruction
def reconstruct_periods(df, min_run=30, min_drop=3):
    """Recover calendar year and month for every row.

    The file has a ``month`` column but no year, and it is sorted by contact date (May 2008 -> Nov 2010). A new year
    starts whenever the month number falls back by at least ``min_drop`` months (e.g. dec -> mar). Short runs
    (< ``min_run`` rows) are treated as ordering noise and inherit the period around them.

    Returns a DataFrame with columns year, month_num, period ('2009-03'), period_idx (0,1,2,... in time order).
    """
    m = df["month"].map(MONTH_NUM).to_numpy()
    # run-length encode consecutive identical months
    starts = np.r_[0, np.flatnonzero(np.diff(m) != 0) + 1]
    ends = np.r_[starts[1:], len(m)]
    year, run_max = 2008, 0
    years = np.empty(len(m), dtype=int)
    for s, e in zip(starts, ends):
        month = m[s]
        if e - s >= min_run:                                     # ignore tiny runs when detecting a wrap
            if month <= run_max - min_drop:
                year += 1
                run_max = 0
            run_max = max(run_max, month)
        years[s:e] = year
    out = pd.DataFrame({"year": years, "month_num": m})
    out["period"] = out["year"].astype(str) + "-" + out["month_num"].astype(str).str.zfill(2)
    out["period_idx"] = pd.factorize(out["period"], sort=False)[0]
    return out


def quality_scorecard(raw, categorical):
    """One-glance table of the data-quality checks, each with a verdict and the action taken in notebook 02."""
    n = len(raw)
    dup = int(raw.duplicated().sum())
    nan_cells = int(raw.isna().sum().sum())
    unk = unknown_table(raw, categorical)
    unk_cols = unk[unk["unknown_rows"] > 0]
    sentinel = float((raw["pdays"] == 999).mean() * 100)
    const = [c for c in raw.columns if raw[c].nunique() <= 1]
    rows = [
        ("Exact duplicate rows", f"{dup} of {n:,}", "minor" if dup < 0.001 * n else "check", "drop, keep first, keep order"),
        ("True missing cells (NaN)", f"{nan_cells}", "ok" if nan_cells == 0 else "check", "none needed"),
        ("Hidden missing values ('unknown')",
         ", ".join(f"{c} {r.unknown_pct:.1f}%" for c, r in unk_cols.iterrows()) or "none",
         "check" if len(unk_cols) else "ok", "keep as own category; education -> NaN + indicator"),
        ("Sentinel value pdays = 999", f"{sentinel:.1f}% of rows", "check", "flag + replace by 0"),
        ("Constant columns", ", ".join(const) or "none", "ok" if not const else "check", "none needed"),
        ("Post-call column present", "duration (call length in seconds)", "critical", "exclude from every deployable model"),
        ("Zero-length calls", f"{int((raw['duration'] == 0).sum())} rows with duration = 0", "minor", "irrelevant once duration is excluded"),
        ("Age range", f"{raw['age'].min()} to {raw['age'].max()} years; {int((raw['age'] < 18).sum())} clients under 18",
         "ok", "keep; age stays numeric (sensitive attribute, audited in notebook 07)"),
    ]
    return pd.DataFrame(rows, columns=["check", "result", "severity", "action"]).set_index("check")
