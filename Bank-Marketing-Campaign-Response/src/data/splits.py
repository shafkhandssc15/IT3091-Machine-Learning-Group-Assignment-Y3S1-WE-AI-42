"""De-duplication and the chronological train / pilot / test split.

The file is ordered by contact date (May 2008 -> Nov 2010) but has no date column, so file order IS time order.
Never shuffle before splitting.
"""
import numpy as np

from src.config import PILOT_ROWS, TARGET, TEST_FRACTION


def deduplicate(raw):
    """Drop exact duplicate rows, keeping the first occurrence and the original order (order = time)."""
    return raw.drop_duplicates().reset_index(drop=True)


def split_xy(df):
    """Separate predictors from the binary target (1 = subscribed)."""
    y = (df[TARGET] == "yes").astype(int).values
    X = df.drop(columns=TARGET)
    return X, y


def chronological_split(df, test_fraction=TEST_FRACTION, pilot_rows=PILOT_ROWS):
    """Train = earliest rows, pilot = the next ``pilot_rows`` labelled rows, test = everything after.

    The pilot block imitates the first calls of a new period. It is never scored; notebook 06 uses it as the first
    labelled block for rolling recalibration.
    Returns a dict with X_*/y_* arrays plus the integer cut positions.
    """
    X, y = split_xy(df)
    cut = int(len(df) * (1 - test_fraction))
    out = dict(
        X_train=X.iloc[:cut].reset_index(drop=True), y_train=y[:cut],
        X_pilot=X.iloc[cut:cut + pilot_rows].reset_index(drop=True), y_pilot=y[cut:cut + pilot_rows],
        X_test=X.iloc[cut + pilot_rows:].reset_index(drop=True), y_test=y[cut + pilot_rows:],
        X_all=X, y_all=y, cut=cut, pilot_end=cut + pilot_rows,
    )
    return out


def describe_split(s):
    """One-row-per-part summary of a split dictionary (rows, response rate)."""
    import pandas as pd
    rows = []
    for name in ("train", "pilot", "test"):
        y = s[f"y_{name}"]
        rows.append(dict(part=name, rows=len(y), subscribers=int(y.sum()), response_rate=float(np.mean(y))))
    return pd.DataFrame(rows).set_index("part")
