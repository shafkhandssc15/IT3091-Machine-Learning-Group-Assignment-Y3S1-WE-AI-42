"""One call that gives notebooks 03-08 the same cleaned data and the same chronological split."""
from src.data.loading import load_raw
from src.data.splits import chronological_split, deduplicate


def load_modelling_data(verbose=True):
    """Raw file -> duplicates dropped (order kept) -> train / pilot / test split dictionary (see ``chronological_split``).

    The returned dict also holds ``df`` (cleaned frame) and ``raw``. Nothing here learns from data, so it cannot leak.
    """
    raw = load_raw(verbose=verbose)
    df = deduplicate(raw)
    s = chronological_split(df)
    s["df"], s["raw"] = df, raw
    if verbose:
        print(f"Rows: raw {len(raw):,} -> cleaned {len(df):,} | train {len(s['y_train']):,} "
              f"(rate {s['y_train'].mean():.2%}) | pilot {len(s['y_pilot']):,} | test {len(s['y_test']):,} (rate {s['y_test'].mean():.2%})")
    return s
