"""Small plotting helpers so every figure in the project looks the same and is saved to ``reports/figures``."""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.config import C_GREY, C_MAIN, C_OK, C_WARN, FIGURES_DIR


def set_style():
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"figure.dpi": 110, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 10})


def save_fig(name, fig=None):
    """Save the current (or given) figure as reports/figures/<name>.png and show it."""
    fig = fig or plt.gcf()
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.show()


def rate_bars(ax, table, title, base_rate, horizontal=True, min_label_n=0):
    """Bar chart of subscription rate (%) per category with 95% intervals and the overall rate as a reference line.

    ``table`` is the output of ``eda_checks.rate_table`` (in the order to be drawn). The number of contacts behind each
    bar is printed next to it, because a high rate on a tiny group is not a finding.
    """
    t = table.iloc[::-1] if horizontal else table
    err = np.vstack([t["rate_%"] - t["ci_low_%"], t["ci_high_%"] - t["rate_%"]])
    pos = np.arange(len(t))
    colors = [C_MAIN if r >= base_rate * 100 else C_GREY for r in t["rate_%"]]
    if horizontal:
        ax.barh(pos, t["rate_%"], xerr=err, color=colors, ecolor="#444", capsize=2)
        ax.set_yticks(pos, [str(i) for i in t.index])
        ax.axvline(base_rate * 100, color=C_WARN, ls="--", lw=1.2)
        for p, (r, n) in enumerate(zip(t["rate_%"], t["contacts"])):
            ax.text(r + err[1][p] + 0.4, p, f"n={n:,}", va="center", fontsize=7.5, color="#555")
        ax.set_xlabel("subscription rate (%)")
    else:
        ax.bar(pos, t["rate_%"], yerr=err, color=colors, ecolor="#444", capsize=2)
        ax.set_xticks(pos, [str(i) for i in t.index], rotation=0)
        ax.axhline(base_rate * 100, color=C_WARN, ls="--", lw=1.2)
        for p, (r, n) in enumerate(zip(t["rate_%"], t["contacts"])):
            ax.text(p, r + err[1][p] + 0.4, f"n={n:,}", ha="center", fontsize=7, color="#555", rotation=90)
        ax.set_ylabel("subscription rate (%)")
    ax.set_title(title)
