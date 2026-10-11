"""Draw the project workflow (business problem -> recommendation) and save docs/figures/workflow_diagram.png.

Run from the project root:  python -m src.evaluation.workflow_diagram
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from src.config import C_MAIN, C_WARN, DOCS_DIR

STEPS = [("1  Business problem", "who to call first?", "framing"), ("2  Data audit", "dictionary, quality, leakage", "01"),
         ("3  EDA", "who subscribes? drift?", "01"), ("4  Preprocessing", "cleaning, features, split", "02"),
         ("5  Model comparison", "baseline + alternatives", "03"), ("6  Imbalance + tuning", "tested, not assumed", "04"),
         ("7  Final evaluation", "later period, once", "05"), ("8  Call policy", "calibrate, rank, threshold", "06"),
         ("9  Segments + fairness", "explain and audit", "07"), ("10 Recommendation", "value, limits, decisions", "08")]


def main():
    fig, ax = plt.subplots(figsize=(15, 5)); ax.set_xlim(0, 10); ax.set_ylim(0, 3.4); ax.axis("off")
    pos = {}
    for i in range(10):
        row, col = divmod(i, 5)
        pos[i] = ((col if row == 0 else 4 - col) * 2.0 + 1.0, 2.5 if row == 0 else 1.1)
    for i, (title, sub, nbk) in enumerate(STEPS):
        x, y = pos[i]
        ax.add_patch(FancyBboxPatch((x - 0.9, y - 0.45), 1.8, 0.9, boxstyle="round,pad=0.02,rounding_size=0.08", fc="#eef3ff", ec=C_MAIN, lw=1.4))
        ax.text(x, y + 0.17, title, ha="center", va="center", fontsize=9.5, weight="bold")
        ax.text(x, y - 0.05, sub, ha="center", va="center", fontsize=8, color="#445")
        ax.text(x, y - 0.29, f"notebook {nbk}" if nbk != "framing" else "README / initial submission", ha="center", va="center", fontsize=7.5, color=C_MAIN)
    for i in range(9):
        (x1, y1), (x2, y2) = pos[i], pos[i + 1]
        if y1 == y2:
            d = 1 if x2 > x1 else -1
            ax.annotate("", xy=(x2 - d * 0.92, y2), xytext=(x1 + d * 0.92, y1), arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.4))
        else:
            ax.annotate("", xy=(x2, y2 + 0.47), xytext=(x1, y1 - 0.47), arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.4))
    ax.text(5, 0.3, "Guards throughout: no `duration` (leakage)  |  every learned step inside a Pipeline  |  chronological validation  |  decisions logged in docs/",
            ha="center", fontsize=9, color=C_WARN)
    ax.set_title("Project workflow: Bank Marketing campaign response (Group 2026-AI-42)", fontsize=12)
    (DOCS_DIR / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(DOCS_DIR / "figures" / "workflow_diagram.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
