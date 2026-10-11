"""Project-wide settings. Every notebook imports from here so that a value is defined exactly once.

Business figures (call cost, subscription value, call budget) are ILLUSTRATIVE: the dataset contains no financial
data. A real bank would replace them with its own numbers; the method does not change.
"""
from pathlib import Path

# --------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "Dataset"
ARTIFACTS_DIR = ROOT / "artifacts"
TABLES_DIR = ROOT / "reports" / "tables"
FIGURES_DIR = ROOT / "reports" / "figures"
DOCS_DIR = ROOT / "docs"
for _d in (ARTIFACTS_DIR, TABLES_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

BANK_ADDITIONAL_FILE = DATASET_DIR / "bank-additional-full.csv"   # the file all notebooks model
BANK_FULL_FILE = DATASET_DIR / "bank-full.csv"                    # the file named in the initial proposal (version check only)
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"

# --------------------------------------------------------------------------- reproducibility
SEED = 42

# --------------------------------------------------------------------------- business configuration (ILLUSTRATIVE)
CALL_BUDGET = 0.20             # the call centre can phone the best-scored 20% of the list
COST_PER_CALL = 1.0            # relative cost units
VALUE_PER_SUBSCRIPTION = 10.0  # relative value units -> break-even probability c / v = 0.10

# --------------------------------------------------------------------------- split design (chronological)
TEST_FRACTION = 0.20           # the last 20% of the (chronologically ordered) file is the held-out future period
PILOT_ROWS = 500               # labelled block between train and test; first block for rolling recalibration
N_TS_SPLITS = 4                # forward-chaining folds used for model selection inside the training period

# --------------------------------------------------------------------------- feature policy
EXCLUDE_MACRO = True           # macro indicators are time proxies (see notebooks 01 and 02); set False to test them
ADOPTED_FAMILIES = []          # proposal feature families adopted in notebook 02 section 6 (names from FE_GROUPS); [] = baseline
TARGET = "y"
LEAKY = ["duration"]           # known only after the call has ended

# --------------------------------------------------------------------------- column groups of bank-additional-full.csv
CLIENT_COLS = ["age", "job", "marital", "education", "default", "housing", "loan"]
CAMPAIGN_COLS = ["contact", "month", "day_of_week", "duration", "campaign"]
HISTORY_COLS = ["pdays", "previous", "poutcome"]
MACRO_COLS = ["emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
CATEGORICAL_RAW = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "day_of_week", "poutcome"]
NUMERIC_RAW = ["age", "duration", "campaign", "pdays", "previous"] + MACRO_COLS

MONTH_ORDER = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
MONTH_NUM = {m: i + 1 for i, m in enumerate(MONTH_ORDER)}
EDU_ORDER = {"illiterate": 0, "basic.4y": 1, "basic.6y": 2, "basic.9y": 3,
             "high.school": 4, "professional.course": 5, "university.degree": 6}

# --------------------------------------------------------------------------- plot colours (one palette everywhere)
C_MAIN = "#2f6fed"     # subscribers / primary series
C_GREY = "#9aa5b1"     # non-subscribers / reference series
C_WARN = "#e4572e"     # warnings, drift, leakage
C_OK = "#2a9d8f"       # adopted / good
