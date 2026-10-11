# Data dictionary: `bank-additional-full.csv`

One row = one telemarketing contact (no client identifier). 41,188 rows x 21 columns, ordered by contact date (May 2008 - Nov 2010).
Machine-readable copy with observed ranges: `reports/tables/data_dictionary.csv` (written by notebook 01 §2).

| Column | Group | Type | Meaning | Known when? | Treatment |
|---|---|---|---|---|---|
| age | Client | numeric | Age in years | A. before contact | keep (numeric) |
| job | Client | categorical | Job type, 12 levels incl. `unknown` | A | one-hot, `unknown` kept |
| marital | Client | categorical | Marital status | A | one-hot; sensitive, audited in notebook 07 |
| education | Client | categorical | Education level | A | ordinal 0-6; `unknown` -> NaN + missing indicator |
| default | Client | categorical | Credit in default? yes / no / unknown | A | one-hot, `unknown` kept |
| housing | Client | categorical | Housing loan? | A | one-hot |
| loan | Client | categorical | Personal loan? | A | one-hot |
| contact | Campaign | categorical | cellular / telephone | B. when planned | one-hot |
| month | Campaign | categorical | Month of last contact | B | one-hot; confounded with year |
| day_of_week | Campaign | categorical | Weekday of last contact | B | one-hot |
| duration | Campaign | numeric | Last call length (s) | **C. only after the call** | **excluded (leakage)** |
| campaign | Campaign | numeric | Contacts to this client in this campaign, incl. last | D. grey zone | keep + `campaign_log`; sensitivity test (02 §7.2) |
| pdays | History | numeric | Days since previous-campaign contact; 999 = never | A | `was_contacted_before` flag, 999 -> 0 |
| previous | History | numeric | Contacts before this campaign | A | keep |
| poutcome | History | categorical | Previous campaign outcome | A | one-hot |
| emp.var.rate | Macro | numeric | Employment variation rate (quarterly) | D | excluded by default (time proxy) |
| cons.price.idx | Macro | numeric | Consumer price index (monthly) | D | excluded by default |
| cons.conf.idx | Macro | numeric | Consumer confidence index (monthly) | D | excluded by default |
| euribor3m | Macro | numeric | 3-month Euribor (daily) | D | excluded by default |
| nr.employed | Macro | numeric | Number of employees (quarterly) | D | excluded by default |
| y | Target | binary | Subscribed to a term deposit? | outcome | target (1 = yes) |

Engineered columns (from `src/features/preprocessing.py`): `was_contacted_before`, `campaign_log`, `education_ord`, and the missing-indicator for `education_ord`.
The full list of model inputs after preprocessing is `reports/tables/final_feature_list.csv` (written by notebook 02 §8).
