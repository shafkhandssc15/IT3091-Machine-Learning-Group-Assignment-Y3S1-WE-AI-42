# Reproducibility notes

* Seed: `SEED = 42` (`src/config.py`) for every split, fold, model and clustering.
* Dataset fingerprint (SHA-256 of content) is printed in notebook 01 §1 and notebook 08 §4. The official file has 41,188 rows x 21 columns and 12 exact duplicates.
* Run order: 01 -> 08. Notebooks 04-07 read files written by earlier notebooks (see `project_map.md`); a fresh clone must run 01-08 in order.
* Everything runs from the project root or from `notebooks/` (each notebook adds the project root to `sys.path`).
* `FAST = True` (notebooks 02-04) reduces folds for a quick dry run; numbers then differ slightly from the full run.
* Timings are machine-specific; every other number reproduces for the same package versions. Print versions with notebook 08 §4.
* Tests: `python -m unittest discover -s tests -t .` (synthetic stand-in data; no download).
* Colab: upload the whole project folder (or `git clone`), `pip install -r requirements.txt`, then open the notebooks. The dataset downloads on first use.
