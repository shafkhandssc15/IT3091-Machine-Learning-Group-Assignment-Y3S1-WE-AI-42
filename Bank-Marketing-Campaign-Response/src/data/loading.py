"""Loading the UCI Bank Marketing data and recording which exact file was used.

Order of lookup for ``bank-additional-full.csv``:
1. the path in the environment variable ``BANK_CSV`` (if set),
2. ``Dataset/bank-additional-full.csv`` inside the project,
3. ``bank-additional-full.csv`` in the current working directory (e.g. a file uploaded to Colab),
4. a download from the UCI repository, which is then cached in ``Dataset/``.
"""
import hashlib
import io
import os
import zipfile
import urllib.request
from pathlib import Path

import pandas as pd

from src.config import BANK_ADDITIONAL_FILE, BANK_FULL_FILE, UCI_ZIP_URL


def _candidates(env_var, default_path, filename):
    paths = []
    if os.environ.get(env_var):
        paths.append(Path(os.environ[env_var]))
    paths += [default_path, Path.cwd() / filename]
    return paths


def _read_semicolon_csv(path):
    d = pd.read_csv(path, sep=";")
    if d.shape[1] == 1:                       # some mirrors ship comma-separated files
        d = pd.read_csv(path, sep=",")
    return d


def _download_zip_member(inner_zip, member):
    outer = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(UCI_ZIP_URL, timeout=120).read()))
    inner = zipfile.ZipFile(io.BytesIO(outer.read(inner_zip)))
    with inner.open(member) as f:
        return f.read()


def load_raw(verbose=True):
    """Return ``bank-additional-full.csv`` exactly as published (41,188 rows x 21 columns, file order preserved).

    Nothing is cleaned here. Cleaning is a documented, tested step of notebook 02.
    """
    for p in _candidates("BANK_CSV", BANK_ADDITIONAL_FILE, "bank-additional-full.csv"):
        if p.exists():
            if verbose:
                print("Loading local file:", p.name)
            return _read_semicolon_csv(p)
    if verbose:
        print("No local copy found. Downloading from", UCI_ZIP_URL)
    payload = _download_zip_member("bank-additional.zip", "bank-additional/bank-additional-full.csv")
    BANK_ADDITIONAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    BANK_ADDITIONAL_FILE.write_bytes(payload)
    if verbose:
        print("Cached a copy at", BANK_ADDITIONAL_FILE)
    return _read_semicolon_csv(BANK_ADDITIONAL_FILE)


def load_bank_full(verbose=True):
    """Return ``bank-full.csv`` (the file named in the initial proposal), with its target column called ``y``.

    Used only for the dataset-version check in notebook 03.
    """
    for p in _candidates("BANKFULL_CSV", BANK_FULL_FILE, "bank-full.csv"):
        if p.exists():
            if verbose:
                print("Loading local file:", p.name)
            return _read_semicolon_csv(p).rename(columns={"deposit": "y"})
    if verbose:
        print("No local copy found. Downloading from", UCI_ZIP_URL)
    payload = _download_zip_member("bank.zip", "bank-full.csv")
    BANK_FULL_FILE.parent.mkdir(parents=True, exist_ok=True)
    BANK_FULL_FILE.write_bytes(payload)
    return _read_semicolon_csv(BANK_FULL_FILE).rename(columns={"deposit": "y"})


def fingerprint(df):
    """SHA-256 of the data content: the same value on two machines means the same data was analysed."""
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()
