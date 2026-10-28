"""Download, parse and normalise the SMS Spam Collection dataset.

The dataset contains 5,574 labelled English SMS messages (ham / spam) and is
published by the UCI Machine Learning Repository. If the automatic download
fails, put a CSV with ``label,message`` columns at ``data/sms_spam.csv`` and
everything else keeps working.

Usage:
    python -m src.dataset          # download only
"""

from __future__ import annotations

import io
import sys
import urllib.request
import zipfile

import pandas as pd

from .config import DATA_DIR, HAM, LABELS, RAW_DATA_FILE, SPAM

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
GITHUB_TSV_URL = (
    "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/"
    "master/data/sms.tsv"
)
SOURCES = (
    ("UCI Machine Learning Repository", UCI_ZIP_URL),
    ("GitHub mirror", GITHUB_TSV_URL),
)

_USER_AGENT = "sms-spam-detection/1.0 (+https://github.com)"
_TIMEOUT = 60


def _download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
        return response.read()


def _read_zip(payload: bytes) -> pd.DataFrame:
    """The UCI zip holds a single tab-separated file named SMSSpamCollection."""
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        name = next(
            entry
            for entry in archive.namelist()
            if "SMSSpamCollection" in entry and not entry.endswith("/")
        )
        content = archive.read(name).decode("latin-1")
    return pd.read_csv(io.StringIO(content), sep="\t", header=None)


def _read_tsv(payload: bytes) -> pd.DataFrame:
    content = payload.decode("utf-8", errors="replace")
    return pd.read_csv(io.StringIO(content), sep="\t", header=None)


def _find_label_column(df: pd.DataFrame) -> str:
    """Pick the column whose values are mostly 'ham' / 'spam'."""
    best_col, best_ratio = None, 0.0
    for column in df.columns:
        values = df[column].astype(str).str.strip().str.lower()
        ratio = values.isin(LABELS).mean()
        if ratio > best_ratio:
            best_col, best_ratio = column, ratio
    if best_col is None or best_ratio < 0.5:
        raise ValueError(
            "Could not find a 'ham'/'spam' label column in the downloaded file."
        )
    return best_col


def _find_text_column(df: pd.DataFrame, label_column: str) -> str:
    """The message column is the non-label column with the longest values."""
    candidates = [column for column in df.columns if column != label_column]
    if not candidates:
        raise ValueError("The dataset has no message column.")
    return max(candidates, key=lambda col: df[col].astype(str).str.len().mean())


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Return a clean ``label`` / ``message`` frame, dropping junk and duplicates."""
    label_column = _find_label_column(df)
    text_column = _find_text_column(df, label_column)

    clean = pd.DataFrame(
        {
            "label": df[label_column].astype(str).str.strip().str.lower(),
            "message": df[text_column].astype(str).str.strip(),
        }
    )
    clean = clean[clean["label"].isin(LABELS)]
    clean = clean[clean["message"].str.len() > 0]
    clean = clean.drop_duplicates(subset=["message"]).reset_index(drop=True)
    return clean


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """US spelling alias for :func:`normalise`."""
    return normalise(df)


def load_dataframe(path=RAW_DATA_FILE) -> pd.DataFrame:
    """Read the normalised dataset from disk."""
    df = pd.read_csv(path)
    missing = {"label", "message"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return df


def download_dataset(force: bool = False, verbose: bool = True) -> pd.DataFrame:
    """Download the dataset and cache the normalised version in ``data/``."""
    if RAW_DATA_FILE.exists() and not force:
        return load_dataframe()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []

    for name, url in SOURCES:
        try:
            if verbose:
                print(f"Downloading dataset from {name} ...", flush=True)
            payload = _download(url)
            raw = _read_zip(payload) if url.endswith(".zip") else _read_tsv(payload)
            df = normalise(raw)
            if len(df) < 1000:
                raise ValueError(f"only {len(df)} rows parsed, source looks broken")
            df.to_csv(RAW_DATA_FILE, index=False)
            if verbose:
                _print_stats(df)
            return df
        except Exception as exc:  # noqa: BLE001 - try every mirror, then report
            errors.append(f"{name}: {exc}")
            if verbose:
                print(f"  failed: {exc}", file=sys.stderr, flush=True)

    raise RuntimeError(
        "Could not download the SMS Spam Collection dataset.\n"
        + "\n".join(f"  - {e}" for e in errors)
        + f"\n\nManual fix: save a CSV with 'label' and 'message' columns to:\n  {RAW_DATA_FILE}"
    )


def ensure_dataset(force: bool = False, verbose: bool = True) -> pd.DataFrame:
    """Return the dataset, downloading it on first use (alias of download_dataset)."""
    return download_dataset(force=force, verbose=verbose)


def _print_stats(df: pd.DataFrame) -> None:
    counts = df["label"].value_counts()
    print(
        f"  {len(df):,} messages "
        f"({counts.get(HAM, 0):,} ham / {counts.get(SPAM, 0):,} spam)"
    )


def dataset_stats(df: pd.DataFrame) -> dict:
    counts = df["label"].value_counts()
    return {
        "rows": int(len(df)),
        "ham": int(counts.get(HAM, 0)),
        "spam": int(counts.get(SPAM, 0)),
        "spam_ratio": round(float(counts.get(SPAM, 0) / len(df)), 4) if len(df) else 0.0,
    }


if __name__ == "__main__":
    frame = download_dataset(force="--force" in sys.argv)
    print(frame["label"].value_counts().to_string())
