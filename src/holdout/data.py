"""Load and validate the Hillstrom email experiment (MineThatData, 2008).

64,000 customers who had bought in the last 12 months were randomly split
into three equal groups for two weeks:
    Mens E-Mail    - received an email featuring men's merchandise
    Womens E-Mail  - received an email featuring women's merchandise
    No E-Mail      - control
Outcomes over the following two weeks: visit (0/1), conversion (0/1), spend (dollars).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"

ARMS = ["No E-Mail", "Mens E-Mail", "Womens E-Mail"]
CONTROL = "No E-Mail"
ARM_LABELS = {"No E-Mail": "No email (control)", "Mens E-Mail": "Men's email", "Womens E-Mail": "Women's email"}
METRICS = {"visit": "Visit rate", "conversion": "Conversion rate", "spend": "Spend per customer"}
PRE_FEATURES = ["recency", "history", "mens", "womens", "newbie", "zip_code", "channel"]
EXPECTED_COLS = {"recency", "history_segment", "history", "mens", "womens", "zip_code",
                 "newbie", "channel", "segment", "visit", "conversion", "spend"}


def find_raw() -> Path:
    files = sorted(RAW_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"Put the Hillstrom CSV in {RAW_DIR}")
    return files[0]


def load(path: Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or find_raw())
    df.columns = [c.strip().lower() for c in df.columns]
    missing = EXPECTED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Not the Hillstrom file - missing columns: {sorted(missing)}")
    if not set(df["segment"].unique()) <= set(ARMS):
        raise ValueError(f"Unexpected arms: {df['segment'].unique()}")
    for c in ["visit", "conversion", "mens", "womens", "newbie"]:
        assert set(df[c].unique()) <= {0, 1}, c
    assert (df["spend"] >= 0).all() and ((df["spend"] > 0) <= (df["conversion"] == 1)).all()
    return df.reset_index(drop=True)


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pre-treatment features only - nothing measured after randomisation."""
    X = df[["recency", "history", "mens", "womens", "newbie"]].astype(float).copy()
    for col in ["zip_code", "channel"]:
        X = X.join(pd.get_dummies(df[col], prefix=col, dtype=float))
    return X


def arm(df: pd.DataFrame, name: str) -> pd.DataFrame:
    return df[df["segment"] == name]


def balance_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Covariates expanded for the balance table (dummies for categoricals)."""
    out = df[["recency", "history", "mens", "womens", "newbie"]].astype(float).copy()
    for col in ["zip_code", "channel"]:
        out = out.join(pd.get_dummies(df[col], prefix=col, dtype=float))
    out["segment"] = df["segment"].to_numpy()
    return out


def rng_order(n: int, seed: int = 0) -> np.ndarray:
    """Hillstrom has no timestamps, so for the sequential replay customers are
    assumed to arrive in random order."""
    return np.random.default_rng(seed).permutation(n)
