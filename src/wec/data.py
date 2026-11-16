"""Loading, validating, cleaning and splitting the UCI Wave Energy Converters data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from wec import N_BUOYS

X_COLS = [f"X{i}" for i in range(1, N_BUOYS + 1)]
Y_COLS = [f"Y{i}" for i in range(1, N_BUOYS + 1)]
P_COLS = [f"P{i}" for i in range(1, N_BUOYS + 1)]
TOTAL = "Powerall"
COLUMNS = X_COLS + Y_COLS + P_COLS + [TOTAL]


@dataclass
class CleaningReport:
    rows_in: int
    exact_duplicates: int
    inconsistent_totals: int
    rows_out: int


def load_site(path: str | Path) -> pd.DataFrame:
    """Read one site's CSV (published without a header row) and validate its shape."""
    df = pd.read_csv(path, header=None)
    if df.shape[1] != len(COLUMNS):
        raise ValueError(f"{path}: expected {len(COLUMNS)} columns, found {df.shape[1]}")
    df.columns = COLUMNS
    if df.isna().any().any():
        raise ValueError(f"{path}: contains missing values")
    return df


def clean(df: pd.DataFrame, rel_tol: float = 1e-4) -> tuple[pd.DataFrame, CleaningReport]:
    """Drop exact duplicate rows and rows whose total disagrees with the sum of buoy powers."""
    rows_in = len(df)
    deduped = df.drop_duplicates()
    mismatch = (deduped[P_COLS].sum(axis=1) - deduped[TOTAL]).abs() > rel_tol * deduped[TOTAL]
    out = deduped.loc[~mismatch].reset_index(drop=True)
    return out, CleaningReport(rows_in, rows_in - len(deduped), int(mismatch.sum()), len(out))


def coordinates(df: pd.DataFrame) -> np.ndarray:
    """Return buoy coordinates as an array of shape (n_layouts, 16, 2)."""
    return np.stack([df[X_COLS].to_numpy(float), df[Y_COLS].to_numpy(float)], axis=-1)


def buoy_powers(df: pd.DataFrame) -> np.ndarray:
    return df[P_COLS].to_numpy(float)


def split_indices(
    n: int, method: str = "random", seed: int = 42, val: float = 0.15, test: float = 0.15
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train/validation/test indices.

    ``random`` shuffles rows. ``blocked`` keeps file order and holds out the final rows,
    a stricter check because consecutive layouts in the published files are often similar.
    """
    idx = np.arange(n)
    if method == "random":
        idx = np.random.default_rng(seed).permutation(n)
    elif method != "blocked":
        raise ValueError(f"unknown split method '{method}'")
    n_test, n_val = int(n * test), int(n * val)
    return idx[: n - n_val - n_test], idx[n - n_val - n_test : n - n_test], idx[n - n_test :]
