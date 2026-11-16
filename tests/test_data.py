from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wec.data import COLUMNS, P_COLS, TOTAL, clean, coordinates, load_site, split_indices

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample"


@pytest.mark.parametrize("site", ["Adelaide", "Perth", "Sydney", "Tasmania"])
def test_sample_files_match_schema(site):
    df = load_site(SAMPLE / f"{site}_Data.csv")
    assert list(df.columns) == COLUMNS
    assert coordinates(df).shape == (len(df), 16, 2)


def test_clean_drops_duplicates_and_inconsistent_totals():
    df = load_site(SAMPLE / "Perth_Data.csv").head(10)
    broken = df.iloc[[0]].copy()
    broken[TOTAL] *= 1.05
    dirty = pd.concat([df, df.iloc[[1, 2]], broken], ignore_index=True)
    out, rep = clean(dirty)
    assert rep.exact_duplicates == 2 and rep.inconsistent_totals == 1 and len(out) == 10
    np.testing.assert_allclose(out[P_COLS].sum(axis=1), out[TOTAL], rtol=1e-4)


@pytest.mark.parametrize("method", ["random", "blocked"])
def test_splits_are_disjoint_and_complete(method):
    tr, va, te = split_indices(1000, method)
    assert len(set(tr) | set(va) | set(te)) == 1000
    assert not (set(tr) & set(te)) and not (set(va) & set(te))
    if method == "blocked":
        assert te.min() > tr.max()
