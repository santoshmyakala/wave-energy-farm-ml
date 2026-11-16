import numpy as np
import torch

from wec.features import canonical_coordinates, interaction_features
from wec.models_dl import SetTransformer


def _layouts(n=4, seed=0):
    return np.random.default_rng(seed).uniform(0, 566, (n, 16, 2))


def test_representations_are_invariant_to_buoy_order():
    coords = _layouts()
    shuffled = coords[:, np.random.default_rng(1).permutation(16)]
    np.testing.assert_allclose(canonical_coordinates(coords), canonical_coordinates(shuffled))
    np.testing.assert_allclose(interaction_features(coords), interaction_features(shuffled))


def test_set_transformer_total_is_permutation_invariant():
    torch.manual_seed(0)
    model = SetTransformer(dim=32, heads=2, layers=1).eval()
    coords = torch.tensor(_layouts() / 566, dtype=torch.float32)
    perm = torch.randperm(16)
    total, per_buoy = model(coords)
    total_p, per_buoy_p = model(coords[:, perm])
    torch.testing.assert_close(total, total_p, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(per_buoy[:, perm], per_buoy_p, rtol=1e-4, atol=1e-5)  # equivariant
    torch.testing.assert_close(total, per_buoy.sum(-1))


def test_benchmark_smoke(tmp_path):
    from pathlib import Path

    from wec.benchmark import main

    sample = Path(__file__).resolve().parents[1] / "data" / "sample"
    main(["--data-dir", str(sample), "--sites", "Perth", "--max-rows", "400", "--epochs", "1",
          "--models", "mean_baseline", "hgb_coords", "mlp", "--out", str(tmp_path), "--no-mlflow"])
    assert "hgb_coords" in (tmp_path / "results.md").read_text()
