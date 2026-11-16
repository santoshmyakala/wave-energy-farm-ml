"""Classical ML models. Each factory returns (feature_fn, estimator)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from wec.features import canonical_coordinates, interaction_features

FeatureFn = Callable[[np.ndarray], np.ndarray]


def _hgb(seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        learning_rate=0.06, max_iter=1500, max_leaf_nodes=63, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=50, random_state=seed,
    )


ML_MODELS: dict[str, Callable[[int], tuple[FeatureFn, object]]] = {
    "mean_baseline": lambda s: (canonical_coordinates, DummyRegressor()),
    "ridge_coords": lambda s: (canonical_coordinates, make_pipeline(StandardScaler(), Ridge(1.0))),
    "hgb_coords": lambda s: (canonical_coordinates, _hgb(s)),
    "hgb_interactions": lambda s: (interaction_features, _hgb(s)),
}
