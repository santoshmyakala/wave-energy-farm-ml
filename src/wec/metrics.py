from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, r2_score, root_mean_squared_error


def report(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse_kw": float(root_mean_squared_error(y_true, y_pred) / 1000),
        "mape_pct": float(mean_absolute_percentage_error(y_true, y_pred) * 100),
    }
