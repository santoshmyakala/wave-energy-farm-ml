"""Layout representations.

The 16 buoys are physically identical, so total farm power does not depend on how the
buoys are numbered. Both representations here are invariant to buoy order.
"""

from __future__ import annotations

import numpy as np

_IU = np.triu_indices(16, k=1)


def canonical_coordinates(coords: np.ndarray) -> np.ndarray:
    """Sort buoys by (x, y) and flatten to 32 values: a fixed ordering for order-sensitive models."""
    order = np.lexsort((coords[..., 1], coords[..., 0]), axis=-1)
    sorted_coords = np.take_along_axis(coords, order[..., None], axis=1)
    return sorted_coords.reshape(len(coords), -1)


def interaction_features(coords: np.ndarray) -> np.ndarray:
    """Order-invariant descriptors of buoy-to-buoy geometry (wave interaction is local and directional).

    - sorted pairwise distances, |dx| and |dy| (120 each)
    - sorted nearest-neighbour distance per buoy (16)
    - spread of the layout (std of x and y, bounding-box extent)
    """
    dx = coords[:, :, None, 0] - coords[:, None, :, 0]
    dy = coords[:, :, None, 1] - coords[:, None, :, 1]
    dist = np.hypot(dx, dy)
    pair_d = np.sort(dist[:, _IU[0], _IU[1]], axis=1)
    pair_dx = np.sort(np.abs(dx[:, _IU[0], _IU[1]]), axis=1)
    pair_dy = np.sort(np.abs(dy[:, _IU[0], _IU[1]]), axis=1)
    masked = dist + np.eye(16)[None] * 1e9
    nn = np.sort(masked.min(axis=2), axis=1)
    spread = np.column_stack([
        coords[..., 0].std(1), coords[..., 1].std(1), np.ptp(coords[..., 0], 1), np.ptp(coords[..., 1], 1),
    ])
    return np.hstack([pair_d, pair_dx, pair_dy, nn, spread])
