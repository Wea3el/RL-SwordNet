"""Intersect model-coordinate authoring windows without changing trajectories.

Both arrays use existing, unwrapped model radians.  A search preference may
narrow a physical/interior box, never replace it with a wider box.  Empty
intersections are errors, not a request to clip a pose or relax a limit.
"""
from __future__ import annotations

import numpy as np


def intersect_authoring_bounds(outer, requested):
    outer = np.asarray(outer, dtype=float)
    requested = np.asarray(requested, dtype=float)
    if (outer.shape != (7, 2) or requested.shape != (7, 2)
            or not np.isfinite(outer).all() or not np.isfinite(requested).all()
            or np.any(outer[:, 0] >= outer[:, 1])
            or np.any(requested[:, 0] >= requested[:, 1])):
        raise ValueError('Finite ordered seven-joint model-coordinate bounds required')
    result = np.column_stack((np.maximum(outer[:, 0], requested[:, 0]),
                              np.minimum(outer[:, 1], requested[:, 1])))
    if np.any(result[:, 0] >= result[:, 1]):
        raise ValueError('Authoring window has no interior within the allowed bounds')
    return result
