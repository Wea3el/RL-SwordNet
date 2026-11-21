"""Dependency-light shinai visual from the supplied STEP's measured profile.

The result preserves the axial silhouette, not the four bamboo sections or
spline-trimmed BRep. It is a visual approximation, not a collision/inertia model.
Runtime requires only the checked-in numerical profile and NumPy, not the STEP,
OpenCASCADE, FreeCAD, SciPy or a downloaded mesh.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


PROFILE_PATH = Path(__file__).resolve().parents[1] / "assets" / "shinai_profile.json"
_DIMENSIONS = (
    "total_length_m", "handle_length_m", "grip_from_pommel_m", "tip_from_grip_m",
    "guard_near_m", "guard_far_m", "guard_radius_m", "grip_radius_m",
    "tip_radius_m", "blade_length_m",
)


def _validated_points(profile):
    if (not isinstance(profile, dict) or profile.get("schema_version") != 1
            or profile.get("units") != "m"):
        raise ValueError("Expected a schema-1 shinai profile in meters")
    try:
        dimensions = profile["dimensions"]
        values = np.asarray([dimensions[key] for key in _DIMENSIONS], dtype=float)
        points = np.asarray(profile["profile_s_r_m"], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Malformed shinai dimensions or profile") from error
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Shinai dimensions must be finite and positive")
    if (points.ndim != 2 or points.shape[1] != 2 or not 2 <= len(points) <= 2048
            or not np.isfinite(points).all() or np.any(points[:, 1] <= 0)):
        raise ValueError("Expected finite positive-radius [s, r] profile rows")
    # Remove identical consecutive stations, NOT same-s different-radius rows:
    # those radial steps encode the guard's annular front/back faces.
    keep = np.r_[True, np.any(np.abs(np.diff(points, axis=0)) > 1e-12, axis=1)]
    points = points[keep]
    if len(points) < 2 or np.any(np.diff(points[:, 0]) < 0):
        raise ValueError("Shinai axial stations must be nondecreasing")
    first, last = points[0, 0], points[-1, 0]
    if (not np.isclose(first, -dimensions["grip_from_pommel_m"], rtol=0, atol=1e-9)
            or not np.isclose(last, dimensions["tip_from_grip_m"], rtol=0, atol=1e-9)
            or not np.isclose(last-first, dimensions["total_length_m"], rtol=0, atol=1e-9)):
        raise ValueError("Profile length/origin differs from declared dimensions")
    if (dimensions["guard_near_m"] >= dimensions["guard_far_m"]
            or not np.isclose(np.max(points[:, 1]), dimensions["guard_radius_m"], rtol=0, atol=1e-9)):
        raise ValueError("Inconsistent guard dimensions")
    for key in ("guard_near_m", "guard_far_m"):
        if not np.any(np.isclose(points[:, 0], dimensions[key], rtol=0, atol=1e-9)
                      & np.isclose(points[:, 1], dimensions["guard_radius_m"], rtol=0, atol=1e-9)):
            raise ValueError("Profile is missing a declared guard outer edge")
    if not np.any(np.isclose(points[:, 0], 0., rtol=0, atol=1e-12)
                  & np.isclose(points[:, 1], dimensions["grip_radius_m"], rtol=0, atol=1e-9)):
        raise ValueError("Profile is missing its declared grip cross section")
    return points


def load_profile() -> dict:
    """Load and validate the numerical STEP-derived profile and provenance."""
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    _validated_points(profile)
    return profile


def revolved_mesh(profile=None, radial_segments=64):
    """Return (vertices[N,3], faces[M,3]) in meters for a closed visual mesh.

    Grip is the origin, tip is +Z, and the pommel is -Z. Same-Z radial steps
    create nondegenerate annular guard faces. End caps use one center vertex
    each instead of duplicate zero-radius rings. Triangle winding is outward.
    This mesh must not be mistaken for the actual slotted CAD volume/inertia.
    """
    if type(radial_segments) is not int or not 8 <= radial_segments <= 512:
        raise ValueError("radial_segments must be an integer between 8 and 512")
    if profile is None:
        profile = load_profile()
    points = _validated_points(profile)
    angles = np.arange(radial_segments, dtype=float) * (2*np.pi/radial_segments)
    rings = np.empty((len(points), radial_segments, 3), dtype=np.float64)
    rings[..., 0] = points[:, 1, None] * np.cos(angles)
    rings[..., 1] = points[:, 1, None] * np.sin(angles)
    rings[..., 2] = points[:, 0, None]
    vertices = np.concatenate([
        rings.reshape(-1, 3), [[0., 0., points[0, 0]], [0., 0., points[-1, 0]]],
    ])
    faces = []
    for ring in range(len(points)-1):
        for index in range(radial_segments):
            following = (index+1) % radial_segments
            a = ring*radial_segments + index
            b = ring*radial_segments + following
            c = (ring+1)*radial_segments + following
            d = (ring+1)*radial_segments + index
            faces.extend(((a, b, d), (b, c, d)))
    bottom, top = len(vertices)-2, len(vertices)-1
    last_ring = (len(points)-1)*radial_segments
    for index in range(radial_segments):
        following = (index+1) % radial_segments
        faces.append((bottom, following, index))
        faces.append((top, last_ring+index, last_ring+following))
    return vertices, np.asarray(faces, dtype=np.int32)
