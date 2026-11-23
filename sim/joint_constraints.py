"""Opt-in sampled angle audit; never changes model limits or wraps a trajectory.

The provided table is in hardware degrees. A confirmed affine mapping is needed
for enforcement. Checking samples does not prove interpolation, tracking, speed,
torque or collision feasibility. Callers must check the actual executed path too.
"""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


DEFAULT_PATH = Path(__file__).resolve().parents[1] / "config/joint_constraints_observed.json"
JOINT_NAMES = tuple(f"J{i}" for i in range(7))
ROUNDING_TOLERANCE_DEG = 1e-9


def _numbers(value, shape, label):
    array = np.asarray(value, dtype=object)
    if array.shape != shape or any(isinstance(x, (bool, np.bool_)) or
            not isinstance(x, (int, float, np.integer, np.floating)) for x in array.flat):
        raise ValueError(f"{label} must contain numbers with shape {shape}")
    array = array.astype(float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must be finite")
    return array


@dataclass(frozen=True)
class JointConstraints:
    source: str
    bounds_deg: tuple
    confirmed: bool
    sign: tuple | None
    offset_deg: tuple | None
    confirmation_source: str

    def __post_init__(self):
        bounds = _numbers(self.bounds_deg, (7, 2), "bounds_deg")
        if np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError("each lower angle bound must be below its upper bound")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source is required")
        if type(self.confirmed) is not bool:
            raise ValueError("confirmed must be a Boolean")
        if self.sign is not None:
            signs = _numbers(self.sign, (7,), "sign")
            if not np.all(np.isin(signs, [-1, 1])):
                raise ValueError("sign must be exactly -1 or +1")
        if self.offset_deg is not None:
            _numbers(self.offset_deg, (7,), "offset_deg")
        if self.confirmed and (self.sign is None or self.offset_deg is None or
                               not isinstance(self.confirmation_source, str) or
                               not self.confirmation_source.strip()):
            raise ValueError("confirmed mapping requires all signs, offsets and confirmation source")
        object.__setattr__(self, "bounds_deg", tuple(tuple(row) for row in bounds))
        for name in ("sign", "offset_deg"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, tuple(float(x) for x in getattr(self, name)))


def load_joint_constraints(path=DEFAULT_PATH):
    """Read a fixed J0..J6 table. Merely loading it has no simulation side effects."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if type(document.get("schema_version")) is not int or document["schema_version"] != 1:
        raise ValueError("unsupported angle-constraint schema")
    if tuple(document.get("joint_names", ())) != JOINT_NAMES:
        raise ValueError("joint_names must be the ordered J0 through J6")
    mapping = document.get("mapping")
    if not isinstance(mapping, dict):
        raise ValueError("mapping object is required")
    return JointConstraints(document.get("source"), document.get("provided_bounds_deg"),
                            mapping.get("confirmed"), mapping.get("sign"),
                            mapping.get("offset_deg"), mapping.get("confirmation_source", ""))


def model_joint_ranges_rad(constraints):
    """Return the confirmed unwrapped model-coordinate bounds (7, 2)."""
    if not constraints.confirmed:
        raise ValueError("cannot build model limits from an unconfirmed mapping")
    bounds = np.asarray(constraints.bounds_deg)
    with np.errstate(over="ignore", invalid="ignore"):
        ranges = (bounds - np.asarray(constraints.offset_deg)[:, None]) / np.asarray(
            constraints.sign)[:, None]
        ranges = np.deg2rad(np.sort(ranges, axis=1))
    if not np.all(np.isfinite(ranges)):
        raise ValueError("model-coordinate joint bounds overflowed")
    return ranges


def audit_joint_trajectory(q_rad, constraints, *, diagnostic_identity=False):
    """Check every supplied sample, preserving unwrapped multi-turn coordinates.

    For an unconfirmed mapping, diagnostic_identity=True explicitly assumes
    hardware zero/sign equal the model's. Such a report cannot be enforced.
    Offsets, when supplied, are fixed per joint across the entire trajectory.
    """
    raw = np.asarray(q_rad, dtype=object)
    if raw.ndim < 1 or raw.shape[-1] != 7 or raw.size == 0:
        raise ValueError("q_rad must contain one or more samples with final dimension 7")
    q = _numbers(q_rad, raw.shape, "q_rad")
    if not constraints.confirmed and not diagnostic_identity:
        raise ValueError("joint mapping is unconfirmed; only an explicit identity diagnostic is allowed")
    assumed = not constraints.confirmed
    sign = np.ones(7) if assumed else np.asarray(constraints.sign)
    offset = np.zeros(7) if assumed else np.asarray(constraints.offset_deg)
    with np.errstate(over="ignore", invalid="ignore"):
        degrees = np.degrees(q) * sign + offset
    if not np.all(np.isfinite(degrees)):
        raise ValueError("mapped joint angles overflowed")
    samples = degrees.reshape(-1, 7)
    limits = np.asarray(constraints.bounds_deg)
    violations = []
    for sample_index, joint_index in np.argwhere(
            (samples < limits[:, 0] - ROUNDING_TOLERANCE_DEG) |
            (samples > limits[:, 1] + ROUNDING_TOLERANCE_DEG)):
        value = float(samples[sample_index, joint_index])
        low, high = limits[joint_index]
        violations.append({"sample_index": int(sample_index),
            "joint": JOINT_NAMES[joint_index], "value_deg": value,
            "lower_deg": float(low), "upper_deg": float(high),
            "excess_deg": float(max(low - value, value - high, 0.0))})
    return {"scope": "provided_samples_only", "source": constraints.source,
        "mapping_confirmed": constraints.confirmed, "diagnostic_identity_assumption": assumed,
        "eligible_for_enforcement": constraints.confirmed,
        "within_limits": not violations, "sample_count": len(samples),
        "sample_shape": list(q.shape[:-1]), "joint_names": list(JOINT_NAMES),
        "bounds_deg": limits.tolist(), "rounding_tolerance_deg": ROUNDING_TOLERANCE_DEG,
        "raw_min_deg": samples.min(axis=0).tolist(),
        "raw_max_deg": samples.max(axis=0).tolist(), "values_deg": degrees.tolist(),
        "violations": violations}


def enforce_joint_trajectory(q_rad, constraints):
    """Reject unconfirmed mappings or any sampled bound violation; never clip."""
    report = audit_joint_trajectory(q_rad, constraints)
    if not report["within_limits"]:
        first = report["violations"][0]
        raise ValueError(f"joint-angle limits violated in {len(report['violations'])} entries; "
                         f"first: sample {first['sample_index']} {first['joint']} "
                         f"= {first['value_deg']:.12g} degrees")
    return report
