"""Offline CAD estimate of the J3 straight-arm coordinate offset.

The user's hardware/table zero is a straight elbow.  The source CAD zero is
not.  J2 and J4 are the roll axes of the adjoining arm tubes; their signed angle
about the J3 hinge is therefore the elbow bend.  This module samples only
forward kinematics.  It never steps physics, edits limits, or certifies a pose,
trajectory, physical range, actuator, or real-machine encoder calibration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
JOINT_NAMES = tuple(f"J{i}" for i in range(7))
PROBE_QPOS_RAD = (
    (0., 0., 0., 0., 0., 0., 0.),
    (.7, .5, 1.2, -.3, .4, .9, -.2),
    (2.6, 1.3, 5., 1.5, 2.8, 1.6, .6),
    (5.8, 2.4, .2, .4, 5.6, .5, -.8),
)


def _joint_ids(model, prefix):
    ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                   prefix + name) for name in JOINT_NAMES])
    if np.any(ids < 0):
        raise ValueError(f"Expected all seven hinge joints with prefix {prefix!r}")
    if np.any(model.jnt_type[ids] != mujoco.mjtJoint.mjJNT_HINGE):
        raise ValueError("J3 calibration requires seven hinge coordinates")
    return ids


def _angle(upper, lower, hinge):
    # Projection makes the angle definition explicit.  Perpendicularity is
    # checked separately rather than silently assuming arbitrary roll axes
    # define a straight elbow.
    upper = upper - np.dot(upper, hinge) * hinge
    lower = lower - np.dot(lower, hinge) * hinge
    if min(np.linalg.norm(upper), np.linalg.norm(lower)) < 1e-10:
        raise ValueError("Tube roll axis is parallel to the elbow hinge")
    upper, lower = upper / np.linalg.norm(upper), lower / np.linalg.norm(lower)
    return float(np.arctan2(np.dot(hinge, np.cross(upper, lower)),
                            np.dot(upper, lower)))


def measure_j3_zero(model, *, prefix="", qposes=PROBE_QPOS_RAD):
    """Measure offset using scratch data and arbitrary CAD-coordinate probes.

    No shared joint-constraint configuration is consulted.  Forward kinematics
    does not enforce joint limits, so narrowing those limits does not bias this
    measurement.  All probe poses here are measurement samples, not legal or
    dynamically reachable states.  Original model parameters remain unchanged.
    """
    values = np.asarray(qposes, dtype=float)
    if values.ndim != 2 or values.shape[1] != 7 or len(values) < 1 or not np.isfinite(values).all():
        raise ValueError("qposes must be a nonempty finite (N, 7) array")
    ids = _joint_ids(model, prefix)
    qids = model.jnt_qposadr[ids]
    data = mujoco.MjData(model)
    rows = []
    for q in values:
        data.qpos[qids] = q
        mujoco.mj_forward(model, data)
        upper, hinge, lower = (data.xaxis[ids[i]].copy() for i in (2, 3, 4))
        off_rad = _angle(upper, lower, hinge) - q[3]
        # atan2 gives a principal angle; remove only this measurement ambiguity,
        # never wrap the actual model coordinates or a motion trajectory.
        off_rad = float(np.arctan2(np.sin(off_rad), np.cos(off_rad)))
        offset_deg = float(np.degrees(off_rad))
        data.qpos[qids[3]] = -off_rad
        mujoco.mj_forward(model, data)
        zero_upper, zero_lower = data.xaxis[ids[2]], data.xaxis[ids[4]]
        residual = float(np.degrees(np.arctan2(np.linalg.norm(np.cross(zero_upper, zero_lower)),
                                               np.dot(zero_upper, zero_lower))))
        rows.append({
            "probe_qpos_rad_J0_to_J6": q.tolist(),
            "offset_deg": offset_deg,
            "upper_axis_dot_hinge": float(upper @ hinge),
            "lower_axis_dot_hinge": float(lower @ hinge),
            "straight_cad_J3_rad": -off_rad,
            "straight_axis_alignment_residual_deg": residual,
        })
    offsets = np.array([r["offset_deg"] for r in rows])
    result = {
        "prefix": prefix,
        "offset_deg": float(offsets.mean()),
        "offset_spread_deg": float(np.ptp(offsets)),
        "maximum_straight_axis_alignment_residual_deg": max(
            r["straight_axis_alignment_residual_deg"] for r in rows),
        "maximum_axis_hinge_dot_abs": max(max(abs(r["upper_axis_dot_hinge"]),
                                               abs(r["lower_axis_dot_hinge"])) for r in rows),
        "samples": rows,
    }
    result["geometry_consistent"] = bool(result["offset_spread_deg"] < 1e-8
        and result["maximum_straight_axis_alignment_residual_deg"] < 1e-8
        and result["maximum_axis_hinge_dot_abs"] < 1e-8)
    return result


def calibration_report():
    """Compare unrounded floor construction and both opposed robot slots."""
    from .floor import build_floor_assembly
    from .duel_scene import build_duel_scene

    # Explicit construction windows are only for forward-kinematics sampling.
    # They are not proposed limits and this report never integrates the model.
    sampling_ranges = np.tile([-2 * np.pi, 2 * np.pi], (7, 1))
    floor = build_floor_assembly(joint_ranges=sampling_ranges)
    duel = build_duel_scene(joint_ranges=sampling_ranges)
    measurements = {"floor": measure_j3_zero(floor.model),
                    "a1": measure_j3_zero(duel.model, prefix="a1_"),
                    "a2": measure_j3_zero(duel.model, prefix="a2_")}
    offsets = np.array([m["offset_deg"] for m in measurements.values()])
    offset = float(offsets.mean())
    hashes = dict(floor.source_hashes)
    for path in ("sim/j3_zero_calibration.py", "sim/floor.py", "sim/duel_scene.py",
                 "config/assembly.json"):
        hashes[path] = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
    return {
        "schema_version": 1,
        "status": "offline_source_CAD_forward_kinematics_estimate_not_hardware_calibration",
        "method": "signed J2-roll-axis to J4-roll-axis angle about J3 hinge; zero is parallel tube axes",
        "mapping_assumption": "User confirms table J3 zero means straight elbow; positive direction unchanged",
        "formula": "hardware_J3_deg = CAD_J3_rad * 180 / pi + offset_deg",
        "offset_deg": offset,
        "straight_CAD_J3_deg": -offset,
        "provided_hardware_bounds_deg": [-20., 120.],
        "derived_CAD_bounds_deg": [-20. - offset, 120. - offset],
        "cross_model_offset_spread_deg": float(np.ptp(offsets)),
        "geometry_consistent": bool(all(m["geometry_consistent"] for m in measurements.values())
                                    and np.ptp(offsets) < 1e-8),
        "measurements": measurements,
        "source_sha256": hashes,
        "mujoco_version": mujoco.__version__,
        "physics_integration_steps": 0,
        "shared_joint_constraints_read": False,
        "limits_or_model_parameters_modified": False,
        "admitted_to_training": False,
        "limitations": [
            "CAD tube-axis alignment is not a measured real encoder zero or physical stop calibration.",
            "Other joint offsets are not measured or changed.",
            "Kinematic samples do not establish collision clearance, holding torque, reachable motion or legal skills.",
            "Old action certificates cannot be transferred merely from this coordinate measurement.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="optional new JSON report (refuses overwrite)")
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("output already exists; preserve prior evidence")
    report = calibration_report()
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(text)
    print(text, end="")
    if not report["geometry_consistent"]:
        raise SystemExit("CAD geometry failed the offset-consistency checks")


if __name__ == "__main__":
    main()
