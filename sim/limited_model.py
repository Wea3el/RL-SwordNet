"""Opt-in floor model with the confirmed, unwrapped joint-angle limits.

This is a geometry/state constructor, not a motion or hardware validation.
It never integrates the provisional construction model, wraps angles, changes
the motor force caps, or repairs an explicitly supplied out-of-range pose.
"""
from __future__ import annotations

import copy

import mujoco
import numpy as np

from .assembly import Assembly
from .floor import build_floor_assembly
from .joint_constraints import (
    JOINT_NAMES,
    JointConstraints,
    enforce_joint_trajectory,
    load_joint_constraints,
    model_joint_ranges_rad,
)


def build_limited_floor(constraints=None, initial_qpos=None):
    """Build the unchanged floor arm with confirmed bounds and a legal reset.

    ``initial_qpos`` uses the existing J0..J6 model coordinates in radians.
    Explicit values are checked, not clipped. If absent, the CAD zero pose is
    projected into a small interior margin (one degree, or a quarter of a
    narrower range). That projection defines a reset only, not a validated
    route from CAD zero to the new pose.

    The legacy floor constructor requires its ranges to contain CAD zero.
    A temporary geometry-only construction therefore extends bounds to zero;
    its spec is copied and replaced with the actual confirmed bounds before
    creating the returned state. No ``mj_step`` is performed at either stage.
    """
    constraints = load_joint_constraints() if constraints is None else constraints
    if not isinstance(constraints, JointConstraints):
        raise TypeError("constraints must be a JointConstraints instance")
    ranges = model_joint_ranges_rad(constraints)
    if initial_qpos is None:
        margin = np.minimum(np.deg2rad(1.), (ranges[:, 1] - ranges[:, 0]) / 4.)
        home = np.clip(np.zeros(7), ranges[:, 0] + margin, ranges[:, 1] - margin)
        initial_kind = "CAD zero projected to the legal interior; reset only"
    else:
        # Enforcement also rejects booleans, nonfinite values and wrong shapes.
        raw = np.asarray(initial_qpos, dtype=object)
        if raw.shape != (7,):
            raise ValueError("initial_qpos must be a seven-joint vector")
        enforce_joint_trajectory(initial_qpos, constraints)
        home = np.asarray(initial_qpos, dtype=float).copy()
        initial_kind = "explicit legal reset; travel to this pose not validated"
    enforce_joint_trajectory(home, constraints)

    staging_ranges = np.column_stack((np.minimum(ranges[:, 0], 0.),
                                      np.maximum(ranges[:, 1], 0.)))
    staging = build_floor_assembly(joint_ranges=staging_ranges)
    # Preserve pre-existing physical limits if they are ever added to the CAD.
    for index, source in enumerate(staging.config["floor"]["source_joint_ranges"]):
        if source["limited"]:
            original = source["range_rad"]
            if ranges[index, 0] < original[0] or ranges[index, 1] > original[1]:
                raise ValueError(f"{JOINT_NAMES[index]} table would expand a source CAD limit")
    spec = staging.spec.copy()
    for index, name in enumerate(JOINT_NAMES):
        joint, actuator = spec.joint(name), spec.actuator(name)
        joint.limited, joint.range = True, ranges[index].copy()
        actuator.ctrllimited, actuator.ctrlrange = True, ranges[index].copy()
    spec.key("floor_home").qpos = home.tolist()
    spec.key("floor_home").ctrl = home.tolist()
    model = spec.compile()
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "floor_home")
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)
    # Feedforward remains inside the same native, force-limited actuator.
    # Saturation can prevent holding the pose; no dynamic hold is claimed.
    data.ctrl[:] = np.clip(home + data.qfrc_bias / model.actuator_gainprm[:, 0],
                           model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
    model.key_ctrl[key] = data.ctrl.copy()
    spec.key("floor_home").ctrl = data.ctrl.tolist()
    mujoco.mj_forward(model, data)

    config = copy.deepcopy(staging.config)
    config["runtime"]["initial_pose"] = initial_kind
    config["runtime"]["initial_qpos_rad"] = home.tolist()
    config["floor"].update({
        "joint_range_status": "confirmed_table_unwrapped_model_angle_limits",
        "requested_joint_ranges_rad": ranges.tolist(),
        "effective_joint_ranges_rad": ranges.tolist(),
        "initial_pose": initial_kind,
        "construction_note": "Zero-containing staging was never integrated; returned limits are the confirmed table.",
    })
    config["joint_constraints"] = {
        "source": constraints.source,
        "mapping_confirmed": constraints.confirmed,
        "confirmation_source": constraints.confirmation_source,
        "joint_names": list(JOINT_NAMES),
        "provided_bounds_deg": [list(row) for row in constraints.bounds_deg],
        "model_ranges_rad": ranges.tolist(),
        "sign": list(constraints.sign),
        "offset_deg": list(constraints.offset_deg),
        "scope": "Joint angles only; no speed, torque, collision, hold or movement certification.",
        "automatic_angle_wrapping": False,
    }
    return Assembly(model, data, config, home, spec, staging.source_hashes.copy())
