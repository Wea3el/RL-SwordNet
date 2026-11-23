"""Opt-in two-arm geometry with confirmed angles and an explicit legal reset.

This builder does not certify that its reset is collision-free, dynamically
stable, reachable by motion, or ready for fencing training. It never steps a
simulation and leaves the historical duel constructor/defaults unchanged.
"""
from __future__ import annotations

import copy

import mujoco
import numpy as np

from .duel_scene import DuelScene, SITE_NAMES, build_duel_scene
from .joint_constraints import (
    JOINT_NAMES, JointConstraints, enforce_joint_trajectory,
    load_joint_constraints, model_joint_ranges_rad,
)
from .limited_model import build_limited_floor


SLOTS = ("a1", "a2")


def _explicit_pose(value, constraints):
    raw = np.asarray(value, dtype=object)
    if raw.shape != (7,):
        raise ValueError("Each initial pose must be a seven-joint vector")
    enforce_joint_trajectory(value, constraints)
    return np.asarray(value, dtype=float).copy()


def _object_id(model, kind, name):
    index = mujoco.mj_name2id(model, kind, name)
    if index < 0:
        raise ValueError(f"Missing required compiled object: {name}")
    return int(index)


def build_limited_duel(base_distance_m=1.75, initial_qpos=None, constraints=None):
    """Build both slots with exact confirmed bounds, preserving CAD joint zero.

    ``initial_qpos`` is either a shared (7,) vector or a dictionary containing
    exactly ``a1`` and ``a2`` vectors. Values use existing, unwrapped model
    radians. Explicit poses are validated before construction/state writes,
    never clipped or wrapped. None uses the limited-floor interior reset.

    The old builder supplies geometry only; its initial zero state is never
    integrated. A copied spec receives the actual bounds and a legal keyframe
    before the returned model/state is compiled. Joint ref/qpos0 are untouched.
    """
    constraints = load_joint_constraints() if constraints is None else constraints
    if not isinstance(constraints, JointConstraints):
        raise TypeError("constraints must be a JointConstraints instance")
    ranges = model_joint_ranges_rad(constraints)
    if initial_qpos is None:
        home = build_limited_floor(constraints).home
        homes = {slot: home.copy() for slot in SLOTS}
        initial_kind = "shared legal limited-floor interior reset; collision/hold not certified"
    elif isinstance(initial_qpos, dict):
        if set(initial_qpos) != set(SLOTS):
            raise ValueError("Per-slot initial poses require exactly a1 and a2")
        homes = {slot: _explicit_pose(initial_qpos[slot], constraints) for slot in SLOTS}
        initial_kind = "explicit independently checked per-slot reset; motion/hold not certified"
    else:
        home = _explicit_pose(initial_qpos, constraints)
        homes = {slot: home.copy() for slot in SLOTS}
        initial_kind = "explicit shared checked reset; motion/hold not certified"

    staging = build_duel_scene(base_distance_m)
    # Keep any future source physical limits: confirmed input cannot silently
    # broaden limits that have since been introduced into the CAD source.
    for index, original in enumerate(staging.config["floor_assembly_config"]["floor"]["source_joint_ranges"]):
        if original["limited"]:
            lower, upper = original["range_rad"]
            if ranges[index, 0] < lower or ranges[index, 1] > upper:
                raise ValueError(f"{JOINT_NAMES[index]} table would expand a source CAD limit")
    spec = staging.spec.copy()
    key_qpos = np.zeros(staging.model.nq)
    for slot in SLOTS:
        key_qpos[staging.agents[slot]["qpos_ids"]] = homes[slot]
        for index, joint_name in enumerate(JOINT_NAMES):
            name = f"{slot}_{joint_name}"
            joint, actuator = spec.joint(name), spec.actuator(name)
            joint.limited, joint.range = True, ranges[index].copy()
            actuator.ctrllimited, actuator.ctrlrange = True, ranges[index].copy()
    key_name = "limited_duel_home"
    spec.add_key(name=key_name, qpos=key_qpos.tolist(), ctrl=key_qpos.tolist())
    model = spec.compile()
    agents = {}
    for slot in SLOTS:
        prefix = slot + "_"
        joints = np.array([_object_id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix+n) for n in JOINT_NAMES])
        actuators = np.array([_object_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, prefix+n) for n in JOINT_NAMES])
        agents[slot] = {
            "prefix": prefix, "joint_ids": joints,
            "qpos_ids": model.jnt_qposadr[joints].copy(),
            "dof_ids": model.jnt_dofadr[joints].copy(), "actuator_ids": actuators,
            "sites": {n: _object_id(model, mujoco.mjtObj.mjOBJ_SITE, prefix+n) for n in SITE_NAMES},
            "placement": copy.deepcopy(staging.agents[slot]["placement"]),
        }
    data = mujoco.MjData(model)
    key = _object_id(model, mujoco.mjtObj.mjOBJ_KEY, key_name)
    mujoco.mj_resetDataKeyframe(model, data, key)
    for slot, mapping in agents.items():
        if not np.array_equal(data.qpos[mapping["qpos_ids"]], homes[slot]):
            raise ValueError("Compiled keyframe joint order did not preserve the explicit reset")
    mujoco.mj_forward(model, data)
    # Same native position-actuator feedforward, bounded by its unchanged force
    # cap. No qfrc_applied, body gravcomp, or external reset assistance is added.
    for slot, mapping in agents.items():
        ids, dofs = mapping["actuator_ids"], mapping["dof_ids"]
        data.ctrl[ids] = np.clip(homes[slot]+data.qfrc_bias[dofs]/model.actuator_gainprm[ids, 0],
                                 model.actuator_ctrlrange[ids, 0], model.actuator_ctrlrange[ids, 1])
    model.key_ctrl[key] = data.ctrl.copy()
    spec.key(key_name).ctrl = data.ctrl.tolist()
    mujoco.mj_forward(model, data)

    config = copy.deepcopy(staging.config)
    config.update({
        "status": "dual_floor_confirmed_angle_geometry_not_motion_or_training_validation",
        "initial_pose": initial_kind,
        "initial_qpos_rad": {slot: value.tolist() for slot, value in homes.items()},
        "initial_keyframe": key_name,
        "joint_range_status": "confirmed_table_unwrapped_model_angle_limits",
        "geometry_staging": {"builder": "build_duel_scene", "integrated": False,
            "ranges_rad": staging.model.jnt_range.tolist(), "initial_CAD_zero_state_never_integrated": True},
        "joint_constraints": {"source": constraints.source, "mapping_confirmed": constraints.confirmed,
            "confirmation_source": constraints.confirmation_source,
            "joint_names": list(JOINT_NAMES), "provided_bounds_deg": [list(x) for x in constraints.bounds_deg],
            "model_ranges_rad": ranges.tolist(), "sign": list(constraints.sign),
            "offset_deg": list(constraints.offset_deg), "automatic_angle_wrapping": False,
            "scope": "Angles only; no collision, speed, torque, hold, motion or game certification."},
    })
    config["floor_assembly_config"]["floor"].update({
        "joint_range_status": config["joint_range_status"],
        "requested_joint_ranges_rad": ranges.tolist(), "effective_joint_ranges_rad": ranges.tolist(),
        "initial_pose": "See explicit per-slot dual keyframe; no CAD zero reset is returned."})
    config["floor_assembly_config"]["runtime"]["initial_pose"] = (
        "See explicit per-slot dual keyframe; inherited CAD zero is not the returned reset.")
    config["limitations"] = [
        "Joint-angle bounds use the confirmed table; CAD physical zero/ref remains unchanged.",
        "Force caps, gains, inertia, rigid fixture and collision proxies retain existing simulation assumptions.",
        "The supplied reset is angle-valid only; collisions, hold stability and travel require separate checks.",
        "No movement, hit, successful defence, self-play or training is established by this constructor.",
    ]
    return DuelScene(model, data, spec, config, copy.deepcopy(staging.source_hashes), agents)
