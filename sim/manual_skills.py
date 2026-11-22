"""Three hand-authored task-space skills, independent of motion datasets.

Guard, straight thrust, and a lateral parry with an upright sword are a small
mechanical test fixture, NOT learned fencing technique or validated defence.
The two roles have different guards to avoid overlapping swords at reset.
Forward-only paths are solved continuously using the real attached-tool sites;
the caller must time, hold and reverse them, then test actual coupled physics.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import mujoco
import numpy as np

from plan1_skills.matching import match_sequence
from .floor import build_floor_assembly


DEFAULT_CONFIG = {
    "schema_version": 1,
    "provenance": "manual Cartesian waypoints; no ASE, motion dataset, imitation learning or RL",
    "joint_limit_status": "hypothetical +/-pi rad, NOT measured hardware bounds",
    "base_distance_m": 1.75,
    "target_world_m": [1.43, 0., 1.05],
    "target_radius_m": .06,
    "attack_grip_start_local_m": [.30, 0., 1.05],
    "attack_grip_end_local_m": [.65, 0., 1.05],
    "attack_axis_local": [1., 0., 0.],
    "defend_grip_start_local_m": [.50, -.25, .75],
    "defend_grip_end_local_m": [.50, .20, .75],
    "defend_axis_local": [0., 0., 1.],
    "attack_seed_rad": [1.89958376, -.23037415, -1.39577222, 1.75932623,
                        -.61548095, -1.46615175, .36924495],
    "defend_seed_rad": [.5570722133146692, -.9015711577861343, -2.5129019073852326,
                        -2.3204492454515653, -.5491356067635892,
                        .5650382599540272, -.15204969258275577],
    "samples": 81,
    "solver_seed": 20261122,
    "solver_max_iterations": 120,
    "maximum_adjacent_joint_change_rad": .20,
    "collision_penetration_m": .001,
    "path_convention": "forward only, guard-to-extension/through-line-side-parry, normalized phase; timing is in config/manual_duel.json",
}


def smoothstep5(phase):
    phase = np.asarray(phase, dtype=float)
    if not np.isfinite(phase).all() or np.any(phase < 0) or np.any(phase > 1):
        raise ValueError("Phase must be finite in [0,1]")
    return phase**3*(10+phase*(-15+6*phase))


def task_reference(start, end, axis, samples=81):
    """Quintic-position, constant-unit-axis forward-only Cartesian reference."""
    start, end, axis = (np.asarray(v, dtype=float) for v in (start, end, axis))
    if any(v.shape != (3,) or not np.isfinite(v).all() for v in (start, end, axis)) or not np.isclose(np.linalg.norm(axis), 1., atol=1e-8):
        raise ValueError("Positions must be finite 3-vectors and sword axis unit")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 3:
        raise ValueError("At least three integer samples are required")
    phase = np.linspace(0., 1., samples)
    position = start+smoothstep5(phase)[:, None]*(end-start)
    return phase, position, np.tile(axis, (samples, 1))


@dataclass
class ManualLibrary:
    config: dict
    single_assembly: object
    guard_attack_q: np.ndarray
    attack_q: np.ndarray
    attack_phase: np.ndarray
    guard_defend_q: np.ndarray
    parry_q: np.ndarray
    parry_phase: np.ndarray
    reports: dict
    attack_grip_local_m: np.ndarray
    attack_axis_local: np.ndarray
    parry_grip_local_m: np.ndarray
    parry_axis_local: np.ndarray


def _report(assembly, result, maximum_step):
    model, data = assembly.model, mujoco.MjData(assembly.model)
    torque, raw_contacts, clearances = [], [], []
    proxy = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "arm_proxy_5")
    plate = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "adapter_plate_collision")
    for index, q in enumerate(result.qpos):
        data.qpos[:] = q
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        torque.append(data.qfrc_bias.copy())
        for contact in data.contact:
            raw_contacts.append({"frame": index,
                "geom1": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)),
                "geom2": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)),
                "distance_m": float(contact.dist)})
        clearances.append(float(mujoco.mj_geomDistance(model, data, proxy, plate, .1, None)))
    torque = np.asarray(torque)
    caps = np.minimum(-model.actuator_forcerange[:, 0], model.actuator_forcerange[:, 1])
    static_over = np.any(np.abs(torque) > caps+1e-9, axis=1)
    summary = result.summary()
    adjacent = np.abs(np.diff(result.qpos, axis=0))
    jumps = np.flatnonzero(np.any(adjacent > maximum_step, axis=1))+1
    summary.update({
        "static_gravity_peak_nm": np.abs(torque).max(0).tolist(),
        "unchanged_debug_torque_caps_nm": caps.tolist(),
        "static_gravity_exceeded_indices": np.flatnonzero(static_over).tolist(),
        "joint_jump_indices": jumps.tolist(),
        "position_error_m": result.position_error_m.tolist(),
        "axis_error_rad": result.angle_error_rad.tolist(),
        "tip_error_m": result.tip_error_m.tolist(),
        "contacts": result.contacts,
        "all_raw_static_contacts": raw_contacts,
        "strict_static_contacts_free": not raw_contacts,
        "proxy5_adapter_minimum_clearance_m": min(clearances),
        "proxy5_adapter_note": "Existing adjacent primitives are nearly tangent at their joint pivot: the maximum possible clearance is approximately 0.14 micrometres. Selected axial-roll branch keeps J6 well inside the contact-free angular interval, not merely below a penetration tolerance. No collider masks or dimensions were changed.",
        "all_frames_retained": True,
        "candidate_passes_single_arm_screen": bool(result.success.all() and not raw_contacts and not static_over.any() and len(jumps) == 0),
        "physical_execution": "NOT RUN; target timing, tracking, coupled sword contact and target protection remain unvalidated",
    })
    return summary


def build_manual_library(config=None):
    """Return explicit forward-only joint paths, never discard failed frames.

    Base A1 is at x=0 facing +X. A2 is at x=base_distance_m with yaw pi;
    all returned Cartesian and joint coordinates are SINGLE-arm local/world
    coordinates before that fixed scene transform. No source-data files are
    opened. Only existing model assets and the assembly calibration are read.
    """
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if config is not None:
        cfg.update(copy.deepcopy(config))
    if not np.isfinite(cfg["base_distance_m"]) or cfg["base_distance_m"] <= 0:
        raise ValueError("Base separation must be finite and positive")
    assembly = build_floor_assembly(joint_ranges=np.tile([-np.pi, np.pi], (7, 1)))
    paths, reports = {}, {}
    for name, role in (("manual_thrust", "attack"), ("manual_side_parry", "defend")):
        phase, grip, axis = task_reference(cfg[f"{role}_grip_start_local_m"],
                                           cfg[f"{role}_grip_end_local_m"],
                                           cfg[f"{role}_axis_local"], cfg["samples"])
        result = match_sequence(assembly.model, grip, axis, assembly.model.jnt_range,
                                initial_q=cfg[f"{role}_seed_rad"], seed=cfg["solver_seed"],
                                max_iterations=cfg["solver_max_iterations"],
                                position_tolerance_m=1e-4,
                                angle_tolerance_rad=np.deg2rad(.05), tip_tolerance_m=.0005,
                                collision_threshold_m=cfg["collision_penetration_m"],
                                interpolation_step_rad=.025, first_frame_starts=1)
        paths[role] = (phase, grip, axis, result.qpos)
        reports[name] = _report(assembly, result, cfg["maximum_adjacent_joint_change_rad"])
    attack_phase, attack_grip, attack_axis, attack_q = paths["attack"]
    parry_phase, parry_grip, parry_axis, parry_q = paths["defend"]
    reports["manual_guard"] = {
        "definition": "role-specific stationary first joint pose; not a common co-located sword reset",
        "attacker_q_rad": attack_q[0].tolist(), "defender_q_rad": parry_q[0].tolist(),
        "initialization_path_checked": False,
        "mutual_reset_contacts": "must be checked after building the coupled scene",
    }
    return ManualLibrary(cfg, assembly, attack_q[0].copy(), attack_q, attack_phase,
                         parry_q[0].copy(), parry_q, parry_phase, reports,
                         attack_grip, attack_axis, parry_grip, parry_axis)
