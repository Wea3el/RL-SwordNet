"""Finite, identical local Guard/Attack/Parry actions for either fixed arm.

All actions start and finish at ONE exact joint Guard. Paths depend only on
authored local geometry and their own elapsed time: no opponent phase, future
commands, scoring event or scheduled attack deadline is accepted here. This is
an action-space candidate, not evidence of autonomous defence or RL validity.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import mujoco
import numpy as np

from plan1_skills.matching import match_sequence, tool_fk
from .floor import build_floor_assembly


DEFAULT_CONFIG = {
    "schema_version": 1,
    "provenance": "hand-authored local Cartesian paths; no ASE, dataset or opponent-future input",
    "guard_q_rad": [.5570722133146692, -.9015711577861343, -2.5129019073852326,
                    -2.3204492454515653, -.5491356067635892, .5650382599540272,
                    -.15204969258275577],
    "attack_ready_grip_local_m": [.30, 0., 1.05],
    "attack_extended_grip_local_m": [.65, 0., 1.05],
    "attack_axis_local": [1., 0., 0.],
    "parry_extended_grip_local_m": [.50, .20, .75],
    "parry_axis_local": [0., 0., 1.],
    "samples_per_motion": 81,
    "ik_joint_margin_rad": .05,
    "solver_seed": 20261122, "solver_max_iterations": 150,
    "guard_hold_s": .30,
    "attack_prepare_mode": "simultaneous",
    "attack_prepare_s": 3., "attack_thrust_s": 4., "attack_hold_s": .4,
    "attack_retract_s": 4., "attack_unprepare_s": 3.,
    "attack_prepare_upright_s": 3., "attack_prepare_tilt_s": 3.,
    "attack_unprepare_upright_s": 3., "attack_unprepare_tilt_s": 3.,
    "parry_sweep_s": 2., "parry_hold_s": .8, "parry_recover_s": 2.,
    "cooldown_s": .5,
    "control_period_s": .01,
    "provisional_command_speed_rad_s": 1.,
    "timing_note": "Conservative authored timing, not identified hardware capability. Own fixed holds/cooldown never wait for an opponent deadline.",
    "joint_limit_status": "actual scene remains hypothetical +/-pi; IK uses an inward margin",
}


def quintic(phase):
    phase = np.asarray(phase, dtype=float)
    if not np.isfinite(phase).all() or np.any((phase < 0) | (phase > 1)):
        raise ValueError("Finite phase in [0,1] required")
    return phase**3*(10+phase*(-15+6*phase))


def task_segment(start_grip, end_grip, start_axis, end_axis, samples=81):
    """Quintic translation plus shortest-arc spherical direction interpolation."""
    p0, p1, a0, a1 = [np.asarray(v, dtype=float) for v in (start_grip, end_grip, start_axis, end_axis)]
    if any(v.shape != (3,) or not np.isfinite(v).all() for v in (p0, p1, a0, a1)) or not all(np.isclose(np.linalg.norm(a), 1., atol=1e-7) for a in (a0, a1)):
        raise ValueError("Finite grip vectors and unit sword axes required")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 3:
        raise ValueError("At least three integer motion samples required")
    phase = np.linspace(0., 1., samples)
    blend = quintic(phase)
    cosine = float(np.clip(a0@a1, -1., 1.))
    if cosine < -.999999:
        raise ValueError("Antipodal axes need an explicit intermediate waypoint")
    angle = float(np.arccos(cosine))
    if angle < 1e-7:
        axis = (1-blend[:, None])*a0+blend[:, None]*a1
    else:
        axis = (np.sin((1-blend)*angle)[:, None]*a0+np.sin(blend*angle)[:, None]*a1)/np.sin(angle)
    axis /= np.linalg.norm(axis, axis=1)[:, None]
    return phase, p0+blend[:, None]*(p1-p0), axis


@dataclass
class SkillPath:
    name: str
    action_id: int
    qpos: np.ndarray
    time_s: np.ndarray
    segment_labels: list
    segments: list

    @property
    def duration_s(self):
        return float(self.time_s[-1])

    def sample(self, elapsed):
        if not np.isfinite(elapsed):
            raise ValueError("Own elapsed time must be finite")
        return np.array([np.interp(elapsed, self.time_s, self.qpos[:, i]) for i in range(self.qpos.shape[1])])

    def phase_at(self, elapsed):
        if not np.isfinite(elapsed):
            raise ValueError("Own elapsed time must be finite")
        if elapsed >= self.duration_s:
            return "ready"
        for item in self.segments:
            if elapsed < item["end_s"]:
                return item["name"]
        return "ready"


def compose_skill(name, action_id, pieces, control_period_s=.01, command_speed_rad_s=1.):
    """Join explicit finite pieces, retaining every row with exact joint seams."""
    if not np.isfinite([control_period_s, command_speed_rad_s]).all() or min(control_period_s, command_speed_rad_s) <= 0:
        raise ValueError("Positive finite control period and provisional command speed required")
    all_q, all_t, labels, segments = [], [], [], []
    current_time, last_q, row = 0., None, 0
    for label, q, requested in pieces:
        q = np.asarray(q, dtype=float)
        if q.ndim != 2 or len(q) < 2 or not np.isfinite(q).all() or not np.isfinite(requested) or requested <= 0:
            raise ValueError("Finite multirow joint pieces and positive own durations required")
        if last_q is not None and not np.array_equal(last_q, q[0]):
            raise ValueError("Every skill piece must have an exact common joint seam")
        per_row = max(float(requested)/(len(q)-1), 1.01*float(np.abs(np.diff(q, axis=0)).max())/command_speed_rad_s)
        interval = max(1., np.ceil(per_row/control_period_s))*control_period_s
        local_t = current_time+np.arange(len(q))*interval
        start_row = row
        if last_q is None:
            all_q.extend(q.copy()); all_t.extend(local_t)
            row = len(q)-1
        else:
            all_q.extend(q[1:].copy()); all_t.extend(local_t[1:])
            row += len(q)-1
        labels.extend([label]*(len(q)-1))
        segments.append({"name": label, "start_s": current_time, "end_s": float(local_t[-1]),
                         "start_row": start_row, "end_row": row, "requested_duration_s": float(requested)})
        current_time, last_q = float(local_t[-1]), q[-1].copy()
    if not all_q:
        raise ValueError("At least one finite skill piece required")
    return SkillPath(name, action_id, np.asarray(all_q), np.asarray(all_t), labels, segments)


def _screen(assembly, qpos, margin):
    """All raw sampled proxy contacts, including interpolated joint intervals."""
    model, data = assembly.model, mujoco.MjData(assembly.model)
    contacts, torque, clearance = [], [], []
    pair = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("arm_proxy_5", "adapter_plate_collision")]
    sampled = []
    for row, q in enumerate(qpos):
        if row == 0:
            sampled.append((row, 1., q))
        else:
            divisions = max(1, int(np.ceil(np.abs(q-qpos[row-1]).max()/.025)))
            sampled.extend((row, float(a), qpos[row-1]+a*(q-qpos[row-1])) for a in np.linspace(0., 1., divisions+1)[1:])
    for row, alpha, q in sampled:
        data.qpos[:], data.qvel[:] = q, 0.
        mujoco.mj_forward(model, data)
        torque.append(data.qfrc_bias.copy())
        clearance.append(float(mujoco.mj_geomDistance(model, data, pair[0], pair[1], .1, None)))
        for c in data.contact:
            contacts.append({"row": row, "interval_fraction": alpha,
                "geom1": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(c.geom1)),
                "geom2": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(c.geom2)), "distance_m": float(c.dist)})
    qs = np.asarray([q for _, _, q in sampled])
    true_margin = np.minimum(qs-model.jnt_range[:, 0], model.jnt_range[:, 1]-qs)
    caps = np.minimum(-model.actuator_forcerange[:, 0], model.actuator_forcerange[:, 1])
    torque = np.asarray(torque)
    exceeded = np.flatnonzero(np.any(np.abs(torque) > caps+1e-9, axis=1))
    return {"source_rows": len(qpos), "screened_configurations": len(qs), "all_raw_contacts": contacts,
            "strict_static_contacts_free": not contacts, "proxy5_adapter_minimum_clearance_m": min(clearance),
            "clearance_note": "Existing proxy5/adapter primitives are nearly tangent at their joint pivot; positive clearance is not a full-CAD structural clearance certificate.",
            "minimum_true_joint_margin_rad": float(true_margin.min()), "required_ik_margin_rad": margin,
            "joint_margin_pass": bool(true_margin.min() >= margin-1e-8),
            "static_gravity_peak_nm": np.abs(torque).max(0).tolist(), "original_debug_torque_caps_nm": caps.tolist(),
            "static_gravity_exceeded_screen_indices": exceeded.tolist(),
            "static_gravity_within_debug_caps": len(exceeded) == 0,
            "motor_diagnostic_note": "Existing debug motor caps are not identified hardware specifications. Gravity exceedance is reported separately and is not a geometry-fit rejection at this stage.",
            "candidate_passes_geometry_screen": bool(not contacts and true_margin.min() >= margin-1e-8),
            "candidate_passes_static_screen": bool(not contacts and true_margin.min() >= margin-1e-8),
            "scope": "sampled actual-model proxy geometry/gravity, not physical execution or skill effectiveness"}


@dataclass
class SharedLibrary:
    config: dict
    single_assembly: object
    guard_q: np.ndarray
    skills: dict
    reports: dict


def build_shared_library(config=None):
    """Build one finite action space, reusable identically by both robots.

    A task-space candidate failing IK is still returned in full and labelled;
    no source row is dropped and no opposite-role joint reset is inserted.
    Reverse paths below are explicitly authored recovery of manual motions,
    not reverse playback represented as original motion-capture data.
    """
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if config is not None:
        cfg.update(copy.deepcopy(config))
    mode = cfg["attack_prepare_mode"]
    if mode not in {"simultaneous", "upright_then_tilt"}:
        raise ValueError("attack_prepare_mode must be simultaneous or upright_then_tilt")
    margin = cfg["ik_joint_margin_rad"]
    if isinstance(margin, (bool, str)) or not isinstance(margin, (int, float)) or not np.isfinite(margin) or not 0 <= margin < np.pi:
        raise ValueError("Finite numerical IK margin in [0,pi) required")
    assembly = build_floor_assembly(joint_ranges=np.tile([-np.pi, np.pi], (7, 1)))
    bounds = assembly.model.jnt_range+np.array([margin, -margin])
    guard = np.asarray(cfg["guard_q_rad"], dtype=float)
    if guard.shape != (7,) or not np.isfinite(guard).all() or np.any(guard < bounds[:, 0]) or np.any(guard > bounds[:, 1]):
        raise ValueError("One common finite Guard inside the inset joint ranges is required")
    grip_g, axis_g, _ = tool_fk(assembly.model, qpos=guard)
    n = cfg["samples_per_motion"]
    targets = {
        "thrust": task_segment(cfg["attack_ready_grip_local_m"], cfg["attack_extended_grip_local_m"], cfg["attack_axis_local"], cfg["attack_axis_local"], n),
        "sweep": task_segment(grip_g, cfg["parry_extended_grip_local_m"], axis_g, cfg["parry_axis_local"], n),
    }
    if mode == "simultaneous":
        prep_names = ["prepare"]
        targets["prepare"] = task_segment(grip_g, cfg["attack_ready_grip_local_m"], axis_g, cfg["attack_axis_local"], n)
    else:
        prep_names = ["prepare_upright", "prepare_tilt"]
        targets["prepare_upright"] = task_segment(grip_g, cfg["attack_ready_grip_local_m"], axis_g, [0., 0., 1.], n)
        targets["prepare_tilt"] = task_segment(cfg["attack_ready_grip_local_m"], cfg["attack_ready_grip_local_m"], [0., 0., 1.], cfg["attack_axis_local"], n)
    solved, reports = {}, {}
    previous = None
    for label in prep_names+["thrust", "sweep"]:
        start = guard if label in {prep_names[0], "sweep"} else solved[previous][-1]
        phase, grip, axis = targets[label]
        result = match_sequence(assembly.model, grip, axis, bounds, initial_q=start,
                                seed=cfg["solver_seed"], max_iterations=cfg["solver_max_iterations"],
                                position_tolerance_m=1e-4, angle_tolerance_rad=np.deg2rad(.05),
                                tip_tolerance_m=.0005, collision_threshold_m=1e-8,
                                interpolation_step_rad=.025, first_frame_starts=1)
        solved[label] = result.qpos
        report = result.summary()
        report.update({"all_target_rows_retained": True, "target_grip_local_m": grip.tolist(),
                       "target_axis_local": axis.tolist(), "position_error_m": result.position_error_m.tolist(),
                       "axis_error_rad": result.angle_error_rad.tolist(), "tip_error_m": result.tip_error_m.tolist()})
        reports[label] = report
        previous = label
    prep, thrust, sweep = solved[prep_names[0]], solved["thrust"], solved["sweep"]
    # The first FK target of preparation and sweep is precisely the common
    # Guard, so these are equality checks, not post-fit teleports or edits.
    if not np.array_equal(prep[0], guard) or not np.array_equal(sweep[0], guard):
        raise RuntimeError("IK initialization failed to preserve the common Guard")
    attack_forward_names = prep_names+["thrust"]
    for before, after in zip(attack_forward_names[:-1], attack_forward_names[1:]):
        if not np.array_equal(solved[before][-1], solved[after][0]):
            raise RuntimeError("Continuous attack preparation/thrust seam was not preserved")
    hold = lambda q: np.vstack([q, q])
    prep_pieces = [(name, solved[name], cfg["attack_"+name+"_s"]) for name in prep_names]
    recovery_pieces = [(name.replace("prepare", "unprepare"), solved[name][::-1],
                        cfg["attack_"+name.replace("prepare", "unprepare")+"_s"])
                       for name in reversed(prep_names)]
    pieces = {
        "guard": [("guard", hold(guard), cfg["guard_hold_s"])],
        "attack": prep_pieces+[("thrust", thrust, cfg["attack_thrust_s"]),
                   ("extended_hold", hold(thrust[-1]), cfg["attack_hold_s"]),
                   ("retract", thrust[::-1], cfg["attack_retract_s"])]+recovery_pieces+[("cooldown", hold(guard), cfg["cooldown_s"])],
        "parry": [("sweep", sweep, cfg["parry_sweep_s"]), ("parry_hold", hold(sweep[-1]), cfg["parry_hold_s"]),
                  ("recover", sweep[::-1], cfg["parry_recover_s"]), ("cooldown", hold(guard), cfg["cooldown_s"])],
    }
    skills = {name: compose_skill(name, action_id, pieces[name], cfg["control_period_s"], cfg["provisional_command_speed_rad_s"])
              for action_id, name in enumerate(("guard", "attack", "parry"))}
    for name, path in skills.items():
        report = _screen(assembly, path.qpos, margin)
        report.update({"action_id": path.action_id, "duration_s": path.duration_s, "segments": copy.deepcopy(path.segments),
                       "exact_guard_start": bool(np.array_equal(path.qpos[0], guard)),
                       "exact_guard_end": bool(np.array_equal(path.qpos[-1], guard)),
                       "provisional_peak_command_speed_rad_s": float(np.abs(np.diff(path.qpos, axis=0)/np.diff(path.time_s)[:, None]).max()),
                       "physical_validation": "NOT RUN; independent policy/timing opportunities and degeneracy checks remain required"})
        fit_keys = tuple(prep_names)+("thrust",) if name == "attack" else (("sweep",) if name == "parry" else ())
        report["candidate_passes_skill_geometry"] = bool(report["candidate_passes_static_screen"] and all(reports[k]["all_frames_passed"] for k in fit_keys) and report["exact_guard_start"] and report["exact_guard_end"])
        reports[name] = report
    reports["library"] = {
        "one_shared_guard": True, "identical_local_action_space_for_both_agents": True,
        "all_skills_pass_geometry": all(reports[name]["candidate_passes_skill_geometry"] for name in skills),
        "opponent_future_inputs": False, "source_data": "none; hand-authored geometric candidates",
        "initial_placement_path_validated": False, "source_hashes": assembly.source_hashes,
        "next_gate": "actual finite transitions and independent-policy dual physics; meaningful attack, timely parry opportunities and anti-degeneracy controls before RL",
    }
    return SharedLibrary(cfg, assembly, guard.copy(), skills, reports)
