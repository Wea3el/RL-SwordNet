"""Separate A1 pose library: longsword guards and cuts between them.

Ten stationary poses (zero-velocity holds) and the cuts between them are
authored as grip position plus blade direction in the robot base frame and
solved on the actual seven-joint arm. Left/right are the robot's own +y/-y
sides when facing +x; "head" and "torso" heights are virtual reference
heights, since the fixed floor arm has neither. Every candidate is retained
and screened; passing the sampled geometry screen is not physical hit, parry
or timing evidence.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from plan1_skills.matching import match_sequence, tool_fk
from .assembly import ROOT, _finite_json
from .floor import build_floor_assembly, configure_floor_camera
from .shared_skills import DEFAULT_CONFIG as SHARED_CONFIG, SkillPath, _screen, compose_skill, quintic


def _unit(v):
    v = np.asarray(v, dtype=float)
    return (v/np.linalg.norm(v)).tolist()


def _pitched(degrees_above_horizontal):
    angle = np.deg2rad(degrees_above_horizontal)
    return [float(np.cos(angle)), 0., float(np.sin(angle))]


_SIDE = .25      # metres from the base centre line to each side ("left/right of torso")
_FORWARD = .30   # grip x for torso-height poses

POSE_ORDER = ("roof_left", "roof_right", "fool_left", "fool_right", "plow_left", "plow_right",
              "ox_left", "ox_right", "long_point", "kron")
CUT_ORDER = ("downward_cut_left_to_right", "downward_cut_right_to_left",
             "upward_cut_right_to_left", "upward_cut_left_to_right", "straight_down")

DEFAULT_CONFIG = {
    "schema_version": 1,
    "library_id": "hema_pose_library_v1",
    "status": "geometric_candidates_not_physically_validated",
    "policy_family": "A1",
    "action_space_kind": "finite_pose_and_cut_selection",
    "provenance": "hand-authored from the user's pose list of 2026-11-22; no motion data, no opponent input",
    "frame": "robot base frame: +x toward the opponent, +y robot left, -y robot right, +z up",
    "body_reference": "virtual heights only; the floor-fixed arm has no head or torso",
    "poses": {
        "roof_left": {"grip_local_m": [_FORWARD-.05, +_SIDE, 1.45], "axis_local": _unit([-.2, 0., 1.]),
                      "spec": "position high, pointed up, left of head, velocity 0",
                      "assumption": "blade tilted 11 deg back from vertical so the grip clears the arm"},
        "roof_right": {"grip_local_m": [_FORWARD-.05, -_SIDE, 1.45], "axis_local": _unit([-.2, 0., 1.]),
                       "spec": "position high, pointed up, right of head, velocity 0",
                       "assumption": "same back tilt as roof_left"},
        "fool_left": {"grip_local_m": [_FORWARD, +_SIDE, .65], "axis_local": _pitched(-40.),
                      "spec": "position down, point down, left of torso, velocity 0",
                      "assumption": "point 40 deg below horizontal; straight down would put the tip 0.14 m under the floor"},
        "fool_right": {"grip_local_m": [_FORWARD, -_SIDE, .65], "axis_local": _pitched(-40.),
                       "spec": "position down, point down, right of torso, velocity 0",
                       "assumption": "same floor-clearing pitch as fool_left"},
        "plow_left": {"grip_local_m": [_FORWARD, +_SIDE, .80], "axis_local": _pitched(20.),
                      "spec": "position down, point forward, left of torso, velocity 0",
                      "assumption": "point raised 20 deg toward the opponent's upper target"},
        "plow_right": {"grip_local_m": [_FORWARD, -_SIDE, .80], "axis_local": _pitched(20.),
                       "spec": "position down, point forward, RIGHT of torso (the source list says left; read as a copy error), velocity 0",
                       "assumption": "mirror of plow_left"},
        "ox_left": {"grip_local_m": [_FORWARD-.10, +_SIDE, 1.35], "axis_local": _pitched(-15.),
                    "spec": "position up, point forward, left of torso, velocity 0",
                    "assumption": "point 15 deg below horizontal from head height"},
        "ox_right": {"grip_local_m": [_FORWARD-.10, -_SIDE, 1.35], "axis_local": _pitched(-15.),
                     "spec": "position up, point forward, right of torso, velocity 0",
                     "assumption": "mirror of ox_left"},
        "long_point": {"grip_local_m": [.55, 0., 1.00], "axis_local": [1., 0., 0.],
                       "spec": "position centre, point forward, velocity 0; blade roll irrelevant for a shinai",
                       "assumption": "grip extended 0.55 m forward at 1.0 m height"},
        "kron": {"grip_local_m": [_FORWARD-.05, 0., 1.55], "axis_local": _unit([-.5, 0., .866]),
                 "spec": "position centre over the head (orientation not given)",
                 "assumption": "point up and 30 deg back; the source list left the blade orientation blank"},
    },
    "cuts": {
        "downward_cut_left_to_right": {"start": "roof_left", "alternative_starts": ["ox_left"], "via": ["long_point"], "end": "fool_right",
                                       "spec": "velocity > 1, any up position to down, left to right, pass through long point"},
        "downward_cut_right_to_left": {"start": "roof_right", "alternative_starts": ["ox_right"], "via": ["long_point"], "end": "fool_left",
                                       "spec": "mirror of the left-to-right downward cut"},
        "upward_cut_right_to_left": {"start": "fool_right", "alternative_starts": ["plow_right"], "via": ["long_point"], "end": "roof_left",
                                     "spec": "velocity > 1, any down position to up, right to left",
                                     "assumption": "routed through long point like the downward cuts; the source list did not say"},
        "upward_cut_left_to_right": {"start": "fool_left", "alternative_starts": ["plow_left"], "via": ["long_point"], "end": "roof_right",
                                     "spec": "mirror of the right-to-left upward cut",
                                     "assumption": "routed through long point"},
        "straight_down": {"start": "kron", "alternative_starts": [], "via": [], "end": "long_point",
                          "spec": "starts from kron, ends in long point"},
    },
    "cut_path_space": "joint",
    "cut_path_note": "joint: piecewise-linear joint path through the exact waypoint poses with one continuous quintic timing (no stop at long point). task: straight grip line and shortest-arc blade direction, solved by warm-started IK, then settled onto the exact end pose.",
    "minimum_cut_tip_speed_m_s": 1.0,
    "cut_segment_s": .6,
    "settle_s": .3,
    "hold_s": .3,
    "transition_s": 1.5,
    "samples_per_segment": 61,
    "control_period_s": .01,
    "provisional_command_speed_rad_s": 6.,
    "ik_joint_margin_rad": .05,
    "solver_seed": 20261122, "solver_max_iterations": 150, "first_frame_starts": 24,
    "seed_q_rad": list(SHARED_CONFIG["guard_q_rad"]),
    "timing_note": "Segment durations and the command-speed cap are authored assumptions, not identified hardware capability.",
    "joint_limit_status": "actual scene remains hypothetical +/-pi; IK uses an inward margin",
}


@dataclass
class HemaLibrary:
    config: dict
    single_assembly: object
    poses: dict          # name -> exact joint vector
    pose_targets: dict   # name -> grip/axis/tip actually reached
    holds: dict          # name -> SkillPath (zero-velocity hold)
    cuts: dict           # name -> SkillPath from its start pose to its end pose
    transitions: dict    # (from, to) -> SkillPath joint-space move between stationary poses
    reports: dict

    @property
    def action_ids(self):
        return {name: i for i, name in enumerate(list(self.holds)+list(self.cuts))}


def _validate(cfg):
    for key in ("cut_segment_s", "settle_s", "hold_s", "transition_s", "control_period_s",
                "provisional_command_speed_rad_s", "minimum_cut_tip_speed_m_s"):
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be a positive finite number")
    margin = cfg["ik_joint_margin_rad"]
    if isinstance(margin, (bool, str)) or not isinstance(margin, (int, float)) or not np.isfinite(margin) or not 0 <= margin < np.pi:
        raise ValueError("Finite numerical IK margin in [0,pi) required")
    n = cfg["samples_per_segment"]
    if isinstance(n, bool) or not isinstance(n, int) or n < 3:
        raise ValueError("samples_per_segment must be an integer of at least 3")
    if cfg["cut_path_space"] not in {"joint", "task"}:
        raise ValueError("cut_path_space must be joint or task")
    if not cfg["poses"]:
        raise ValueError("At least one stationary pose is required")
    for name, pose in cfg["poses"].items():
        grip, axis = np.asarray(pose["grip_local_m"], dtype=float), np.asarray(pose["axis_local"], dtype=float)
        if grip.shape != (3,) or axis.shape != (3,) or not np.isfinite(np.r_[grip, axis]).all() or not np.isclose(np.linalg.norm(axis), 1., atol=1e-6):
            raise ValueError(f"Pose {name} needs a finite grip 3-vector and a unit blade axis")
    for name, cut in cfg["cuts"].items():
        starts = [cut["start"], *cut.get("alternative_starts", [])]
        for start in starts:
            chain = [start, *cut["via"], cut["end"]]
            missing = [p for p in chain if p not in cfg["poses"]]
            if missing:
                raise ValueError(f"Cut {name} references unknown poses {missing}")
            if len(set(chain)) != len(chain):
                raise ValueError(f"Cut {name} from {start} must not revisit a pose")
    seed = np.asarray(cfg["seed_q_rad"], dtype=float)
    if seed.shape != (7,) or not np.isfinite(seed).all():
        raise ValueError("seed_q_rad must be seven finite joint values")
    return float(margin)


def _solver_options(cfg, starts):
    return dict(seed=cfg["solver_seed"], max_iterations=cfg["solver_max_iterations"],
                position_tolerance_m=1e-4, angle_tolerance_rad=np.deg2rad(.05),
                tip_tolerance_m=.0005, collision_threshold_m=1e-8,
                interpolation_step_rad=.025, first_frame_starts=starts)


def quintic_inverse(value):
    """Phase p in [0,1] with quintic(p) == value, by bisection."""
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Quintic inverse needs a value in [0,1]")
    low, high = 0., 1.
    for _ in range(80):
        mid = .5*(low+high)
        if float(quintic(mid)) < value:
            low = mid
        else:
            high = mid
    return .5*(low+high)


def _slerp(a0, a1, blend):
    cosine = float(np.clip(a0@a1, -1., 1.))
    if cosine < -.999999:
        raise ValueError("Antipodal blade axes need an intermediate waypoint")
    angle = float(np.arccos(cosine))
    if angle < 1e-7:
        axis = (1-blend[:, None])*a0+blend[:, None]*a1
    else:
        axis = (np.sin((1-blend)*angle)[:, None]*a0+np.sin(blend*angle)[:, None]*a1)/np.sin(angle)
    return axis/np.linalg.norm(axis, axis=1)[:, None]


def waypoint_path(lengths, samples):
    """One continuous quintic timing over consecutive segments of given lengths.

    Returns, per segment, the global phase rows and the local blend alpha in
    [0,1]. Every segment gets ``samples`` rows; consecutive segments share the
    waypoint row exactly. Velocity is continuous through interior waypoints.
    """
    lengths = np.asarray(lengths, dtype=float)
    if lengths.ndim != 1 or len(lengths) == 0 or not np.isfinite(lengths).all() or np.any(lengths <= 0):
        raise ValueError("Positive finite segment lengths required")
    fractions = np.r_[0., np.cumsum(lengths)]/lengths.sum()
    phases = [quintic_inverse(f) for f in fractions]
    phases[0], phases[-1] = 0., 1.
    pieces = []
    for i in range(len(lengths)):
        p = np.linspace(phases[i], phases[i+1], samples)
        alpha = (quintic(p)-fractions[i])/(fractions[i+1]-fractions[i])
        alpha = np.clip(alpha, 0., 1.)
        alpha[0], alpha[-1] = 0., 1.
        pieces.append({"phase": p, "alpha": alpha, "phase_span": float(phases[i+1]-phases[i])})
    return pieces


def tip_speed_profile(model, path):
    """Tip and grip speeds implied by the authored timing; kinematic, not measured."""
    grips, tips = [], []
    data = mujoco.MjData(model)
    for q in path.qpos:
        grip, _, tip = tool_fk(model, data, q)
        grips.append(grip); tips.append(tip)
    grips, tips, dt = np.asarray(grips), np.asarray(tips), np.diff(path.time_s)
    return {"tip_speed_m_s": np.linalg.norm(np.diff(tips, axis=0), axis=1)/dt,
            "grip_speed_m_s": np.linalg.norm(np.diff(grips, axis=0), axis=1)/dt,
            "tip_world_m": tips, "grip_world_m": grips}


def _joint_move(start, end, samples):
    phase = np.linspace(0., 1., samples)
    blend = quintic(phase)
    rows = start+blend[:, None]*(end-start)
    rows[0], rows[-1] = start, end   # exact seams, not rounded endpoints
    return rows


def _finish(assembly, path, margin, extra):
    report = _screen(assembly, path.qpos, margin)
    report.update({"action_id": path.action_id, "duration_s": path.duration_s, "segments": copy.deepcopy(path.segments),
                   "provisional_peak_command_speed_rad_s": float(np.abs(np.diff(path.qpos, axis=0)/np.diff(path.time_s)[:, None]).max()) if len(path.qpos) > 1 else 0.,
                   "physical_validation": "NOT RUN; no hit, parry, timing or dual-arm evidence"})
    report.update(extra)
    return report


def _cut_pieces(cfg, model, bounds, poses, reached, chain):
    """Rows for one cut through ``chain``; exact at every waypoint pose."""
    n, total_s = cfg["samples_per_segment"], cfg["cut_segment_s"]*(len(chain)-1)
    pairs = list(zip(chain[:-1], chain[1:]))
    pieces, ik_reports = [], []
    if cfg["cut_path_space"] == "joint":
        lengths = [np.abs(poses[b]-poses[a]).max() for a, b in pairs]
        for (a, b), piece in zip(pairs, waypoint_path(lengths, n)):
            rows = poses[a]+piece["alpha"][:, None]*(poses[b]-poses[a])
            rows[0], rows[-1] = poses[a], poses[b]
            pieces.append((f"{a}_to_{b}", rows, total_s*piece["phase_span"]))
        return pieces, ik_reports, 0.
    grips, axes, labels, spans = [], [], [], []
    lengths = []
    for a, b in pairs:
        ga, gb = np.asarray(reached[a]["grip_local_m"]), np.asarray(reached[b]["grip_local_m"])
        aa, ab = np.asarray(reached[a]["axis_local"]), np.asarray(reached[b]["axis_local"])
        lengths.append(float(np.linalg.norm(gb-ga))+.79*float(np.arccos(np.clip(aa@ab, -1., 1.))))
    for (a, b), piece in zip(pairs, waypoint_path(lengths, n)):
        ga, gb = np.asarray(reached[a]["grip_local_m"]), np.asarray(reached[b]["grip_local_m"])
        aa, ab = np.asarray(reached[a]["axis_local"]), np.asarray(reached[b]["axis_local"])
        alpha = piece["alpha"]
        grip, axis = ga+alpha[:, None]*(gb-ga), _slerp(aa, ab, alpha)
        grip[0], grip[-1], axis[0], axis[-1] = ga, gb, aa, ab
        start = 0 if not grips else 1   # share the seam row with the previous piece
        grips.extend(grip[start:]); axes.extend(axis[start:])
        labels.append((f"{a}_to_{b}", len(grip), piece["phase_span"]))
    result = match_sequence(model, np.asarray(grips), np.asarray(axes), bounds, initial_q=poses[chain[0]], **_solver_options(cfg, 1))
    if not np.array_equal(result.qpos[0], poses[chain[0]]):
        raise RuntimeError(f"IK did not preserve the exact {chain[0]} seam")
    row = 0
    for index, (label, count, span) in enumerate(labels):
        rows = result.qpos[row:row+count] if index == 0 else result.qpos[row-1:row-1+count]
        row += count if index == 0 else count-1
        pieces.append((label, rows.copy(), total_s*span))
    summary = result.summary()
    summary.update({"chain": chain, "warm_started_from": chain[0]})
    ik_reports.append(summary)
    end_q = poses[chain[-1]]
    settle_delta = float(np.abs(end_q-pieces[-1][1][-1]).max())
    if settle_delta > 0:
        pieces.append(("settle_"+chain[-1], _joint_move(pieces[-1][1][-1], end_q, n), cfg["settle_s"]))
    return pieces, ik_reports, settle_delta


def build_hema_library(config=None):
    """Solve every pose and cut on the actual arm; keep and label every result.

    Stationary poses are solved from one common seed with bounded multistart.
    Cuts pass exactly through their waypoint poses with one continuous timing
    profile, so every seam is an equality and no joint is reset mid-skill.
    Nothing here edits qpos during execution or validates physical contact.
    """
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if config is not None:
        cfg.update(copy.deepcopy(config))
    margin = _validate(cfg)
    assembly = build_floor_assembly(joint_ranges=np.tile([-np.pi, np.pi], (7, 1)))
    model = assembly.model
    bounds = model.jnt_range+np.array([margin, -margin])
    seed = np.asarray(cfg["seed_q_rad"], dtype=float)
    speed, period, n = cfg["provisional_command_speed_rad_s"], cfg["control_period_s"], cfg["samples_per_segment"]
    poses, reached, holds, reports = {}, {}, {}, {"poses": {}, "cuts": {}, "transitions": {}}
    hold = lambda q: np.vstack([q, q])
    for index, name in enumerate(cfg["poses"]):
        pose = cfg["poses"][name]
        result = match_sequence(model, [pose["grip_local_m"]], [pose["axis_local"]], bounds,
                                initial_q=seed, **_solver_options(cfg, cfg["first_frame_starts"]))
        q = result.qpos[0].copy()
        grip, axis, tip = tool_fk(model, qpos=q)
        poses[name], reached[name] = q, {"grip_local_m": grip.tolist(), "axis_local": axis.tolist(), "tip_local_m": tip.tolist()}
        holds[name] = compose_skill(name, index, [(name, hold(q), cfg["hold_s"])], period, speed)
        report = _finish(assembly, holds[name], margin, {
            "kind": "stationary_hold", "spec": pose["spec"], "assumption": pose.get("assumption", ""),
            "target_grip_local_m": list(pose["grip_local_m"]), "target_axis_local": list(pose["axis_local"]),
            "reached": reached[name], "solver": result.summary(),
            "position_error_m": float(result.position_error_m[0]), "axis_error_rad": float(result.angle_error_rad[0]),
            "tip_above_floor_m": float(tip[2]), "velocity_zero_hold": True})
        report["candidate_passes_pose_geometry"] = bool(result.success[0] and report["candidate_passes_static_screen"])
        reports["poses"][name] = report
    cuts = {}
    for name, cut in cfg["cuts"].items():
        for start in [cut["start"], *cut.get("alternative_starts", [])]:
            label = name if start == cut["start"] else f"{name}_from_{start}"
            chain = [start, *cut["via"], cut["end"]]
            pieces, ik_reports, settle_delta = _cut_pieces(cfg, model, bounds, poses, reached, chain)
            pieces.append(("hold_"+cut["end"], hold(poses[cut["end"]]), cfg["hold_s"]))
            path = compose_skill(label, len(poses)+len(cuts), pieces, period, speed)
            cuts[label] = path
            profile = tip_speed_profile(model, path)
            by_name = {s["name"]: s for s in path.segments}
            settle = by_name.get("settle_"+cut["end"])
            settle_tip_travel = float(np.linalg.norm(profile["tip_world_m"][settle["end_row"]]-profile["tip_world_m"][settle["start_row"]])) if settle else 0.
            passage = {}
            for via in cut["via"]:
                row = by_name[f"{chain[chain.index(via)-1]}_to_{via}"]["end_row"]
                passage[via] = {"row": row, "time_s": float(path.time_s[row]),
                                "tip_speed_m_s": float(np.mean(profile["tip_speed_m_s"][max(row-1, 0):row+1])),
                                "exact_pose_row": bool(np.array_equal(path.qpos[row], poses[via]))}
            motion_end = settle["start_row"] if settle else by_name["hold_"+cut["end"]]["start_row"]
            peak_tip = float(profile["tip_speed_m_s"][:motion_end].max())
            report = _finish(assembly, path, margin, {
                "kind": "dynamic_cut", "spec": cut["spec"], "assumption": cut.get("assumption", ""),
                "cut_path_space": cfg["cut_path_space"], "chain": chain, "task_ik": ik_reports,
                "exact_start_pose": bool(np.array_equal(path.qpos[0], poses[start])),
                "exact_end_pose": bool(np.array_equal(path.qpos[-1], poses[cut["end"]])),
                "settle_max_joint_delta_rad": settle_delta, "settle_tip_travel_m": settle_tip_travel,
                "long_point_passage": passage,
                "peak_tip_speed_m_s": peak_tip, "mean_tip_speed_m_s": float(profile["tip_speed_m_s"][:motion_end].mean()),
                "tip_path_local_m": profile["tip_world_m"].tolist(),
                "minimum_cut_tip_speed_m_s": cfg["minimum_cut_tip_speed_m_s"],
                "tip_speed_requirement_met": bool(peak_tip >= cfg["minimum_cut_tip_speed_m_s"] and all(p["tip_speed_m_s"] >= cfg["minimum_cut_tip_speed_m_s"] for p in passage.values())),
                "speed_note": "authored timing under a provisional command-speed cap; not a measured or torque-feasible execution speed"})
            report["candidate_passes_cut_geometry"] = bool(report["candidate_passes_static_screen"] and all(s["all_frames_passed"] for s in ik_reports)
                                                           and report["exact_start_pose"] and report["exact_end_pose"] and all(p["exact_pose_row"] for p in passage.values()))
            reports["cuts"][label] = report
    transitions = {}
    for a in poses:
        for b in poses:
            if a == b:
                continue
            path = compose_skill(f"{a}_to_{b}", -1, [(f"{a}_to_{b}", _joint_move(poses[a], poses[b], n), cfg["transition_s"])], period, speed)
            transitions[(a, b)] = path
            report = _finish(assembly, path, margin, {"kind": "joint_space_transition", "from": a, "to": b,
                                                      "max_joint_delta_rad": float(np.abs(poses[b]-poses[a]).max())})
            report["candidate_passes_transition_geometry"] = bool(report["candidate_passes_static_screen"])
            reports["transitions"][f"{a}->{b}"] = report
    reports["library"] = {
        "library_id": cfg["library_id"], "status": cfg["status"], "policy_family": cfg["policy_family"],
        "pose_count": len(poses), "cut_count": len(cuts), "transition_count": len(transitions),
        "all_poses_pass_geometry": all(r["candidate_passes_pose_geometry"] for r in reports["poses"].values()),
        "all_cuts_pass_geometry": all(r["candidate_passes_cut_geometry"] for r in reports["cuts"].values()),
        "all_cuts_meet_tip_speed": all(r["tip_speed_requirement_met"] for r in reports["cuts"].values()),
        "failed_cuts": [k for k, r in reports["cuts"].items() if not r["candidate_passes_cut_geometry"]],
        "failed_transitions": [k for k, r in reports["transitions"].items() if not r["candidate_passes_transition_geometry"]],
        "separate_from_shared_and_ase_libraries": True,
        "opponent_future_inputs": False, "source_data": "none; hand-authored pose list",
        "source_hashes": assembly.source_hashes,
        "next_gate": "dual-arm physical attack/parry validation of each cut against each guard before any training use",
    }
    return HemaLibrary(cfg, assembly, poses, reached, holds, cuts, transitions, reports)


def library_summary(library):
    reports = copy.deepcopy(library.reports)
    for group in ("poses", "cuts", "transitions"):
        for report in reports[group].values():
            report.pop("all_raw_contacts", None)
    return _finite_json({"config": library.config, "action_ids": library.action_ids,
                         "poses_q_rad": {k: v.tolist() for k, v in library.poses.items()},
                         "reports": reports})


def _view(library, names, dwell_s=1.5):
    import mujoco.viewer
    model, data = library.single_assembly.model, mujoco.MjData(library.single_assembly.model)
    items = []
    for name in names:
        if name in library.holds:
            items.append((name, library.holds[name]))
        elif name in library.cuts:
            items.append((name, library.cuts[name]))
        else:
            raise ValueError(f"Unknown pose or cut: {name}")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        configure_floor_camera(library.single_assembly, viewer.cam)
        viewer.opt.geomgroup[3] = 0
        print("KINEMATIC VIEW ONLY: joint targets are played without physics; close the viewer to exit.")
        while viewer.is_running():
            for name, path in items:
                print(f"  {name}")
                start = time.time()
                while viewer.is_running():
                    elapsed = time.time()-start
                    if elapsed > path.duration_s+dwell_s:
                        break
                    data.qpos[:] = path.sample(min(elapsed, path.duration_s))
                    mujoco.mj_kinematics(model, data)
                    viewer.sync()
                    time.sleep(.02)


def render_library(library, output, size=480, frames_per_cut=5):
    """Write PNG strips of every stationary pose and every cut; kinematic, no physics."""
    from PIL import Image, ImageDraw
    model = library.single_assembly.model
    model.vis.global_.offwidth = model.vis.global_.offheight = max(size, 1024)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, size, size)
    option = mujoco.MjvOption(); option.geomgroup[3] = 0
    camera = mujoco.MjvCamera(); camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [.3, 0., 1.1]; camera.distance, camera.azimuth, camera.elevation = 3.2, 130, -12

    def shot(q, label):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera, option)
        image = Image.fromarray(renderer.render())
        ImageDraw.Draw(image).text((10, 10), label, fill=(255, 255, 0))
        return image

    def strip(items, path):
        sheet = Image.new("RGB", (size*len(items), size))
        for i, (q, label) in enumerate(items):
            sheet.paste(shot(q, label), (i*size, 0))
        sheet.save(path)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    names = list(library.holds)
    written = []
    for row in range(0, len(names), 5):
        path = output/f"poses_row{row//5+1}.png"
        strip([(library.poses[n], n) for n in names[row:row+5]], path)
        written.append(path)
    for name, cut in library.cuts.items():
        end_hold = next(s for s in cut.segments if s["name"].startswith("hold_"))
        rows = np.linspace(0, end_hold["start_row"], frames_per_cut).round().astype(int)
        path = output/f"cut_{name}.png"
        strip([(cut.qpos[r], f"{name}  t={cut.time_s[r]:.2f}s") for r in rows], path)
        written.append(path)
    renderer.close()
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/hema_poses")
    parser.add_argument("--config", type=Path, default=None, help="JSON file whose top-level keys override DEFAULT_CONFIG")
    parser.add_argument("--path-space", choices=("joint", "task"), default=None, help="override cut_path_space")
    parser.add_argument("--view", nargs="*", metavar="NAME", help="play the named poses/cuts (default: all) in a kinematic viewer")
    parser.add_argument("--render", action="store_true", help="write PNG strips of every pose and cut into OUTPUT/renders")
    args = parser.parse_args()
    overrides = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    if args.path_space:
        overrides["cut_path_space"] = args.path_space
    library = build_hema_library(overrides or None)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = library_summary(library)
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    lib = library.reports["library"]
    print("GEOMETRY SCREEN ONLY: sampled IK and proxy collisions on the actual arm; no physical attack/parry evidence.")
    for name, report in library.reports["poses"].items():
        print(f"  pose {report['action_id']:2d} {name:12s} geometry={'pass' if report['candidate_passes_pose_geometry'] else 'FAIL'} tip_z={report['tip_above_floor_m']:.2f} m")
    for name, report in library.reports["cuts"].items():
        passage = report["long_point_passage"].get("long_point", {}).get("tip_speed_m_s")
        print(f"  cut  {report['action_id']:2d} {name:44s} geometry={'pass' if report['candidate_passes_cut_geometry'] else 'FAIL'} peak_tip={report['peak_tip_speed_m_s']:.2f} m/s"
              + (f" at_long_point={passage:.2f} m/s" if passage is not None else "") + f" speed_req={'met' if report['tip_speed_requirement_met'] else 'NOT met'} duration={report['duration_s']:.2f} s")
    print(f"  transitions failing geometry: {len(lib['failed_transitions'])} of {lib['transition_count']}")
    if args.render:
        written = render_library(library, args.output/"renders")
        print(f"  rendered {len(written)} PNG strips into {(args.output/'renders').resolve()}")
    print(f"Artifacts: {args.output.resolve()}")
    if args.view is not None:
        _view(library, args.view or list(library.holds)+list(library.cuts))


if __name__ == "__main__":
    main()
