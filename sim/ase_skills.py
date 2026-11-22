"""Full, verified ASE Stab/ParryUp references retargeted to actual tool sites.

The native clips already contain recovery. Play each full clip ONCE, then its
separate authored guard transition; never treat it as a forward-only thrust and
reverse it as if that reverse were original data. No physical success is claimed.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
from pathlib import Path

import mujoco
import numpy as np

from plan1_skills.ase_numeric import read_verified_motion
from plan1_skills.library import ROOT, DEFAULT_CACHE, SkillClip, body_frame_motion, load_manifest
from plan1_skills.floor_test import dense_reference
from plan1_skills.matching import match_sequence
from .floor import build_floor_assembly


_R30 = [[.8660254037844386, -.5, 0.], [.5, .8660254037844386, 0.], [0., 0., 1.]]
_SEED = [1.069, .336, -.636, 1.523, -1.045, -1.201, .115]
DEFAULT_CONFIG = {
    "schema_version": 1, "base_distance_m": 1.75,
    "target_world_m": [1.43, 0., 1.05], "target_radius_m": .06,
    "path_convention": "native_full_clip_once_then_separate_authored_guard_transition",
    "mapping_status": "original single-arm shared mapping baseline; not optimized for duel geometry",
    "attack_neutral_grip_local_m": [.05, -.45, 1.05],
    "defend_neutral_grip_local_m": [.05, -.45, 1.05],
    "attack_rotation_matrix": _R30, "defend_rotation_matrix": _R30,
    "grip_displacement_scale": .5,
    "attack_seed_rad": _SEED, "defend_seed_rad": _SEED,
    "substeps_per_source_interval": 4, "transition_samples": 61,
    "solver_seed": 20261122, "solver_max_iterations": 120,
    "first_frame_starts": 1,
    "ik_joint_margin_rad": .05,
    "joint_limit_status": "hypothetical +/-pi radians, not measured hardware bounds",
}


@dataclass
class AseReferences:
    clips: dict
    source_time_s: dict
    neutral_axis: np.ndarray
    provenance: dict


def load_ase_references(source_dir=None):
    """Restricted numeric conversion of hash/size-verified cached data only."""
    manifest = load_manifest()
    if source_dir is None:
        candidates = [DEFAULT_CACHE/"sources", ROOT/"outputs/plan1_skills_validation/downloaded_sources"]
        source_dir = next((p for p in candidates if all((p/s["filename"]).is_file() for s in manifest["sources"].values())), candidates[0])
    source_dir = Path(source_dir).resolve()
    motions, paths = {}, {}
    for name, source in manifest["sources"].items():
        path = source_dir/source["filename"]
        motion = read_verified_motion(path, source["sha256"], source["size_bytes"])
        if motion["frames"] != source["frames"] or not np.isclose(motion["fps"], source["fps"], rtol=1e-9):
            raise ValueError(f"{name}: source frame/fps metadata mismatch")
        motions[name], paths[name] = motion, str(path)
    body = {name: body_frame_motion(motion) for name, motion in motions.items()}
    neutral = body["stab"][0][0]
    clips = {name: SkillClip(np.linspace(0., 1., len(grip)), grip-neutral, axes)
             for name, (grip, axes) in body.items()}
    provenance = {
        "upstream": copy.deepcopy(manifest["upstream"]),
        "sources": copy.deepcopy(manifest["sources"]), "source_paths": paths,
        "manifest_sha256": hashlib.sha256((ROOT/"motion_skills/plan1/manifest.json").read_bytes()).hexdigest(),
        "verification": "source size+SHA256 then restricted numeric reader; no unrestricted numpy pickle",
        "processing": "remove root translation; each clip expressed in its first horizontal shoulder frame; shared grip origin from Stab frame zero; no cropped frames",
        "displacement_units": "unscaled metres; builder applies explicit grip_displacement_scale exactly once",
    }
    return AseReferences(clips, {name: m["time"].copy() for name, m in motions.items()},
                         body["stab"][1][0].copy(), provenance)


def map_reference(clip, neutral, rotation, scale, substeps=4):
    """One rigid role transform and one uniform grip displacement scale."""
    rotation, neutral = np.asarray(rotation, dtype=float), np.asarray(neutral, dtype=float)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all() or not np.allclose(rotation.T@rotation, np.eye(3), atol=1e-8) or not np.isclose(np.linalg.det(rotation), 1., atol=1e-8):
        raise ValueError("rotation_matrix must be a finite proper orthogonal 3x3 matrix")
    if neutral.shape != (3,) or not np.isfinite(neutral).all() or not np.isfinite(scale) or scale <= 0:
        raise ValueError("Finite neutral 3-vector and positive displacement scale required")
    phase, delta, axis = dense_reference(clip, substeps)
    return phase, neutral+float(scale)*delta@rotation.T, axis@rotation.T


def guard_transition(start, end, samples=61):
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 3:
        raise ValueError("At least three integer transition samples required")
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    if start.shape != end.shape or start.ndim != 1 or not np.isfinite(np.r_[start, end]).all():
        raise ValueError("Finite equal-sized joint endpoints required")
    phase = np.linspace(0., 1., samples)
    blend = phase**3*(10+phase*(-15+6*phase))
    return {"phase": phase, "qpos": start+blend[:, None]*(end-start)}


def _validate_ik_margin(margin):
    if isinstance(margin, (bool, str)) or not isinstance(margin, (int, float, np.integer, np.floating)) or not np.isfinite(margin) or not 0 <= margin < np.pi:
        raise ValueError("ik_joint_margin_rad must be a finite numeric value in [0,pi), not bool")
    return float(margin)


def inset_joint_ranges(joint_ranges, margin):
    """Inset solver-only bounds; never modify the model's actual joint limits."""
    margin = _validate_ik_margin(margin)
    ranges = np.asarray(joint_ranges, dtype=float)
    if ranges.ndim != 2 or ranges.shape[1] != 2 or not np.isfinite(ranges).all() or np.any(ranges[:, 0] >= ranges[:, 1]):
        raise ValueError("Finite ordered Nx2 joint ranges required")
    result = ranges+np.array([margin, -margin])
    if np.any(result[:, 0] >= result[:, 1]):
        raise ValueError("IK margin leaves an empty joint interval")
    return result


def _screen(assembly, qs, ik_joint_margin_rad=0.):
    model, data = assembly.model, mujoco.MjData(assembly.model)
    required_margin = _validate_ik_margin(ik_joint_margin_rad)
    qs = np.asarray(qs, dtype=float)
    true_margin = np.minimum(qs-model.jnt_range[:, 0], model.jnt_range[:, 1]-qs)
    contacts, torque = [], []
    for index, q in enumerate(qs):
        data.qpos[:] = q
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        torque.append(data.qfrc_bias.copy())
        for c in data.contact:
            contacts.append({"frame": index,
                "geom1": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(c.geom1)),
                "geom2": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(c.geom2)), "distance_m": float(c.dist)})
    torque = np.asarray(torque)
    caps = np.minimum(-model.actuator_forcerange[:, 0], model.actuator_forcerange[:, 1])
    exceeded = np.flatnonzero(np.any(np.abs(torque) > caps+1e-9, axis=1)).tolist()
    return {"samples": len(qs), "all_raw_static_contacts": contacts,
            "strict_static_contacts_free": not contacts,
            "static_gravity_peak_nm": np.abs(torque).max(0).tolist(),
            "unchanged_debug_torque_caps_nm": caps.tolist(), "static_gravity_exceeded_indices": exceeded,
            "ik_joint_margin_rad": required_margin,
            "minimum_true_joint_margin_rad": float(true_margin.min()),
            "minimum_true_joint_margin_by_joint_rad": true_margin.min(0).tolist(),
            "joint_margin_failed_indices": np.flatnonzero(np.any(true_margin < required_margin-1e-8, axis=1)).tolist(),
            "scope": "static sampled single-arm proxy geometry/gravity only; not coupled physical playback"}


# Public name for experiment/search code; retain the internal helper spelling
# for callers already inspecting an early diagnostic version of this module.
screen_configurations = _screen


@dataclass
class AseLibrary:
    config: dict
    single_assembly: object
    guard_attack_q: np.ndarray
    attack_q: np.ndarray
    attack_phase: np.ndarray
    guard_defend_q: np.ndarray
    parry_q: np.ndarray
    parry_phase: np.ndarray
    reports: dict
    transitions: dict
    attack_source_time_s: np.ndarray
    parry_source_time_s: np.ndarray
    attack_grip_local_m: np.ndarray
    attack_axis_local: np.ndarray
    parry_grip_local_m: np.ndarray
    parry_axis_local: np.ndarray
    provenance: dict


def build_ase_library(config=None, source_dir=None):
    """Return complete native clips and separate, explicitly authored transitions.

    All original frames are retained; unsuccessful fits/collisions are returned
    as evidence, not discarded. 5D IK leaves sword roll unconstrained. Role
    transforms are local to the corresponding fixed robot base, before the
    outer scene yaw/translation. No original CAD or cached trajectory is edited.
    """
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if config is not None:
        cfg.update(copy.deepcopy(config))
    if cfg["path_convention"] != DEFAULT_CONFIG["path_convention"]:
        raise ValueError("ASE clips must be treated as full native clips, not forward-only actions")
    margin = _validate_ik_margin(cfg["ik_joint_margin_rad"])
    cfg["ik_joint_margin_rad"] = margin
    refs = load_ase_references(source_dir)
    assembly = build_floor_assembly(joint_ranges=np.tile([-np.pi, np.pi], (7, 1)))
    ik_ranges = inset_joint_ranges(assembly.model.jnt_range, margin)
    roles, reports, transitions = {}, {}, {}
    for role, source, prefix in (("attack", "stab", "attack"), ("defend", "parry_up", "parry")):
        neutral, rotation = cfg[f"{role}_neutral_grip_local_m"], cfg[f"{role}_rotation_matrix"]
        phase, grip, axes = map_reference(refs.clips[source], neutral, rotation,
                                         cfg["grip_displacement_scale"], cfg["substeps_per_source_interval"])
        options = dict(seed=cfg["solver_seed"], max_iterations=cfg["solver_max_iterations"],
                       position_tolerance_m=1e-4, angle_tolerance_rad=np.deg2rad(.05),
                       tip_tolerance_m=.0005, collision_threshold_m=.001,
                       interpolation_step_rad=.025, first_frame_starts=cfg["first_frame_starts"])
        guard = match_sequence(assembly.model, [neutral], [np.asarray(rotation)@refs.neutral_axis],
                               ik_ranges, initial_q=cfg[f"{role}_seed_rad"], **options)
        result = match_sequence(assembly.model, grip, axes, ik_ranges,
                                initial_q=guard.qpos[0], **options)
        times = np.interp(phase, refs.clips[source].phase, refs.source_time_s[source])
        roles[role] = (guard.qpos[0].copy(), result.qpos, phase, times, grip, axes)
        report = result.summary()
        report.update(_screen(assembly, result.qpos, margin))
        report.update({"original_source_frames": len(refs.clips[source].phase),
                       "all_original_frames_retained": True, "source_phase_indices": list(range(0, len(phase), cfg["substeps_per_source_interval"])),
                       "pose_error_m": result.position_error_m.tolist(), "axis_error_rad": result.angle_error_rad.tolist(),
                       "tip_error_m": result.tip_error_m.tolist(), "interpolated_path_contacts": result.contacts,
                       "guard_solver": guard.summary(), "guard_static_screen": _screen(assembly, guard.qpos, margin),
                       "solver_joint_ranges_rad": ik_ranges.tolist(), "unchanged_scene_joint_ranges_rad": assembly.model.jnt_range.tolist(),
                       "source_native_duration_s": float(times[-1]-times[0]), "timing_status": "source time is provenance, NOT approved execution timing"})
        reports[source] = report
        transitions[prefix+"_in"] = guard_transition(guard.qpos[0], result.qpos[0], cfg["transition_samples"])
        transitions[prefix+"_out"] = guard_transition(result.qpos[-1], guard.qpos[0], cfg["transition_samples"])
        for key in (prefix+"_in", prefix+"_out"):
            reports[key] = _screen(assembly, transitions[key]["qpos"], margin)
        guard_screen = report["guard_static_screen"]
        transition_pass = all(reports[key]["strict_static_contacts_free"] and not reports[key]["static_gravity_exceeded_indices"] and not reports[key]["joint_margin_failed_indices"] for key in (prefix+"_in", prefix+"_out"))
        report["candidate_passes_single_arm_screen"] = bool(result.success.all() and report["strict_static_contacts_free"] and not report["static_gravity_exceeded_indices"] and not report["joint_margin_failed_indices"] and guard.success.all() and guard_screen["strict_static_contacts_free"] and not guard_screen["static_gravity_exceeded_indices"] and not guard_screen["joint_margin_failed_indices"] and transition_pass)
    a, d = roles["attack"], roles["defend"]
    provenance = copy.deepcopy(refs.provenance)
    provenance.update({"ik_status": "re-solved entire verified references using actual shinai_grip/shinai_tip sites; not loaded frozen qpos",
                       "roles": {"attack": "ASE RL_Avatar_Atk_Stab_Motion.npy", "defend": "ASE RL_Avatar_Sword_ParryUp_Motion.npy"},
                       "source_files_modified": False, "physical_execution": "NOT RUN by this builder"})
    return AseLibrary(cfg, assembly, a[0], a[1], a[2], d[0], d[1], d[2], reports, transitions,
                      a[3], d[3], a[4], a[5], d[4], d[5], provenance)
