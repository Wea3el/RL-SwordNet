"""A1 policy-family ASE-derived pilot, NEVER an admitted training skill bank.

A1 is a policy/action-space family, not the scene's fixed ``a1`` robot slot.
Two A1 policy instances may occupy both slots for A1 self-play. This finite
library does not restrict or implement A2's continuous joint-command space.

Reads only trusted-hash cached numeric source files. Explicit keyframe selection,
fixed-base approximation, scaling, smooth interpolation and new timing are
authored adaptations, not faithful playback of expert demonstrations. Optional
actual-tool IK retains failures; this module performs no physical attack/defence
validation, scoring, runtime tactical planning or opponent-dependent adaptation.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from plan1_skills.ase_numeric import UPSTREAM_COMMIT, read_verified_motion
from plan1_skills.library import ROOT, body_frame_motion
from plan1_skills.matching import match_sequence, tool_fk
from .ase_skills import inset_joint_ranges
from .assembly import _finite_json
from .floor import build_floor_assembly
from .shared_skills import _screen, task_segment


CONFIG_PATH = ROOT / "config/ase_pilot_v2.json"
SKILL_IDS = ("guard", "stab", "slash_left", "parry_left", "parry_right")
DEFAULT_SOURCE_DIRS = (
    ROOT / "outputs/dataset_research/ase_inventory_20261122/sources",
    ROOT / "outputs/dataset_research/ase_audit",
    ROOT / "outputs/plan1_skills/sources",
)


def load_config(config=None):
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config is not None:
        cfg.update(copy.deepcopy(config))
    if (cfg.get("schema_version") != 2 or cfg.get("source_commit") != UPSTREAM_COMMIT
            or tuple(s["id"] for s in cfg["skills"]) != SKILL_IDS):
        raise ValueError("Unsupported ASE pilot schema, source commit or skill IDs")
    if (cfg.get("policy_family") != "A1" or cfg.get("action_space_kind") != "finite_skill_selection"
            or cfg.get("robot_slot_independent") is not True or cfg.get("a2_continuous_action_space") is not False):
        raise ValueError("This candidate library is for the A1 policy family, not A2 continuous control")
    rotation = np.asarray(cfg["rotation_matrix"], dtype=float)
    grip = np.asarray(cfg["guard_grip_local_m"], dtype=float)
    if (rotation.shape != (3, 3) or not np.isfinite(rotation).all()
            or not np.allclose(rotation.T@rotation, np.eye(3), atol=1e-10)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-10)
            or grip.shape != (3,) or not np.isfinite(grip).all()):
        raise ValueError("Finite Guard and proper rotation required")
    for key in ("grip_displacement_scale", "keypose_interval_s", "authored_seam_s", "guard_hold_s"):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], (float, int)) or not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"Positive finite {key} required")
    if (type(cfg["samples_per_segment"]) is not int or cfg["samples_per_segment"] < 3
            or cfg["robot_grip_to_tip_m"] != .79):
        raise ValueError("At least three samples and unchanged .79m tool length required")
    bounds = inset_joint_ranges(np.tile([-np.pi, np.pi], (7, 1)), cfg["ik_joint_margin_rad"])
    if cfg.get("guard_seed_rad") is not None:
        seed = np.asarray(cfg["guard_seed_rad"])
        if (seed.shape != (7,) or seed.dtype.kind not in "fiu" or not np.isfinite(seed).all()
                or np.any(seed < bounds[:, 0]) or np.any(seed > bounds[:, 1])):
            raise ValueError("guard_seed_rad must be seven finite numbers inside the hypothetical inset +/-pi bounds")
        cfg["guard_seed_rad"] = seed.astype(float).tolist()
    for skill in cfg["skills"]:
        if skill["source"] not in cfg["sources"]:
            raise ValueError("Unknown cached source")
        frames, count = skill["keyframes"], cfg["sources"][skill["source"]]["frames"]
        if (not frames or any(type(i) is not int or not 0 <= i < count for i in frames)
                or any(b <= a for a, b in zip(frames[:-1], frames[1:]))):
            raise ValueError("Ordered in-range zero-based source keyframes required")
    return cfg


@dataclass
class TaskTrajectory:
    time_s: np.ndarray
    grip_local_m: np.ndarray
    axis_local: np.ndarray
    source_frame_coordinate: np.ndarray
    segments: list


@dataclass
class PilotLibrary:
    config: dict
    trajectories: dict
    metadata: dict
    ik: dict


def _cached_motion(source, directories):
    # No downloads or unverified fallback. If a same-named cache is corrupt,
    # report it instead of silently choosing another directory.
    path = next((Path(p)/source["filename"] for p in directories
                 if (Path(p)/source["filename"]).is_file()), None)
    if path is None:
        raise FileNotFoundError(f"Verified local source required: {source['filename']}")
    motion = read_verified_motion(path, source["sha256"], source["size_bytes"])
    if motion["frames"] != source["frames"] or not np.isclose(motion["fps"], source["fps"], rtol=1e-10):
        raise ValueError("Source frame/fps metadata mismatch")
    return motion, path.resolve()


def _trajectory(guard_grip, guard_axis, mapped_grip, mapped_axes, frame_ids, cfg, is_guard=False):
    pieces = []
    if is_guard:
        pieces.append(("guard_hold", guard_grip, guard_grip, guard_axis, guard_axis,
                       cfg["guard_hold_s"], None))
    else:
        pieces.append(("authored_guard_entry", guard_grip, mapped_grip[0], guard_axis,
                       mapped_axes[0], cfg["authored_seam_s"], None))
        for n, (a, b) in enumerate(zip(frame_ids[:-1], frame_ids[1:])):
            pieces.append((f"ase_derived_{a}_{b}", mapped_grip[n], mapped_grip[n+1],
                           mapped_axes[n], mapped_axes[n+1], cfg["keypose_interval_s"], [a, b]))
        pieces.append(("authored_guard_exit", mapped_grip[-1], guard_grip, mapped_axes[-1],
                       guard_axis, cfg["authored_seam_s"], None))
    times, grips, axes, coordinates, segments = [], [], [], [], []
    clock = 0.
    for label, p0, p1, a0, a1, duration, source in pieces:
        phase, pp, aa = task_segment(p0, p1, a0, a1, cfg["samples_per_segment"])
        local_time = clock+duration*phase
        # -1 explicitly denotes authored sections. Intermediate source frame
        # coordinates are provenance only: they are NOT retained native frames.
        fc = np.full(len(phase), -1.) if source is None else source[0]+phase*(source[1]-source[0])
        start_row = max(0, len(times)-1)
        take = slice(None) if not times else slice(1, None)
        times.extend(local_time[take]); grips.extend(pp[take]); axes.extend(aa[take]); coordinates.extend(fc[take])
        segments.append({"name": label, "start_s": clock, "end_s": clock+duration,
                         "start_row": start_row, "end_row": len(times)-1,
                         "source_frame_interval": source,
                         "interpolation": "quintic grip; shortest-arc unit-axis; not native in-between frames"})
        clock += duration
    return TaskTrajectory(np.asarray(times), np.asarray(grips), np.asarray(axes), np.asarray(coordinates), segments)


def build_pilot(config=None, *, source_dirs=None, with_ik=False):
    """Generate candidates from explicit cached source keyframes; never admit them.

    No source rows are discarded by IK: all authored trajectory rows and failure
    indices remain in the result. Keypose selection itself is an explicit
    adaptation recorded in config/metadata, not full-clip motion imitation.
    """
    cfg = load_config(config)
    directories = DEFAULT_SOURCE_DIRS if source_dirs is None else tuple(source_dirs)
    motions, source_paths, body = {}, {}, {}
    for key, source in cfg["sources"].items():
        motions[key], source_paths[key] = _cached_motion(source, directories)
        body[key] = body_frame_motion(motions[key])
    reference_grip, reference_axis = body["ready"][0][0], body["ready"][1][0]
    rotation = np.asarray(cfg["rotation_matrix"], dtype=float)
    guard_grip = np.asarray(cfg["guard_grip_local_m"], dtype=float)
    guard_axis = rotation@reference_axis
    trajectories, skill_metadata = {}, {}
    for skill in cfg["skills"]:
        key, indices = skill["source"], skill["keyframes"]
        grip, axes = body[key]
        selected_grip, selected_axes = grip[indices], axes[indices]
        mapped_grip = guard_grip+cfg["grip_displacement_scale"]*(selected_grip-reference_grip)@rotation.T
        mapped_axes = selected_axes@rotation.T
        trajectories[skill["id"]] = _trajectory(guard_grip, guard_axis, mapped_grip, mapped_axes,
                                                   indices, cfg, skill["id"] == "guard")
        skill_metadata[skill["id"]] = {
            "policy_family": "A1",
            "source": key, "zero_based_keyframes": indices,
            "selected_source_time_s": motions[key]["time"][indices].tolist(),
            "source_frame_count": motions[key]["frames"],
            "all_native_frames_retained": False,
            "authored_candidate_duration_s": float(trajectories[skill["id"]].time_s[-1]),
            "timing_status": "authored geometric prototype timing, NOT approved native motor execution",
            "keypose_selection": "explicit authored candidate indices; not expert action/outcome labels",
            "source_first_grip_difference_from_ready_m": (grip[0]-reference_grip).tolist(),
            "source_first_axis_difference_from_ready_deg": float(np.rad2deg(np.arccos(np.clip(axes[0]@reference_axis, -1., 1.)))),
            "selected_source_grip_in_fixed_body_frame_m": selected_grip.tolist(),
            "selected_source_axis_in_fixed_body_frame": selected_axes.tolist(),
            "admitted_to_training": False, "physical_attack_or_defence_validation": "NOT RUN",
        }
    metadata = {
        "schema_version": 2, "library_id": cfg["library_id"], "status": "candidate_only",
        "policy_family": "A1", "action_space_kind": "finite_skill_selection",
        "robot_slot_independent": True, "a2_continuous_action_space": False,
        "role_scope": "A1-family skill candidates usable in either robot slot; not an A2 action restriction or a self-play implementation",
        "may_start_RL": False, "admitted_to_training": False,
        "conversion": "ASE-derived sparse keyposes; source indices, authored interpolation and new time recorded",
        "fixed_base_approximation": "remove root translation; fixed first horizontal shoulder frame per clip; torso rotation NOT removed",
        "source_grip_definition": "right_hand skeletal origin; not calibrated clamp",
        "source_grip_to_tip_m": .74, "robot_grip_to_tip_m": .79,
        "guard_axis_local": guard_axis.tolist(), "sources": cfg["sources"],
        "source_paths": {k: str(v) for k, v in source_paths.items()}, "source_commit": cfg["source_commit"],
        "source_notice": cfg["source_notice"], "skills": skill_metadata,
        "runtime_opponent_inputs": False, "physics_model_parameters_modified": False,
        "joint_seams_certified": False,
        "required_next_gate": "actual-model geometry then native dual-arm full-window attack/parry v2 validation; no automatic admission",
    }
    result = PilotLibrary(cfg, trajectories, metadata, {})
    if with_ik:
        fit_pilot(result)
    return result


def fit_pilot(library):
    """Optional whole-candidate IK, preserving errors and colliding/failed rows."""
    cfg = library.config
    assembly = build_floor_assembly(joint_ranges=np.tile([-np.pi, np.pi], (7, 1)))
    _, _, tip = tool_fk(assembly.model)
    grip, _, _ = tool_fk(assembly.model)
    if not np.isclose(np.linalg.norm(tip-grip), .79, atol=1e-8):
        raise ValueError("Actual robot grip-to-tip length changed")
    bounds = inset_joint_ranges(assembly.model.jnt_range, cfg["ik_joint_margin_rad"])
    options = dict(seed=cfg["solver_seed"], max_iterations=cfg["solver_max_iterations"],
                   position_tolerance_m=1e-4, angle_tolerance_rad=np.deg2rad(.05),
                   tip_tolerance_m=.0005, interpolation_step_rad=.025, collision_threshold_m=1e-8)
    guard = library.trajectories["guard"]
    guard_seed = cfg.get("guard_seed_rad")
    fit = match_sequence(assembly.model, guard.grip_local_m[:1], guard.axis_local[:1], bounds,
                         initial_q=guard_seed, first_frame_starts=1 if guard_seed is not None else 12, **options)
    for name, trajectory in library.trajectories.items():
        match = match_sequence(assembly.model, trajectory.grip_local_m, trajectory.axis_local, bounds,
                               initial_q=fit.qpos[0], first_frame_starts=1, **options)
        screen = _screen(assembly, match.qpos, cfg["ik_joint_margin_rad"])
        library.ik[name] = match
        library.metadata["skills"][name].update({"ik": match.summary(),
            "all_authored_rows_retained": len(match.qpos) == len(trajectory.time_s),
            "strict_static_and_interpolated_screen": screen,
            "joint_start_difference_from_common_guard_rad": (match.qpos[0]-fit.qpos[0]).tolist(),
            "joint_end_difference_from_common_guard_rad": (match.qpos[-1]-fit.qpos[0]).tolist(),
            "joint_start_end_max_difference_rad": float(np.max(np.abs(match.qpos[-1]-match.qpos[0]))),
            "diagnostic_peak_reference_speed_rad_s": float(np.max(np.abs(np.diff(match.qpos, axis=0)/np.diff(trajectory.time_s)[:, None]))),
            "joint_seams_certified": False,
            "geometry_candidate_only": bool(fit.success.all() and match.success.all()
                                             and screen["candidate_passes_geometry_screen"])})
    library.metadata["ik_initial_guard"] = fit.summary()
    library.metadata["guard_solver_initialization"] = {
        "explicit_guard_seed_rad": guard_seed,
        "first_frame_starts": 1 if guard_seed is not None else 12,
        "scope": "reproducible IK branch selection; not calibrated hardware limits or valid physical placement path",
    }
    library.metadata["unchanged_model_source_hashes"] = assembly.source_hashes
    library.metadata["ik_scope"] = "sampled proxy IK/interpolation only; no dynamics, scoring or validated skill switching"
    return library


def export_pilot(library, output_dir):
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Preserve evidence: choose a new or empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    for name, trajectory in library.trajectories.items():
        arrays = {"time_s": trajectory.time_s, "grip_local_m": trajectory.grip_local_m,
                  "axis_local": trajectory.axis_local, "source_frame_coordinate": trajectory.source_frame_coordinate}
        if name in library.ik:
            fit = library.ik[name]
            arrays.update(qpos=fit.qpos, pose_success=fit.pose_success, collision_free=fit.collision_free,
                          position_error_m=fit.position_error_m, angle_error_rad=fit.angle_error_rad,
                          tip_error_m=fit.tip_error_m)
        np.savez_compressed(output/f"{name}.npz", **arrays)
    payload = copy.deepcopy(library.metadata)
    payload["config"] = library.config
    payload["segments"] = {k: t.segments for k, t in library.trajectories.items()}
    payload["generator_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (output/"pilot.json").write_text(json.dumps(_finite_json(payload), indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return output/"pilot.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--source-dir", type=Path, action="append")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ik", action="store_true", help="Retain actual-tool IK failures, not physical certification")
    args = parser.parse_args()
    cfg = None if args.config is None else json.loads(args.config.read_text(encoding="utf-8"))
    library = build_pilot(cfg, source_dirs=args.source_dir, with_ik=args.ik)
    print(export_pilot(library, args.output))
    print("A1 POLICY-FAMILY CANDIDATE ONLY: not A2 continuous control; physical attack/parry validation NOT RUN; may_start_RL=False")


if __name__ == "__main__":
    main()
