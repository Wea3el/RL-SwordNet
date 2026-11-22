"""Explicit authored Cartesian defence candidates for the finite A1 family.

This builder cannot certify a parry. It keeps the original source arrays and
unchanged skills, records an authored recipe, retains failed IK rows, screens
the actual C2 command/return curve and leaves training admission FALSE. No
opponent input, target placement, contact override or motor tuning is accepted.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from plan1_skills.matching import match_sequence, tool_fk
from .a1_skills import (A1Library, SKILL_IDS, _build_path, _check_curve,
                        _peak_derivatives, export_a1_library, load_a1_library)
from .assembly import _finite_json
from .shared_skills import _screen, task_segment


DEFAULT_OPTIONS = {
    "schema_version": 1, "skill_id": "parry_right", "samples_per_segment": 17,
    "solver_seed": 20261122, "solver_max_iterations": 200,
    "position_tolerance_m": 1e-4, "axis_tolerance_deg": .05,
    "tip_tolerance_m": .0005, "interpolation_step_rad": .025,
    "collision_threshold_m": 1e-8,
}


def validate_recipe(recipe):
    if not isinstance(recipe, dict) or "keyposes" not in recipe:
        raise ValueError("An explicit authored defence recipe with keyposes is required")
    unknown = set(recipe)-set(DEFAULT_OPTIONS)-{"keyposes", "description"}
    if unknown:
        raise ValueError(f"Unknown defence options (no physics overrides): {sorted(unknown)}")
    cfg = dict(DEFAULT_OPTIONS, **copy.deepcopy(recipe))
    if type(cfg["schema_version"]) is not int or cfg["schema_version"] != 1 or cfg["skill_id"] not in {"parry_left", "parry_right"}:
        raise ValueError("Only an explicit A1 left/right defence candidate can be replaced")
    fixed = set(DEFAULT_OPTIONS)-{"skill_id", "samples_per_segment"}
    if any(type(cfg[k]) is not type(DEFAULT_OPTIONS[k]) or cfg[k] != DEFAULT_OPTIONS[k] for k in fixed):
        raise ValueError("Pinned defence solver/screen tolerances cannot be relaxed")
    samples = cfg["samples_per_segment"]
    if type(samples) is not int or not 5 <= samples <= 81:
        raise ValueError("Use 5..81 integer samples per segment")
    poses = cfg["keyposes"]
    if not isinstance(poses, list) or not 1 <= len(poses) <= 20:
        raise ValueError("Use 1..20 explicitly authored keyposes")
    names = set()
    for pose in poses:
        if not isinstance(pose, dict) or set(pose) != {"name", "grip_local_m", "axis_local", "duration_s"}:
            raise ValueError("Each keypose needs name, grip_local_m, axis_local, duration_s")
        name = pose["name"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("Unique nonempty phase names required")
        names.add(name)
        p, axis = (np.asarray(pose[k], dtype=float) for k in ("grip_local_m", "axis_local"))
        duration = pose["duration_s"]
        if (p.shape != (3,) or axis.shape != (3,) or not np.isfinite(p).all()
                or not np.isfinite(axis).all() or not np.isclose(np.linalg.norm(axis), 1., atol=1e-7)):
            raise ValueError("Finite local grip vectors and UNIT sword axes required")
        if isinstance(duration, bool) or not isinstance(duration, (float, int)) or not np.isfinite(duration) or not .1 <= duration <= 60.:
            raise ValueError("Each own-clock duration must be 0.1..60 seconds")
        pose.update(grip_local_m=p.tolist(), axis_local=axis.tolist(), duration_s=float(duration))
    if sum(p["duration_s"] for p in poses) > 180.:
        raise ValueError("Finite candidate development duration exceeded")
    cfg["description"] = str(cfg.get("description", "explicit authored finite defence candidate"))
    return cfg


def task_targets(model, guard_q, recipe):
    """One fixed local task path; no opponent observation or future clock."""
    cfg = validate_recipe(recipe)
    start_p, start_axis, _ = tool_fk(model, qpos=guard_q)
    times, grips, axes, segments = [], [], [], []
    clock = 0.
    for pose in cfg["keyposes"]:
        phase, pp, aa = task_segment(start_p, pose["grip_local_m"], start_axis,
                                     pose["axis_local"], cfg["samples_per_segment"])
        start_row = max(0, len(times)-1)
        take = slice(None) if not times else slice(1, None)
        times.extend((clock+phase*pose["duration_s"])[take])
        grips.extend(pp[take]); axes.extend(aa[take])
        segments.append({"name": pose["name"], "start_s": clock,
            "end_s": clock+pose["duration_s"], "start_row": start_row,
            "end_row": len(times)-1, "source_frame_interval": None,
            "provenance": "authored local task segment, NOT ASE source frames"})
        clock += pose["duration_s"]
        start_p, start_axis = pp[-1], aa[-1]
    return np.asarray(times), np.asarray(grips), np.asarray(axes), segments


def _hash_array(value):
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.dtype.str.encode()+str(array.shape).encode()+array.tobytes()).hexdigest()


def build_defence_candidate(base, recipe):
    """Return a new unadmitted bank; native full-window tests remain mandatory."""
    cfg = validate_recipe(recipe)
    if (base.config.get("policy_family") != "A1" or tuple(base.skills) != SKILL_IDS
            or base.reports["library"].get("admitted_to_training") is not False):
        raise ValueError("An unadmitted five-skill A1 bank is required")
    model, guard = base.single_assembly.model, base.guard_q
    time_s, grip, axis, segments = task_targets(model, guard, cfg)
    bounds = model.jnt_range+np.array([.05, -.05])
    fit = match_sequence(model, grip, axis, bounds, initial_q=guard,
        seed=cfg["solver_seed"], max_iterations=cfg["solver_max_iterations"],
        position_tolerance_m=cfg["position_tolerance_m"], angle_tolerance_rad=np.deg2rad(cfg["axis_tolerance_deg"]),
        tip_tolerance_m=cfg["tip_tolerance_m"], interpolation_step_rad=cfg["interpolation_step_rad"],
        collision_threshold_m=cfg["collision_threshold_m"], first_frame_starts=1)
    # These times are deliberately authored runtime seconds, NOT the old ASE
    # candidate clock. Other four paths retain their exact original timing.
    path_cfg = dict(base.config, source_time_scale=1.)
    name = cfg["skill_id"]
    arrays = {"qpos": fit.qpos, "time_s": time_s}
    path = _build_path(name, base.skills[name].action_id, arrays, segments, guard, path_cfg)
    continuity = _check_curve(path.knot_time_s, path.knot_qpos, path.coefficients)
    raw = _screen(base.single_assembly, fit.qpos, .05)
    smooth = _screen(base.single_assembly, path.qpos, .05)
    start = bool(np.array_equal(path.qpos[0], guard))
    end = bool(np.array_equal(path.qpos[-1], guard))
    passed = bool(fit.success.all() and raw["candidate_passes_geometry_screen"]
                  and smooth["candidate_passes_geometry_screen"] and start and end)
    result = A1Library(copy.deepcopy(base.config), base.single_assembly, guard.copy(), dict(base.skills),
                       copy.deepcopy(base.reports), copy.deepcopy(base.source_arrays), copy.deepcopy(base.provenance))
    result.skills[name] = path
    originals = {k: _hash_array(v) for k,v in base.source_arrays[name].items()}
    result.source_arrays[name].update(authored_defence_qpos=fit.qpos.copy(), authored_defence_time_s=time_s,
        authored_defence_grip_local_m=grip, authored_defence_axis_local=axis,
        authored_defence_pose_success=fit.pose_success.copy(), authored_defence_collision_free=fit.collision_free.copy(),
        authored_defence_position_error_m=fit.position_error_m.copy(), authored_defence_axis_error_rad=fit.angle_error_rad.copy(),
        authored_defence_tip_error_m=fit.tip_error_m.copy())
    if any(not np.array_equal(result.source_arrays[name][k], v) for k,v in base.source_arrays[name].items()):
        raise RuntimeError("Never overwrite original or earlier adaptation source arrays")
    result.reports[name] = dict(smooth, action_id=path.action_id, duration_s=path.duration_s,
        candidate_passes_skill_geometry=passed, exact_guard_start=start, exact_guard_end=end,
        all_authored_rows_retained=len(fit.qpos) == len(time_s), original_source_arrays_unchanged=True,
        authored_defence_not_original_ASE=True, runtime_time_basis="explicit authored seconds, no ASE time-scale multiplication",
        source_time_scale=1., ik=fit.summary(), raw_joint_path_screen=raw,
        command_continuity=continuity, derivative_diagnostics=_peak_derivatives(path.knot_time_s, path.coefficients),
        physical_attack_or_defence_validation="NOT RUN; a direction label/geometry pass never certifies protection",
        admitted_to_training=False)
    result.reports["library"].update(status="executable_candidate", admitted_to_training=False,
        may_start_RL=False, native_physical_joint_seams_certified=False, authored_defence_present=True,
        all_skills_pass_geometry=all(result.reports[n]["candidate_passes_skill_geometry"] for n in SKILL_IDS),
        exact_shared_joint_guard_seams=all(result.reports[n]["exact_guard_start"] and result.reports[n]["exact_guard_end"] for n in SKILL_IDS),
        required_next_gate="native matched whole-window attacks, parry contact causality, negatives, perturbations and transitions")
    descriptor = {"recipe": cfg, "recipe_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
        "original_array_sha256": originals, "authored_joint_path_sha256": _hash_array(fit.qpos),
        "common_guard_unchanged": True, "unchanged_other_skills": [n for n in SKILL_IDS if n != name],
        "model_target_masks_caps_unchanged": True, "runtime_opponent_inputs": False,
        "scope": "offline authored geometry; NOT source motion imitation, validated parry or training admission",
        "generator_path": "sim/a1_defence.py", "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    result.provenance.setdefault("authored_defence_recipes", {})[name] = descriptor
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bank = build_defence_candidate(load_a1_library(args.bank), json.loads(args.recipe.read_text(encoding="utf-8")))
    print(export_a1_library(bank, args.output))
    print(json.dumps(_finite_json(bank.reports["library"]), indent=2))


if __name__ == "__main__":
    main()
