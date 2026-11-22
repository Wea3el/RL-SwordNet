"""Optional, authored strike enhancement of an ASE-derived A1 candidate.

This is offline development calibration, not an expert ASE demonstration, an
online interception oracle, a proven defensive skill, or training admission.
Original source arrays remain unchanged; adapted numeric arrays are additional
fields and the changed runtime curve is declared in provenance and reports.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from plan1_skills.matching import MatchResult, match_sequence, screen_joint_path
from .a1_skills import (A1Library, DEFAULT_PILOT, SKILL_IDS, _build_path, _check_curve,
                        _peak_derivatives, build_a1_library, export_a1_library)
from .ase_skills import inset_joint_ranges
from .shared_skills import _screen, quintic


DEFAULT_RECIPE = {
    "schema_version": 1, "skill_id": "slash_left", "amplitude": .8,
    "center_source_time_s": 3.45, "half_width_source_s": 1.35,
    "peak_vector_local_m": [.20, .037, -.106], "continuation_step": .1,
    "solver_seed": 20261122, "solver_max_iterations": 160,
    "position_tolerance_m": 1e-4, "axis_tolerance_deg": .05,
    "tip_tolerance_m": .0005, "interpolation_step_rad": .025,
    "collision_threshold_m": 1e-8,
}
DEVELOPMENT_REFERENCE = {
    "path": "outputs/a1_slash_adaptation_development_source_guided_runner.py",
    "sha256": "8024d9ff3ccccae99ba0115c866b50851e7b2746f31e81fdab9fd2c1a180ed0a",
    "scope": "independent earlier linear-retimed native probe; its physical results are NOT inherited by this C2 bank",
}


def _recipe(recipe=None):
    result = copy.deepcopy(DEFAULT_RECIPE)
    if recipe is not None:
        if set(recipe)-set(result):
            raise ValueError("Unknown A1 adaptation recipe parameters")
        result.update(copy.deepcopy(recipe))
    if result["schema_version"] != 1 or result["skill_id"] != "slash_left":
        raise ValueError("Only the explicit SlashLeft adaptation recipe is implemented")
    fixed = ("solver_seed", "solver_max_iterations", "position_tolerance_m", "axis_tolerance_deg",
             "tip_tolerance_m", "interpolation_step_rad", "collision_threshold_m")
    if any(result[key] != DEFAULT_RECIPE[key] for key in fixed):
        raise ValueError("Do not silently relax pinned adaptation solver/screen tolerances")
    for key in ("amplitude", "center_source_time_s", "half_width_source_s", "continuation_step"):
        if isinstance(result[key], bool) or not isinstance(result[key], (float, int)) or not np.isfinite(result[key]) or result[key] <= 0:
            raise ValueError(f"Positive finite {key} required")
    if result["amplitude"] > 1.2 or int(np.ceil(result["amplitude"]/result["continuation_step"])) > 20:
        raise ValueError("Adaptation continuation must stay within the bounded development budget")
    vector = np.asarray(result["peak_vector_local_m"], dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all() or np.linalg.norm(vector) > .30:
        raise ValueError("Finite bounded local grip-offset vector required")
    result["peak_vector_local_m"] = vector.tolist()
    return result


def authored_grip_offsets(source_time_s, recipe=None):
    """Compact support in ORIGINAL candidate clock, independent of runtime scale."""
    recipe = _recipe(recipe)
    times = np.asarray(source_time_s, dtype=float)
    if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
            or times[0] != 0. or not np.all(np.diff(times) > 0)):
        raise ValueError("Ordered original source-candidate clock required")
    center, width = recipe["center_source_time_s"], recipe["half_width_source_s"]
    if not times[0] < center-width < center+width < times[-1]:
        raise ValueError("Offset support must be strictly inside the source clock; Guard endpoints cannot change")
    weights = quintic(np.clip(1.-np.abs(times-center)/width, 0., 1.))
    offsets = recipe["amplitude"]*weights[:, None]*np.asarray(recipe["peak_vector_local_m"])
    return offsets, weights


def source_guided_match(model, base_grips, axes, original_qpos, common_guard, bounds, weights, recipe=None):
    """Offline continuation from each ORIGINAL row's own IK branch.

    Independent row solutions are NOT physical jumps: every final adjacent
    joint path is checked separately, then the actual C2 runtime curve is
    screened again by the caller. Failures and all row indices are retained.
    """
    recipe = _recipe(recipe)
    base_grips, axes, guide = (np.asarray(base_grips, dtype=float), np.asarray(axes, dtype=float),
                               np.asarray(original_qpos, dtype=float).copy())
    weights = np.asarray(weights, dtype=float)
    count = len(guide)
    if (guide.shape != (count, 7) or base_grips.shape != (count, 3) or axes.shape != (count, 3)
            or weights.shape != (count,) or not all(np.isfinite(a).all() for a in (guide, base_grips, axes, weights))
            or not np.allclose(np.linalg.norm(axes, axis=1), 1., atol=1e-7)
            or np.any((weights < 0) | (weights > 1))):
        raise ValueError("Finite matched source q/grip/axis/weight rows required")
    options = dict(seed=recipe["solver_seed"], max_iterations=recipe["solver_max_iterations"],
        position_tolerance_m=recipe["position_tolerance_m"], angle_tolerance_rad=np.deg2rad(recipe["axis_tolerance_deg"]),
        tip_tolerance_m=recipe["tip_tolerance_m"], collision_threshold_m=recipe["collision_threshold_m"],
        interpolation_step_rad=recipe["interpolation_step_rad"], first_frame_starts=1)
    levels = np.linspace(0., recipe["amplitude"], int(np.ceil(recipe["amplitude"]/recipe["continuation_step"]))+1)[1:]
    history, final = [], []
    vector = np.asarray(recipe["peak_vector_local_m"])
    for level in levels:
        final = []
        desired = base_grips+level*weights[:, None]*vector
        for row in range(count):
            fit = match_sequence(model, desired[row:row+1], axes[row:row+1], bounds,
                                 initial_q=guide[row].copy(), **options)
            guide[row] = fit.qpos[0]
            final.append(fit)
        history.append({"amplitude": float(level), "failed_pose_rows": [i for i, fit in enumerate(final) if not fit.pose_success[0]]})
    contacts, samples, collision_free = [], [], []
    for row, q in enumerate(guide):
        found, size = screen_joint_path(model, q if row == 0 else guide[row-1], q,
            interpolation_step_rad=recipe["interpolation_step_rad"], collision_threshold_m=recipe["collision_threshold_m"])
        contacts.append(found); samples.append(size); collision_free.append(not found)
    poses, free = np.asarray([fit.pose_success[0] for fit in final], dtype=bool), np.asarray(collision_free, dtype=bool)
    result = MatchResult(guide, np.array([f.position_error_m[0] for f in final]),
        np.array([f.angle_error_rad[0] for f in final]), np.array([f.tip_error_m[0] for f in final]),
        np.vstack([guide[0]-common_guard, np.diff(guide, axis=0)]), poses, free, poses & free,
        contacts, np.array([f.iterations[0] for f in final]), np.asarray(samples))
    guidance = {"method": "per-original-row continuation in grip-offset amplitude; not temporal previous-row warm-start",
                "all_intermediate_solver_stages": history, "not_physical_execution": True,
                "final_adjacent_joint_paths_separately_screened": True,
                "all_final_rows_retained": len(result.qpos) == len(original_qpos)}
    return result, guidance


def _array_hash(array):
    value = np.ascontiguousarray(array)
    return hashlib.sha256(value.dtype.str.encode()+str(value.shape).encode()+value.tobytes()).hexdigest()


def adapt_a1_library(base, recipe=None):
    """Return a separate adapted A1Library; original source arrays are untouched."""
    recipe = _recipe(recipe)
    if (base.config.get("policy_family") != "A1" or tuple(base.skills) != SKILL_IDS
            or base.reports["library"].get("admitted_to_training") is not False):
        raise ValueError("An unadmitted five-skill A1 library is required")
    library = A1Library(copy.deepcopy(base.config), base.single_assembly, base.guard_q.copy(), dict(base.skills),
                        copy.deepcopy(base.reports), copy.deepcopy(base.source_arrays), copy.deepcopy(base.provenance))
    skill = recipe["skill_id"]
    original = base.source_arrays[skill]
    offsets, weights = authored_grip_offsets(original["time_s"], recipe)
    adapted_grip = original["grip_local_m"]+offsets
    bounds = inset_joint_ranges(base.single_assembly.model.jnt_range, base.config["ik_joint_margin_rad"])
    fit, guidance = source_guided_match(base.single_assembly.model, original["grip_local_m"], original["axis_local"],
                                        original["qpos"], base.guard_q, bounds, weights, recipe)
    raw_screen = _screen(base.single_assembly, fit.qpos, base.config["ik_joint_margin_rad"])
    runtime_arrays = dict(original, qpos=fit.qpos, grip_local_m=adapted_grip)
    source_segments = base.provenance["source_pilot_metadata"]["segments"][skill]
    path = _build_path(skill, base.skills[skill].action_id, runtime_arrays, source_segments, base.guard_q, base.config)
    continuity = _check_curve(path.knot_time_s, path.knot_qpos, path.coefficients)
    smooth_screen = _screen(base.single_assembly, path.qpos, base.config["ik_joint_margin_rad"])
    closed = np.array_equal(path.qpos[0], base.guard_q) and np.array_equal(path.qpos[-1], base.guard_q)
    passed = bool(fit.success.all() and raw_screen["candidate_passes_geometry_screen"]
                  and smooth_screen["candidate_passes_geometry_screen"] and closed)
    library.skills[skill] = path
    # Additional fields are saved as source_adapted_* by the existing numeric
    # exporter. NEVER replace qpos/time_s/grip_local_m/axis_local source fields.
    library.source_arrays[skill].update(adapted_qpos=fit.qpos.copy(), adapted_grip_local_m=adapted_grip.copy(),
        adapted_axis_local=original["axis_local"].copy(), authored_grip_offset_m=offsets,
        adaptation_weights=weights, adapted_pose_success=fit.pose_success.copy(), adapted_collision_free=fit.collision_free.copy())
    original_hashes = {key: _array_hash(value) for key, value in original.items()}
    if any(not np.array_equal(library.source_arrays[skill][key], value) for key, value in original.items()):
        raise RuntimeError("Original source arrays must never change during adaptation")
    report = dict(smooth_screen, action_id=path.action_id, exact_guard_start=bool(np.array_equal(path.qpos[0], base.guard_q)),
        exact_guard_end=bool(np.array_equal(path.qpos[-1], base.guard_q)), candidate_passes_skill_geometry=passed,
        source_rows_preserved=True, original_source_arrays_unchanged=True, source_qpos_used_as_runtime_knots=False,
        adapted_rows_retained=len(fit.qpos), source_time_scale=base.config["source_time_scale"], duration_s=path.duration_s,
        authored_strike_enhancement_not_original_ASE=True, source_ik=fit.summary(), source_ik_guidance=guidance,
        adapted_raw_qpos_screen=raw_screen, command_continuity=continuity,
        derivative_diagnostics=_peak_derivatives(path.knot_time_s, path.coefficients),
        physical_attack_or_defence_validation="NOT RUN for this C2 library; earlier linear probe results are NOT inherited",
        admitted_to_training=False)
    library.reports[skill] = report
    library.reports["library"].update(
        all_skills_pass_geometry=all(library.reports[n]["candidate_passes_skill_geometry"] for n in SKILL_IDS),
        exact_shared_joint_guard_seams=all(library.reports[n]["exact_guard_start"] and library.reports[n]["exact_guard_end"] for n in SKILL_IDS),
        status="executable_candidate", admitted_to_training=False, may_start_RL=False,
        native_physical_joint_seams_certified=False, authored_adaptation_present=True,
        required_next_gate="fresh native C2 full-window mirrored attack/defence/transition validation; earlier linear probe success cannot be inherited")
    library.provenance["adaptation_recipe"] = {
        "parameters": recipe, "source_time_scale": library.config["source_time_scale"],
        "scope": "offline development-calibrated ASE-derived authored strike enhancement, NOT original expert motion or held-out performance",
        "formula": "amplitude * quintic(clip(1-abs(original_candidate_time-center)/half_width,0,1)) * peak_vector",
        "original_source_indices_and_clock_unchanged": True, "source_axes_unchanged": True,
        "common_guard_unchanged": True, "model_target_masks_caps_unchanged": True,
        "runtime_opponent_inputs": False, "original_array_sha256": original_hashes,
        "adapted_qpos_sha256": _array_hash(fit.qpos), "adapted_grip_sha256": _array_hash(adapted_grip),
        "guidance": guidance, "development_reference": DEVELOPMENT_REFERENCE,
        "adapter_implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_bank_generator_sha256": base.provenance.get("generator_sha256"),
        "input_sha256": copy.deepcopy(base.provenance.get("input_sha256", {})),
    }
    return library


def build_adapted_a1_library(pilot_dir=DEFAULT_PILOT, config=None, recipe=None):
    cfg = dict(config or {})
    cfg.setdefault("source_time_scale", 4.)
    return adapt_a1_library(build_a1_library(pilot_dir, cfg), recipe)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--source-time-scale", type=float, default=4.)
    parser.add_argument("--amplitude", type=float, default=.8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    library = build_adapted_a1_library(args.pilot_dir, {"source_time_scale": args.source_time_scale}, {"amplitude": args.amplitude})
    print(export_a1_library(library, args.output))
    print("Authored adapted A1 candidate only; geometry:", library.reports["library"]["all_skills_pass_geometry"], "; admission: False")


if __name__ == "__main__":
    main()
