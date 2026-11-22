"""Controlled paired physics tests of hand-authored thrust and side-parry.

One trial keeps the defender at Guard; its matched partner executes a scheduled
parry. The attack targets/controller/window are IDENTICAL and never change in
response to a contact or score. Only ctrl drives motion after the initial reset.
This is skill validation, not a trained/reactive agent or self-play environment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from .assembly import ROOT, _finite_json
from .duel_scene import build_duel_scene, duel_contact_records
from .manual_skills import build_manual_library

DEFAULT_CONFIG = ROOT / "config/manual_duel.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def swept_sphere_fraction(start, end, center, radius):
    """Earliest intersection of a point segment with a sphere, or None."""
    start, end, center = (np.asarray(v, dtype=float) for v in (start, end, center))
    if any(v.shape != (3,) or not np.isfinite(v).all() for v in (start, end, center)) or not np.isfinite(radius) or radius <= 0:
        raise ValueError("Finite three-vectors and a positive sphere radius required")
    relative, delta = start-center, end-start
    c = relative@relative-radius**2
    if c <= 0:
        return 0.
    a = delta@delta
    if a < 1e-24:
        return None
    b = 2*(relative@delta)
    discriminant = b*b-4*a*c
    if discriminant < 0:
        return None
    at = (-b-np.sqrt(max(0., discriminant)))/(2*a)
    return float(at) if 0 <= at <= 1 else None


def sample_q(phase, q, at):
    return np.array([np.interp(np.clip(at, 0, 1), phase, q[:, i]) for i in range(q.shape[1])])


def minimum_duration(phase, q, requested, ceiling):
    """Conservative finite-difference bound for the authored phase trajectory."""
    derivative = np.max(np.abs(np.diff(q, axis=0)/np.diff(phase)[:, None]))
    return max(float(requested), float(derivative/ceiling)*1.01)


def schedule(library, cfg, speed_scale, delay):
    if not np.isfinite([speed_scale, delay]).all() or speed_scale <= 0:
        raise ValueError("A positive speed scale and finite delay required")
    ceiling = cfg["command_speed_ceiling_rad_s"]
    forward = minimum_duration(library.attack_phase, library.attack_q, cfg["attack_duration_s"]/speed_scale, ceiling)
    backward = minimum_duration(library.attack_phase, library.attack_q, cfg["attack_return_duration_s"]/speed_scale, ceiling)
    parry = minimum_duration(library.parry_phase, library.parry_q, cfg["parry_duration_s"], ceiling)
    parry_back = minimum_duration(library.parry_phase, library.parry_q, cfg["parry_return_duration_s"], ceiling)
    start = cfg["initial_guard_hold_s"]
    if start+delay < 0:
        raise ValueError("Parry must not start before the common reset")
    attack_end = start+forward+cfg["attack_extended_hold_s"]+backward
    release = max(attack_end+.2, start+delay+parry)
    return {"attack_start_s": start, "attack_duration_s": forward,
            "attack_hold_s": cfg["attack_extended_hold_s"], "attack_return_s": backward,
            "attack_end_s": attack_end, "parry_start_s": start+delay,
            "parry_duration_s": parry, "parry_release_s": release, "parry_return_s": parry_back,
            "duration_s": release+parry_back+cfg["final_guard_hold_s"],
            "requested_attack_speed_scale": float(speed_scale), "requested_parry_delay_s": float(delay)}


def attack_command(library, timing, time):
    elapsed = time-timing["attack_start_s"]
    if elapsed <= 0:
        return library.guard_attack_q.copy()
    if elapsed < timing["attack_duration_s"]:
        phase = elapsed/timing["attack_duration_s"]
    elif elapsed < timing["attack_duration_s"]+timing["attack_hold_s"]:
        phase = 1.
    else:
        phase = 1-(elapsed-timing["attack_duration_s"]-timing["attack_hold_s"])/timing["attack_return_s"]
    return sample_q(library.attack_phase, library.attack_q, phase)


def parry_command(library, timing, time, enabled):
    if not enabled or time <= timing["parry_start_s"]:
        return library.guard_defend_q.copy()
    if time < timing["parry_start_s"]+timing["parry_duration_s"]:
        phase = (time-timing["parry_start_s"])/timing["parry_duration_s"]
    elif time < timing["parry_release_s"]:
        phase = 1.
    else:
        phase = 1-(time-timing["parry_release_s"])/timing["parry_return_s"]
    return sample_q(library.parry_phase, library.parry_q, phase)


def position_control(scene, gravity, commands):
    model, data = scene.model, scene.data
    gravity.qpos[:] = data.qpos
    gravity.qvel[:] = 0
    mujoco.mj_forward(model, gravity)
    for owner, target in commands.items():
        mapping = scene.agents[owner]
        actuator, dof = mapping["actuator_ids"], mapping["dof_ids"]
        output = target+gravity.qfrc_bias[dof]/model.actuator_gainprm[actuator, 0]
        data.ctrl[actuator] = np.clip(output, model.actuator_ctrlrange[actuator, 0], model.actuator_ctrlrange[actuator, 1])


def run_trial(scene, library, cfg, *, parry_enabled, speed_scale=1., delay=0., mirrored=False, output=None, program=None):
    model, data = scene.model, scene.data
    # A dataset clip may already include its recovery. An explicit program can
    # play it once, without misusing the manual forward/hold/reverse contract.
    timing = (schedule(library, cfg, speed_scale, delay) if program is None
              else program.schedule(library, cfg, speed_scale, delay))
    attacker, defender = ("a2", "a1") if mirrored else ("a1", "a2")
    aid, did = scene.agents[attacker], scene.agents[defender]
    target = np.asarray(scene.config["virtual_targets"][defender]["center_world_m"])
    scene_radius = scene.config["virtual_targets"][defender]["radius_m"]
    if not np.isclose(scene_radius, cfg["virtual_target_radius_m"], rtol=0., atol=1e-12):
        raise ValueError("Scored target radius must match the visible scene fixture")
    radius = scene_radius+cfg["scoring_tip_radius_m"]
    mujoco.mj_resetData(model, data)
    data.qpos[aid["qpos_ids"]] = library.guard_attack_q
    data.qpos[did["qpos_ids"]] = library.guard_defend_q
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    initial_q = data.qpos.copy()
    initial_contacts = duel_contact_records(scene)
    gravity = mujoco.MjData(model)
    timestep, period = cfg["physics_timestep_s"], cfg["control_period_s"]
    if not np.isclose(model.opt.timestep, timestep) or not np.isclose(period/timestep, round(period/timestep)):
        raise ValueError("Physics/control periods must match the model and divide exactly")
    if np.any(model.body_gravcomp) or np.any(data.qfrc_applied) or np.any(data.xfrc_applied):
        raise ValueError("No external forces or body gravity cancellation are permitted")
    steps = int(np.ceil(timing["duration_s"]/timestep))
    control_steps = int(round(period/timestep))
    caps = np.minimum(-model.actuator_forcerange[:, 0], model.actuator_forcerange[:, 1])
    peak_torque, peak_speed, peak_error = np.zeros(model.nu), np.zeros(model.nv), np.zeros(model.nq)
    saturation = np.zeros(model.nu, dtype=int)
    force_peak, impulse, penetration = 0., 0., 0.
    illegal_peak, illegal_impulse = 0., 0.
    joint_limit_force_peak, joint_limit_active_steps = 0., 0
    minimum_joint_margin = np.inf
    first_blade_contact, first_native_contact, first_hit, min_distance = None, None, None, np.inf
    native_impulse = 0.
    positive_contact_steps = 0
    contact_pairs, illegal_pairs = {}, {}
    failure, phase = None, "initial_hold"
    rows_t, rows_q, rows_v, rows_target, rows_force, rows_ctrl, rows_torque = ([] for _ in range(7))
    target_digest = hashlib.sha256()
    commands = {attacker: library.guard_attack_q.copy(), defender: library.guard_defend_q.copy()}
    position_control(scene, gravity, commands)
    mujoco.mj_forward(model, data)
    previous_tip = data.site_xpos[aid["sites"]["shinai_tip"]].copy()
    previous_time = 0.
    completed = 0.
    for step in range(steps+1):
        if step % control_steps == 0:
            commands = {attacker: (attack_command(library, timing, data.time) if program is None
                                  else program.attack_command(library, timing, data.time)),
                        defender: (parry_command(library, timing, data.time, parry_enabled) if program is None
                                   else program.parry_command(library, timing, data.time, parry_enabled))}
            target_digest.update(np.asarray(commands[attacker], dtype="<f8").tobytes())
            position_control(scene, gravity, commands)
        mujoco.mj_forward(model, data)
        time = float(data.time)
        completed = time
        full_target = np.empty(model.nq)
        full_target[aid["qpos_ids"]] = commands[attacker]
        full_target[did["qpos_ids"]] = commands[defender]
        finite = all(np.isfinite(a).all() for a in [data.qpos, data.qvel, data.qacc, data.qfrc_actuator])
        if not finite:
            failure = "nonfinite_state"
            break
        torque = np.abs(data.actuator_force)
        speed = np.abs(data.qvel)
        error = np.abs(data.qpos-full_target)
        minimum_joint_margin = min(minimum_joint_margin, float(np.min(np.minimum(
            data.qpos-model.jnt_range[:, 0], model.jnt_range[:, 1]-data.qpos))))
        limit_rows = np.flatnonzero(data.efc_type[:data.nefc] == int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT))
        limit_force = float(np.max(np.abs(data.efc_force[limit_rows]), initial=0.))
        joint_limit_force_peak = max(joint_limit_force_peak, limit_force)
        joint_limit_active_steps += limit_force > 1e-8
        peak_torque, peak_speed, peak_error = np.maximum(peak_torque, torque), np.maximum(peak_speed, speed), np.maximum(peak_error, error)
        saturation += torque >= caps*(1-1e-6)
        contacts = duel_contact_records(scene)
        force = 0.
        for contact in contacts:
            normal = max(0., contact["normal_force_n"])
            pair = " / ".join(sorted([contact["geom1"], contact["geom2"]]))
            penetration = max(penetration, -contact["distance_m"])
            if contact["category"] == "cross_blade_blade":
                force += normal
                record = contact_pairs.setdefault(pair, {"peak_force_n": 0., "minimum_distance_m": 0.})
                record["peak_force_n"] = max(record["peak_force_n"], normal)
                record["minimum_distance_m"] = min(record["minimum_distance_m"], contact["distance_m"])
            else:
                illegal_peak = max(illegal_peak, normal)
                illegal_impulse += normal*timestep
                record = illegal_pairs.setdefault(pair, {"category": contact["category"], "peak_force_n": 0., "minimum_distance_m": 0.})
                record["peak_force_n"] = max(record["peak_force_n"], normal)
                record["minimum_distance_m"] = min(record["minimum_distance_m"], contact["distance_m"])
        force_peak = max(force_peak, force)
        impulse += force*timestep
        native_window = (timing.get("attack_native_start_s", np.inf) <= time <= timing.get("attack_native_end_s", -np.inf)
                         and timing.get("parry_native_start_s", np.inf) <= time <= timing.get("parry_native_end_s", -np.inf))
        if native_window:
            native_impulse += force*timestep
        if force > cfg["minimum_blade_contact_force_n"]:
            positive_contact_steps += 1
            if first_blade_contact is None:
                first_blade_contact = time
            if native_window and first_native_contact is None:
                first_native_contact = time
        tip = data.site_xpos[aid["sites"]["shinai_tip"]].copy()
        min_distance = min(min_distance, float(np.linalg.norm(tip-target)))
        if timing["attack_start_s"] <= time <= timing["attack_end_s"]:
            fraction = swept_sphere_fraction(previous_tip, tip, target, radius)
            if fraction is not None and first_hit is None:
                first_hit = previous_time+fraction*(time-previous_time)
        if step % 5 == 0 or step == steps:
            rows_t.append(time); rows_q.append(data.qpos.copy()); rows_v.append(data.qvel.copy())
            rows_target.append(full_target.copy()); rows_force.append(force)
            rows_ctrl.append(data.ctrl.copy()); rows_torque.append(data.actuator_force.copy())
        if any(w.number for w in data.warning):
            failure = "mujoco_warning"
        elif np.any(torque > caps+1e-6):
            failure = "native_actuator_force_exceeded_cap"
        elif speed.max() > cfg["actual_speed_ceiling_rad_s"]:
            failure = "actual_joint_speed_exceeded_2rad_s"
        elif error.max() > cfg["maximum_tracking_error_rad"]:
            failure = "excessive_joint_tracking_error"
        elif illegal_peak > cfg["illegal_contact_force_threshold_n"]:
            failure = "illegal_contact_force"
        elif joint_limit_force_peak > cfg.get("hypothetical_joint_limit_force_ceiling_nm", .1):
            failure = "unidentified_joint_limit_supplied_constraint_force"
        elif penetration > cfg["legal_contact_penetration_limit_m"]:
            failure = "excessive_proxy_penetration"
        elif np.any(data.qpos < model.jnt_range[:, 0]-.005) or np.any(data.qpos > model.jnt_range[:, 1]+.005):
            failure = "joint_left_hypothetical_range"
        if failure or step == steps:
            break
        previous_tip, previous_time = tip, time
        mujoco.mj_step(model, data)
    result = {"status": "failed" if failure else "completed", "failure": failure,
              "parry_enabled": bool(parry_enabled), "mirrored": bool(mirrored), "attacker": attacker, "defender": defender,
              "timing": timing, "completed_time_s": completed, "initial_qpos": initial_q.tolist(),
              "initial_contacts": initial_contacts, "qpos_resets_after_initialization": 0,
              "target_world_m": target.tolist(), "effective_scoring_radius_m": radius,
              "target_has_collision_geometry": False,
              "target_hit": first_hit is not None, "first_target_hit_time_s": first_hit,
              "minimum_tip_target_center_distance_m": min_distance,
              "first_positive_blade_contact_time_s": first_blade_contact,
              "first_native_clip_blade_contact_time_s": first_native_contact,
              "native_clip_overlap_blade_impulse_ns": native_impulse,
              "first_target_hit_during_native_attack": bool(first_hit is not None and timing.get("attack_native_start_s", np.inf) <= first_hit <= timing.get("attack_native_end_s", -np.inf)),
              "positive_blade_contact_steps": positive_contact_steps, "blade_contact_peak_force_n": force_peak,
              "blade_contact_normal_impulse_ns": impulse, "blade_contact_pairs": contact_pairs,
              "illegal_contact_peak_force_n": illegal_peak, "illegal_contact_impulse_ns": illegal_impulse,
              "illegal_contact_pairs": illegal_pairs, "maximum_contact_penetration_m": penetration,
              "peak_hypothetical_joint_limit_constraint_force_nm": joint_limit_force_peak,
              "hypothetical_joint_limit_active_steps": int(joint_limit_active_steps),
              "minimum_actual_joint_limit_margin_rad": minimum_joint_margin,
              "peak_actuator_force_nm": peak_torque.tolist(), "unchanged_force_caps_nm": caps.tolist(),
              "peak_actual_joint_speed_rad_s": peak_speed.tolist(), "peak_joint_tracking_error_rad": peak_error.tolist(),
              "actuator_saturation_steps": saturation.tolist(), "warnings": [int(w.number) for w in data.warning],
              "attack_target_command_sha256": target_digest.hexdigest(),
              "command_program": "manual_forward_hold_reverse" if program is None else program.name,
              "limitations": "Controlled scheduled rigid-proxy experiment; not reactive defence, RL, real motor validation or continuous concave-CAD clearance."}
    if output:
        folder = Path(output)
        folder.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(folder/"trajectory.npz", time_s=rows_t, qpos=rows_q, qvel=rows_v,
                            target_qpos=rows_target, blade_normal_force_n=rows_force,
                            ctrl=rows_ctrl, actuator_force_nm=rows_torque)
        (folder/"summary.json").write_text(json.dumps(_finite_json(result), indent=2, allow_nan=False)+"\n")
    return result


def evaluate_pair(baseline, defended, cfg):
    same_attack = baseline["attack_target_command_sha256"] == defended["attack_target_command_sha256"]
    same_initial = np.array_equal(baseline["initial_qpos"], defended["initial_qpos"])
    valid = baseline["status"] == defended["status"] == "completed" and same_attack and same_initial
    before_hit = (baseline["first_target_hit_time_s"] is not None
                  and defended["first_positive_blade_contact_time_s"] is not None
                  and defended["first_positive_blade_contact_time_s"] < baseline["first_target_hit_time_s"])
    success = (valid and baseline["target_hit"] and not defended["target_hit"] and before_hit
               and defended["blade_contact_normal_impulse_ns"] >= cfg["minimum_blade_contact_impulse_ns"])
    return {"valid_paired_experiment": bool(valid), "same_attack_targets": same_attack,
            "same_initial_state": bool(same_initial), "unopposed_attack_hits": baseline["target_hit"],
            "defended_attack_hits": defended["target_hit"], "blade_contact_precedes_baseline_hit": bool(before_hit),
            "successful_physical_parry": bool(success)}


def run_experiment(output, cfg, *, matrix=False, geometry=None):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Use a new output path; existing evidence is preserved")
    output.mkdir(parents=True)
    library = build_manual_library(geometry)
    scene = build_duel_scene(library.config["base_distance_m"])
    if not np.allclose(library.config["target_world_m"], scene.config["virtual_targets"]["a2"]["center_world_m"], rtol=0., atol=1e-12):
        raise ValueError("Manual reference target must match the fixed scene target")
    if not np.isclose(library.config["target_radius_m"], cfg["virtual_target_radius_m"], rtol=0., atol=1e-12):
        raise ValueError("Manual geometry and runtime target radii must agree")
    scene.model.opt.timestep = cfg["physics_timestep_s"]
    scene.spec.option.timestep = cfg["physics_timestep_s"]
    (output/"runtime_config.json").write_text(json.dumps(cfg, indent=2, allow_nan=False)+"\n")
    (output/"geometry_config.json").write_text(json.dumps(library.config, indent=2, allow_nan=False)+"\n")
    (output/"scene.xml").write_text(scene.spec.to_xml())
    np.savez_compressed(output/"manual_paths.npz", attack_q=library.attack_q, attack_phase=library.attack_phase,
                        parry_q=library.parry_q, parry_phase=library.parry_phase,
                        guard_attack_q=library.guard_attack_q, guard_defend_q=library.guard_defend_q)
    report = {"status": "running", "scope": "manually_authored_guard_thrust_side_parry_physical_validation_no_dataset_or_RL",
              "runtime_config": cfg, "manual_geometry_config": library.config,
              "single_arm_screen": library.reports, "scene_config": scene.config,
              "source_hashes": scene.source_hashes,
              "implementation_sha256": {p: digest(ROOT/p) for p in ["sim/manual_duel.py", "sim/manual_skills.py", "sim/duel_scene.py"]},
              "software_versions": {"mujoco": mujoco.__version__, "numpy": np.__version__},
              "pairs": []}
    summary = output/"summary.json"
    summary.write_text(json.dumps(_finite_json(report), indent=2)+"\n")
    for name in ["manual_thrust", "manual_side_parry"]:
        if not library.reports[name]["candidate_passes_single_arm_screen"]:
            report["status"] = "failed_single_arm_preflight"
            summary.write_text(json.dumps(_finite_json(report), indent=2)+"\n")
            return report
    cases = [("nominal", 1., cfg["nominal_parry_delay_s"], False),
             ("late_control", 1., cfg["late_control_delay_s"], False),
             ("mirrored", 1., cfg["nominal_parry_delay_s"], True)]
    if matrix:
        cases += [(f"speed_{speed:.2f}_delay_{delay:+.2f}", speed, delay, False)
                  for speed in cfg["attack_speed_scales"] for delay in cfg["parry_start_delays_s"]
                  if not (speed == 1 and delay == cfg["nominal_parry_delay_s"])]
    for label, speed, delay, mirrored in cases:
        baseline = run_trial(scene, library, cfg, parry_enabled=False, speed_scale=speed, delay=delay,
                             mirrored=mirrored, output=output/label/"no_defence")
        defended = run_trial(scene, library, cfg, parry_enabled=True, speed_scale=speed, delay=delay,
                             mirrored=mirrored, output=output/label/"parry")
        comparison = evaluate_pair(baseline, defended, cfg)
        report["pairs"].append({"label": label, "comparison": comparison,
                                "no_defence": baseline, "parry": defended})
        summary.write_text(json.dumps(_finite_json(report), indent=2, allow_nan=False)+"\n")
        print(json.dumps({"case": label, **comparison, "baseline_failure": baseline["failure"],
                          "parry_failure": defended["failure"], "parry_min_distance_m": defended["minimum_tip_target_center_distance_m"],
                          "blade_peak_force_n": defended["blade_contact_peak_force_n"]}), flush=True)
    report["status"] = "completed_controlled_tests_not_RL_or_hardware_certification"
    report["nominal_success"] = report["pairs"][0]["comparison"]["successful_physical_parry"]
    report["successful_pairs"] = sum(pair["comparison"]["successful_physical_parry"] for pair in report["pairs"])
    report["tested_pairs"] = len(report["pairs"])
    controls = {pair["label"]: pair["comparison"] for pair in report["pairs"]}
    late = controls["late_control"]
    report["controlled_fixture_checks_pass"] = bool(
        report["nominal_success"] and controls["mirrored"]["successful_physical_parry"]
        and late["valid_paired_experiment"] and late["unopposed_attack_hits"]
        and late["defended_attack_hits"] and not late["blade_contact_precedes_baseline_hit"])
    report["matrix_interpretation"] = "Deterministic sensitivity grid, including expected mistimed failures; NOT an RL win rate or a success probability."
    summary.write_text(json.dumps(_finite_json(report), indent=2, allow_nan=False)+"\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/manual_duel_round1")
    parser.add_argument("--matrix", action="store_true", help="Also test predeclared speed/start-time variations")
    parser.add_argument("--geometry", type=Path, help="Optional manual Cartesian geometry JSON; target must match the fixed scene fixture")
    args = parser.parse_args(argv)
    cfg = json.loads(args.config.read_text())
    if cfg.get("schema_version") != 1 or cfg.get("status") != "controlled_manual_skill_validation_not_RL_or_hardware_certification":
        raise ValueError("Expected explicit manual-test assumptions")
    geometry = json.loads(args.geometry.read_text()) if args.geometry else None
    report = run_experiment(args.output, cfg, matrix=args.matrix, geometry=geometry)
    print(str(args.output.resolve()/"summary.json"), flush=True)
    return 0 if report.get("controlled_fixture_checks_pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
