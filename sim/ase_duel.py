"""Paired native-physics tests of verified full ASE clips and explicit adaptations.

Each clip is played ONCE, including its original recovery, with separate authored
guard connections. Retiming preserves every geometric sample but does not claim
source rhythm, imitation learning, or expert combat decisions.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import mujoco
import numpy as np

from .assembly import ROOT, _finite_json
from .ase_skills import build_ase_library
from .duel_scene import build_duel_scene
from .manual_duel import digest, evaluate_pair, run_trial


def retime_samples(qpos, speed, period):
    """All input samples receive distinct control-aligned times; no dropped rows."""
    q = np.asarray(qpos, dtype=float)
    if q.ndim != 2 or len(q) < 2 or not np.isfinite(q).all() or not np.isfinite([speed, period]).all() or min(speed, period) <= 0:
        raise ValueError("Finite joint samples and positive speed/control period required")
    needed = 1.01*np.max(np.abs(np.diff(q, axis=0)), axis=1)/speed
    intervals = np.maximum(1., np.ceil(needed/period))*period
    return np.r_[0., np.cumsum(intervals)]


def join_role(library, role, speed, period):
    prefix = "attack" if role == "attack" else "parry"
    q = library.attack_q if role == "attack" else library.parry_q
    native_phase = library.attack_phase if role == "attack" else library.parry_phase
    incoming, outgoing = (library.transitions[prefix+suffix]["qpos"] for suffix in ("_in", "_out"))
    if not np.allclose(incoming[-1], q[0], atol=1e-10) or not np.allclose(outgoing[0], q[-1], atol=1e-10):
        raise ValueError("Guard connections must meet complete native clips continuously")
    full = np.vstack([incoming, q[1:], outgoing[1:]])
    time = retime_samples(full, speed, period)
    first, last = len(incoming)-1, len(incoming)+len(q)-2
    return {"qpos": full, "time_s": time, "native_start_row": first,
            "native_end_row": last, "native_phase": native_phase.copy(),
            "native_row_indices": np.arange(first, last+1),
            "duration_s": float(time[-1]), "native_start_s": float(time[first]),
            "native_end_s": float(time[last])}


def sample_timed(path, time):
    return np.array([np.interp(time, path["time_s"], path["qpos"][:, i])
                     for i in range(path["qpos"].shape[1])])


class AseClipProgram:
    name = "ASE_complete_clips_once_plus_explicit_guard_connections_geometric_retiming"

    def schedule(self, library, cfg, speed_scale, delay):
        if not np.isfinite([speed_scale, delay]).all() or speed_scale <= 0:
            raise ValueError("Positive speed scale and finite parry delay required")
        speed = min(cfg["ase_nominal_target_speed_rad_s"]*speed_scale, cfg["command_speed_ceiling_rad_s"])
        attack = join_role(library, "attack", speed, cfg["control_period_s"])
        defend = join_role(library, "defend", min(cfg["ase_parry_target_speed_rad_s"], cfg["command_speed_ceiling_rad_s"]), cfg["control_period_s"])
        start = cfg["initial_guard_hold_s"]
        if start+delay < 0:
            raise ValueError("Parry cannot begin before the common reset")
        attack_end = start+attack["duration_s"]
        parry_end = start+delay+defend["duration_s"]
        # Paths themselves are kept on this program instance, not mixed into
        # the serializable experiment timing dictionary.
        self.attack_path, self.parry_path = attack, defend
        return {"attack_start_s": start, "attack_end_s": attack_end,
                "parry_start_s": start+delay, "parry_end_s": parry_end,
                "duration_s": max(attack_end, parry_end)+cfg["final_guard_hold_s"],
                "attack_duration_s": attack["duration_s"], "parry_duration_s": defend["duration_s"],
                "attack_native_start_s": start+attack["native_start_s"],
                "attack_native_end_s": start+attack["native_end_s"],
                "parry_native_start_s": start+delay+defend["native_start_s"],
                "parry_native_end_s": start+delay+defend["native_end_s"],
                "requested_attack_speed_scale": float(speed_scale), "requested_parry_delay_s": float(delay),
                "retiming": "Each joint sample interval is stretched to integer control periods under a target-speed bound; all native samples retained. Source velocity/rhythm is NOT preserved.",
                "attack_target_speed_bound_rad_s": speed,
                "parry_target_speed_bound_rad_s": min(cfg["ase_parry_target_speed_rad_s"], cfg["command_speed_ceiling_rad_s"])}

    def attack_command(self, library, timing, time):
        return sample_timed(self.attack_path, time-timing["attack_start_s"])

    def parry_command(self, library, timing, time, enabled):
        if not enabled:
            return library.guard_defend_q.copy()
        return sample_timed(self.parry_path, time-timing["parry_start_s"])


def runtime_config(overrides=None):
    cfg = json.loads((ROOT/"config/manual_duel.json").read_text())
    cfg.update({"status": "ASE_full_clip_controlled_geometry_experiment_not_RL_or_hardware_validation",
                "ase_nominal_target_speed_rad_s": .35,
                "ase_parry_target_speed_rad_s": .35,
                "hypothetical_joint_limit_force_ceiling_nm": .1,
                "nominal_parry_delay_s": 0.,
                "notes": ["Verified ASE Stab+ParryUp full clips; authored guard connections, no full-clip reverse.",
                          "All native sample geometry retained; sample-wise clock does not preserve source rhythm.",
                          "Same 7cm scoring radius, original robot/tool sizes/gains/debug motor caps and hypothetical +/-pi ranges as manual tests.",
                          "These are scripted controlled motion tests, not a trained imitation policy or self-play."]})
    if overrides:
        cfg.update(copy.deepcopy(overrides))
    return cfg


def preflight_ok(library):
    for key in ("stab", "parry_up"):
        report = library.reports[key]
        if not report["candidate_passes_single_arm_screen"]:
            return False
        guard = report["guard_static_screen"]
        if not guard["strict_static_contacts_free"] or guard["static_gravity_exceeded_indices"]:
            return False
    for key in ("attack_in", "attack_out", "parry_in", "parry_out"):
        if not library.reports[key]["strict_static_contacts_free"] or library.reports[key]["static_gravity_exceeded_indices"]:
            return False
    return True


def evaluate_ase_pair(baseline, defended, cfg):
    comparison = evaluate_pair(baseline, defended, cfg)
    first = defended["first_native_clip_blade_contact_time_s"]
    native_contact_first = (first is not None and first == defended["first_positive_blade_contact_time_s"])
    native_before_hit = (first is not None and baseline["first_target_hit_time_s"] is not None
                         and first < baseline["first_target_hit_time_s"])
    comparison["successful_coupled_parry_regardless_of_motion_source"] = comparison["successful_physical_parry"]
    comparison["baseline_hit_in_native_stab"] = baseline["first_target_hit_during_native_attack"]
    comparison["first_blade_contact_in_both_native_clips"] = native_contact_first
    comparison["successful_physical_parry"] = bool(
        comparison["successful_physical_parry"] and baseline["first_target_hit_during_native_attack"]
        and native_contact_first and native_before_hit
        and defended["native_clip_overlap_blade_impulse_ns"] >= cfg["minimum_blade_contact_impulse_ns"])
    return comparison


def run_experiment(output, *, geometry=None, cfg=None, cases=None, source_dir=None, allow_failed_preflight=False):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    cfg = runtime_config() if cfg is None else copy.deepcopy(cfg)
    library = build_ase_library(geometry, source_dir=source_dir)
    scene = build_duel_scene(library.config["base_distance_m"])
    fixture = scene.config["virtual_targets"]["a2"]
    if not np.allclose(fixture["center_world_m"], library.config["target_world_m"], rtol=0, atol=1e-12) or not np.isclose(fixture["radius_m"], cfg["virtual_target_radius_m"], rtol=0, atol=1e-12):
        raise ValueError("Scoring target and reference fixture must remain identical")
    scene.model.opt.timestep = scene.spec.option.timestep = cfg["physics_timestep_s"]
    (output/"scene.xml").write_text(scene.spec.to_xml())
    (output/"runtime_config.json").write_text(json.dumps(cfg, indent=2)+"\n")
    (output/"geometry_config.json").write_text(json.dumps(library.config, indent=2)+"\n")
    np.savez_compressed(output/"ase_paths.npz", attack_q=library.attack_q, attack_phase=library.attack_phase,
                        parry_q=library.parry_q, parry_phase=library.parry_phase,
                        attack_source_time_s=library.attack_source_time_s, parry_source_time_s=library.parry_source_time_s,
                        guard_attack_q=library.guard_attack_q, guard_defend_q=library.guard_defend_q,
                        **{key+"_q": value["qpos"] for key, value in library.transitions.items()})
    report = {"status": "running", "scope": "paired_native_physics_of_complete_ASE_geometric_paths",
              "runtime_config": cfg, "manual_geometry_config": library.config, "ase_geometry_config": library.config,
              "provenance": library.provenance, "single_arm_screen": library.reports,
              "preflight_pass": preflight_ok(library), "scene_config": scene.config, "source_hashes": scene.source_hashes,
              "implementation_sha256": {p: digest(ROOT/p) for p in ["sim/ase_duel.py", "sim/ase_skills.py", "sim/manual_duel.py", "sim/duel_scene.py"]},
              "software_versions": {"mujoco": mujoco.__version__, "numpy": np.__version__}, "pairs": []}
    summary = output/"summary.json"
    def save():
        summary.write_text(json.dumps(_finite_json(report), indent=2, allow_nan=False)+"\n")
    save()
    if not report["preflight_pass"] and not allow_failed_preflight:
        report["status"] = "failed_single_arm_preflight_no_pose_frames_removed"
        save()
        return report
    program = AseClipProgram()
    cases = [("nominal", 1., cfg["nominal_parry_delay_s"], False)] if cases is None else cases
    for label, speed, delay, mirrored in cases:
        baseline = run_trial(scene, library, cfg, parry_enabled=False, speed_scale=speed, delay=delay,
                             mirrored=mirrored, output=output/label/"no_defence", program=program)
        defended = run_trial(scene, library, cfg, parry_enabled=True, speed_scale=speed, delay=delay,
                             mirrored=mirrored, output=output/label/"parry", program=program)
        comparison = evaluate_ase_pair(baseline, defended, cfg)
        report["pairs"].append({"label": label, "comparison": comparison, "no_defence": baseline, "parry": defended})
        save()
        print(json.dumps({"case": label, **comparison,
                          "baseline_failure": baseline["failure"], "parry_failure": defended["failure"],
                          "no_defence_min_distance_m": baseline["minimum_tip_target_center_distance_m"],
                          "parry_min_distance_m": defended["minimum_tip_target_center_distance_m"],
                          "blade_force_n": defended["blade_contact_peak_force_n"]}), flush=True)
    report["status"] = "completed_controlled_ASE_tests_not_trained_or_hardware_validated"
    report["successful_pairs"] = sum(p["comparison"]["successful_physical_parry"] for p in report["pairs"])
    report["tested_pairs"] = len(report["pairs"])
    save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--geometry", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--delays", type=float, nargs="+", help="Explicit whole-clip start delays, seconds; failures are retained")
    parser.add_argument("--speed-scales", type=float, nargs="+", default=[1.])
    args = parser.parse_args()
    cfg = runtime_config(json.loads(args.config.read_text()) if args.config else None)
    cases = ([(f"speed_{speed:.3f}_delay_{delay:+.3f}", speed, delay, False)
              for speed in args.speed_scales for delay in args.delays] if args.delays else None)
    report = run_experiment(args.output, geometry=json.loads(args.geometry.read_text()) if args.geometry else None,
                            cfg=cfg, source_dir=args.source_dir, cases=cases)
    print(args.output.resolve()/"summary.json", flush=True)
    return 0 if report.get("successful_pairs", 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
