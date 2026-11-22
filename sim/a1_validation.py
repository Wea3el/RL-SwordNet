"""Reproducible native A1 execution/transition and paired-defence development.

The learned A1 chooser does not exist here. Both scene slots use independent
A1 executors; scheduled requests are controls, not self-play training. A
completed single motion is NOT a parry certification. Failed candidates and
post-first-touch shadow entries stay in their original evidence directories.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .a1_control import A1Controller, A1ScriptedPolicy, A1SequencePolicy
from .assembly import ROOT
from .duel_scene import build_duel_scene
from .skill_validation import V2_CONFIG, compare_defence, freeze_experiment, run_trial, save_json


IMPLEMENTATIONS = ("sim/a1_skills.py", "sim/a1_control.py", "sim/a1_validation.py")
ATTACK_CANDIDATES = ("stab", "slash_left")
PARRY_CANDIDATES = ("parry_left", "parry_right")


def execution_gate(result, expected_by_slot):
    """Require actual completion of every requested finite skill, not elapsed video."""
    checks = {}
    for slot, expected in expected_by_slot.items():
        requested = [e["action_id"] for e in result["action_events"][slot]]
        completed = [e["action_id"] for e in result["controller_diagnostics"][slot]["completions"]]
        checks[slot] = {"expected_actions": list(expected), "accepted_actions": requested,
                       "completed_actions": completed,
                       "all_requests_completed_in_order": requested == completed == list(expected)}
    full_window = abs(result["completed_time_s"]-result["requested_duration_s"]) <= result["runtime_config"]["physics_timestep_s"]+1e-7
    return {"native_execution_pass": bool(result["status"] == "completed" and full_window
                and all(c["all_requests_completed_in_order"] for c in checks.values())),
            "full_window_completed": full_window, "slots": checks,
            "effective_attack_or_defence_certification": False}


def defence_gate(baseline, defended, attacker, cfg=None):
    """Physical evidence AND fully executed expected skills are both required."""
    result = compare_defence(baseline, defended, attacker=attacker, cfg=cfg)
    execution = all(r.get("a1_execution_gate", {}).get("native_execution_pass") is True
                    for r in (baseline, defended))
    result["both_expected_execution_gates_passed"] = execution
    result["successful_physical_defence"] = bool(result["successful_physical_defence"] and execution)
    return result


def _digest(paths):
    return {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}


def validate_a1(library, output, *, transitions=True, opportunities=False,
                parry_starts_s=(0., .75, 1.75, 2.75, 3.75, 4.75), cfg=None,
                progress=print):
    """Development gate, preserving original targets/caps/collision eligibility.

    Opportunity sweep uses a fresh matched Guard baseline with the SAME full
    window and attack command hashes. Source skill names are only candidates.
    No finite bank is admitted here: perturbations, direction expansion,
    anti-degeneracy, policy/environment and independent held-out checks remain.
    """
    runtime = dict(V2_CONFIG, decision_period_s=.02)
    runtime.update(cfg or {})
    if runtime["rules_version"] != "whole_blade_first_touch_v2":
        raise ValueError("A1 validation requires ungated whole-blade v2 scoring")
    starts = tuple(float(t) for t in parry_starts_s)
    if not starts or not np.isfinite(starts).all() or min(starts) < 0:
        raise ValueError("Finite nonnegative parry start samples required")
    scene = build_duel_scene(runtime["base_distance_m"])
    implementations = IMPLEMENTATIONS
    if library.provenance.get("adaptation_recipe"):
        implementations += ("sim/a1_adaptation.py",)
    if library.provenance.get("authored_defence_recipes"):
        implementations += ("sim/a1_defence.py",)
    folder = freeze_experiment(output, library, scene, cfg=runtime,
        implementation_paths=implementations,
        metadata={"policy_family_in_both_physical_slots": "A1",
                  "status": "candidate_native_development_not_RL_or_admission",
                  "guard_initialization_only": True,
                  "reset_to_Guard_travel_from_hardware_home_validated": False,
                  "reference_sampling": "use exact bank sampler, not linear interpolation of shared_paths.npz"})
    initial_hashes = json.loads((folder/"experiment.json").read_text(encoding="utf-8"))["implementation_sha256"]
    from .a1_skills import export_a1_library
    export_a1_library(library, folder/"library")
    summary = {"schema_version": 1, "status": "running", "policy_family": "A1",
        "runtime_config": runtime, "admitted_to_training": False, "may_start_RL": False,
        "trials": {}, "paired_defence": {}, "development_start_samples_s": starts,
        "limitations": ["No learned selector, PPO or self-play is run",
            "Guard-reset initialization only; physical setup/reset travel remains separate",
            "A1 candidates occupy either slot; a2 slot is NOT the A2 continuous policy",
            "Single successful attack or contact does not certify a bank or direction coverage",
            "Speed is a diagnostic, not identified motor certification",
            "No target/cap/mask change or post-initialization qpos reset"]}
    save_json(folder/"summary.json", summary)
    if not library.reports["library"]["all_skills_pass_geometry"]:
        summary.update(status="geometry_rejected_before_native")
        save_json(folder/"summary.json", summary)
        return summary
    initial, final = runtime["initial_hold_s"], runtime["final_hold_s"]
    timeout = A1Controller(library).settle_timeout_s
    maximum = max(path.duration_s for path in library.skills.values())
    window = initial+maximum+timeout+final
    slots = ("a1", "a2")

    def trial(name, policies, duration, expected):
        result = run_trial(scene, library, policies, duration, cfg=runtime,
                           output=folder/name, controller_factory=A1Controller)
        # Replace historical manual-library wording; runner mechanics unchanged.
        result["limitations"] = "A1 finite candidate execution with recorded source/authored provenance; scheduled diagnostic requests, not learned A1/A2 or certified hardware/defence"
        gate = execution_gate(result, expected)
        result["a1_execution_gate"] = gate
        save_json(folder/name/"summary.json", result)
        summary["trials"][name] = {"status": result["status"], "failure": result["failure"],
            "completed_time_s": result["completed_time_s"], "round_result": result["round_result"],
            "all_shadow_entries": result["hits"], "execution_gate": gate,
            "minimum_blade_target_clearance_m": result["minimum_blade_target_clearance_m"],
            "peak_tracking_error_rad": max(result["peak_joint_tracking_error_rad"]),
            "peak_actual_speed_rad_s": max(result["peak_actual_joint_speed_rad_s"])}
        save_json(folder/"summary.json", summary)
        if progress:
            progress(f"{name}: {result['status']}; executed={gate['native_execution_pass']}; "
                     f"hits={len(result['hits'])}; failure={result['failure']}", flush=True)
        return result

    def scripted(events=None):
        events = events or {}
        return {slot: A1ScriptedPolicy(library, events.get(slot, ())) for slot in slots}

    trial("guard_hold", scripted(), window, {slot: [] for slot in slots})
    for slot in slots:
        for name, path in library.skills.items():
            if path.action_id == 0:
                continue
            trial(f"{slot}_{name}_vs_guard", scripted({slot: [(initial, path.action_id)]}),
                  window, {s: [path.action_id] if s == slot else [] for s in slots})
    if transitions:
        # Finite diagnostics only: all candidate types, then re-entry into Stab.
        actions = [p.action_id for p in library.skills.values() if p.action_id]+[library.skills["stab"].action_id]
        gap = .1
        duration = initial+sum(library.skills[next(n for n,p in library.skills.items() if p.action_id == a)].duration_s
                               +timeout+gap+runtime["decision_period_s"] for a in actions)+final
        for slot in slots:
            policies = scripted()
            policies[slot] = A1SequencePolicy(library, actions, initial, gap)
            trial(f"{slot}_transition_chain", policies, duration,
                  {s: actions if s == slot else [] for s in slots})
    if opportunities:
        pair_window = max(initial, max(starts))+maximum+timeout+final
        for slot in slots:
            defender = slots[1] if slot == slots[0] else slots[0]
            for name in ATTACK_CANDIDATES:
                attack = library.skills[name].action_id
                base = trial(f"{slot}_{name}_paired_guard", scripted({slot: [(initial, attack)]}),
                             pair_window, {slot: [attack], defender: []})
                if (not base["a1_execution_gate"]["native_execution_pass"]
                        or not any(h["agent"] == slot for h in base["hits"])):
                    summary["paired_defence"][f"{slot}_{name}"] = {"status": "no_valid_attack_baseline",
                        "reason": "Cannot certify parries against an attack that misses or is invalid"}
                    continue
                for parry in PARRY_CANDIDATES:
                    action = library.skills[parry].action_id
                    for index, start in enumerate(starts):
                        key = f"{slot}_{name}_{parry}_start_{index}"
                        run = trial(key, scripted({slot: [(initial, attack)], defender: [(start, action)]}),
                                    pair_window, {slot: [attack], defender: [action]})
                        comparison = defence_gate(base, run, attacker=slot, cfg=runtime)
                        comparison.update(requested_parry_start_s=start,
                            source_skill_name_is_not_direction_certification=True)
                        summary["paired_defence"][key] = comparison
                        save_json(folder/"summary.json", summary)
    summary["implementation_hashes_unchanged"] = initial_hashes == _digest(tuple(initial_hashes))
    summary["status"] = "development_complete_not_admitted" if summary["implementation_hashes_unchanged"] else "implementation_changed_during_run"
    summary["all_scheduled_execution_gates_passed"] = all(v["execution_gate"]["native_execution_pass"] for v in summary["trials"].values())
    summary["next_gate"] = "Repair invalid/missing attack and defence opportunities; validate negatives, transitions and perturbations before exposing a training bank"
    save_json(folder/"summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--pilot", type=Path, default=ROOT/"outputs/a1_ase_pilot_labeled")
    inputs.add_argument("--bank", type=Path, help="Load a closed numeric bank, including explicitly authored adaptations")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bank-config", type=Path)
    parser.add_argument("--source-time-scale", type=float)
    parser.add_argument("--no-transitions", action="store_true")
    parser.add_argument("--opportunities", action="store_true")
    parser.add_argument("--parry-starts", type=float, nargs="+", default=[0., .75, 1.75, 2.75, 3.75, 4.75])
    args = parser.parse_args()
    from .a1_skills import build_a1_library, load_a1_library
    if args.bank is not None:
        if args.bank_config is not None or args.source_time_scale is not None:
            parser.error("A frozen --bank has its own timing/config; do not override it")
        library = load_a1_library(args.bank)
    else:
        config = {} if args.bank_config is None else json.loads(args.bank_config.read_text(encoding="utf-8"))
        if args.source_time_scale is not None:
            config["source_time_scale"] = args.source_time_scale
        library = build_a1_library(args.pilot, config=config)
    result = validate_a1(library, args.output, transitions=not args.no_transitions,
                         opportunities=args.opportunities, parry_starts_s=args.parry_starts)
    print(f"{args.output.resolve()}/summary.json\n{result['status']}; may_start_RL=False")


if __name__ == "__main__":
    main()
