"""Reproducible continuous rule/random-policy diagnostics, BEFORE RL.

These controllers share the identical action and current-observation interface.
Scripted paired skill evidence does not imply continuous-policy success. No
learning, reward shaping, winner settlement or opponent checkpoints occur here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .assembly import ROOT
from .duel_scene import build_duel_scene
from .shared_control import ConstantPolicy, RandomPolicy, ReactiveCounterPolicy
from .shared_skills import build_shared_library
from .skill_validation import DEFAULT_CONFIG, freeze_experiment, run_trial, save_json


def policy_cases():
    reactive = lambda seed, wait: (lambda: ReactiveCounterPolicy(seed=seed, attack_wait_s=wait))
    repeat_attack, repeat_parry = lambda: ConstantPolicy(1), lambda: ConstantPolicy(2)
    return {
        "counter_vs_repeat_attack": ({"a1": repeat_attack, "a2": reactive(1,(30.,30.))},
            {"a1": "always Attack when ready", "a2": {"name":"ReactiveCounterPolicy", "seed":1,"attack_wait_s":[30.,30.]}}),
        "mirrored_counter": ({"a2": repeat_attack, "a1": reactive(1,(30.,30.))},
            {"a2": "always Attack when ready", "a1": {"name":"ReactiveCounterPolicy", "seed":1,"attack_wait_s":[30.,30.]}}),
        "repeat_attack_vs_always_parry": ({"a1": repeat_attack, "a2": repeat_parry},
            {"a1":"always Attack when ready", "a2":"always Parry when ready"}),
        "reactive_vs_random": ({"a1": reactive(2,(2.,6.)), "a2": lambda: RandomPolicy(17)},
            {"a1":{"name":"ReactiveCounterPolicy","seed":2,"attack_wait_s":[2.,6.]},"a2":{"name":"RandomPolicy","seed":17}}),
        "reactive_vs_reactive": ({"a1": reactive(3,(.5,2.)), "a2": reactive(4,(4.,8.))},
            {"a1":{"name":"ReactiveCounterPolicy","seed":3,"attack_wait_s":[.5,2.]},"a2":{"name":"ReactiveCounterPolicy","seed":4,"attack_wait_s":[4.,8.]}}),
    }


def run_policy_suite(output, library_config, *, cases=None, duration_s=60.):
    available = policy_cases()
    cases = list(available) if cases is None else list(cases)
    if not cases or len(set(cases)) != len(cases) or any(k not in available for k in cases):
        raise ValueError("Select unique known policy cases")
    library = build_shared_library(library_config)
    scene = build_duel_scene(DEFAULT_CONFIG["base_distance_m"])
    descriptions = {k: available[k][1] for k in cases}
    folder = freeze_experiment(output, library, scene, metadata={
        "scope":"continuous native physics diagnostics before RL", "policies":descriptions,
        "shared_rule_parameters":{"preparation_height_band_m":[.22,.27],"threat_horizon_s":4.,
             "counter":"own Parry finished AND visible axial opponent tip distance>.21m with retreat/rising velocity"},
        "driver_sha256":hashlib.sha256((ROOT/"sim/policy_validation.py").read_bytes()).hexdigest()})
    report = {"status":"running", "runtime_config":DEFAULT_CONFIG, "policy_descriptions":descriptions,
              "duration_s":duration_s, "trials":{}, "may_start_RL":False}
    save_json(folder/"summary.json", report)
    if not library.reports["library"]["all_skills_pass_geometry"]:
        report.update(status="failed_geometry_preflight", invalid_cases=["geometry_preflight"])
        save_json(folder/"summary.json", report)
        return report
    for name in cases:
        policies = {k: factory() for k,factory in available[name][0].items()}
        result = run_trial(scene, library, policies, duration_s, output=folder/name)
        report["trials"][name] = result
        save_json(folder/"summary.json", report)
        print(name, result["status"], result["failure"], result["hits"], flush=True)
    report["status"] = "completed_diagnostics_not_training_or_statistical_policy_comparison"
    report["invalid_cases"] = [k for k,r in report["trials"].items() if r["status"] != "completed"]
    report["remaining_gate"] = "Gym observation/action contract, fair episode termination/reward and multi-seed held-out evaluation before RL"
    save_json(folder/"summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library-config", type=Path, required=True)
    parser.add_argument("--duration-s", type=float, default=60.)
    parser.add_argument("--cases", nargs="+", choices=list(policy_cases()))
    args = parser.parse_args()
    report = run_policy_suite(args.output, json.loads(args.library_config.read_text()), cases=args.cases, duration_s=args.duration_s)
    return 2 if report.get("invalid_cases") else 0


if __name__ == "__main__":
    raise SystemExit(main())
