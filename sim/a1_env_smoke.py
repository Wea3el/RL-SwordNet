"""One scripted native training-environment smoke, never PPO or self-play."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from .a1_control import A1ScriptedPolicy
from .a1_env import A1DuelEnv
from .a1_runtime import load_runtime_profile
from .assembly import ROOT
from .skill_validation import freeze_experiment, save_json
from .training_contacts import TrainingEnvironmentError


IMPLEMENTATIONS = ("sim/a1_skills.py", "sim/a1_control.py", "sim/a1_runtime.py",
    "sim/a1_env.py", "sim/a1_env_smoke.py", "sim/training_contacts.py", "sim/foul_attribution.py")


def smoke(bank, profile_path, output):
    profile = load_runtime_profile(profile_path)
    env = A1DuelEnv(bank, profile, max_duration_s=16., development_mode=True)
    observations, _ = env.reset()
    folder = freeze_experiment(output, env.library, env.scene, cfg=env.cfg,
        implementation_paths=(*IMPLEMENTATIONS, str(Path(profile_path).resolve().relative_to(ROOT))),
        metadata={"scope":"one Stab vs Guard scripted environment smoke, not training or selfplay",
                  "development_mode":True,"bank_admission":False})
    hashes = {path:hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in IMPLEMENTATIONS}
    policies = {"a1":A1ScriptedPolicy(env.library,[(.75,1)]),
                "a2":A1ScriptedPolicy(env.library)}
    error = None
    try:
        while True:
            # Both choices use the same completed-state observation snapshot.
            actions = {slot:policy.decide(observations[slot]) for slot,policy in policies.items()}
            observations, _, terminated, truncated, _ = env.step(actions)
            if terminated or truncated:break
    except TrainingEnvironmentError as failure:
        error = failure.details
    env.export(folder/"episode")
    result = env.result()
    summary = {"status":result["status"], "failure":error,
        "round_result":result["round_result"], "completed_time_s":result["completed_time_s"],
        "implementation_hashes_unchanged":all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items()),
        "expected_a1_native_stab_target_win":bool(result["round_result"] and
            result["round_result"]["status"]=="target_win" and result["round_result"]["match_winner"]=="a1"),
        "native_physics_tick_termination_not_full_shadow_validation":True,
        "admitted_to_training":False, "training_or_selfplay_performed":False}
    save_json(folder/"summary.json",summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank",required=True)
    parser.add_argument("--runtime-profile",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    result=smoke(args.bank,args.runtime_profile,args.output)
    print(result,flush=True)
    return 0 if result["expected_a1_native_stab_target_win"] and result["implementation_hashes_unchanged"] else 1


if __name__=="__main__":raise SystemExit(main())
