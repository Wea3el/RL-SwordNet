"""Reproduce current rule contracts without changing the game or physics.

Runs existing rule unit tests plus explicit synthetic geometry/settlement
cases. Synthetic cylinders are not a robot trajectory or a native duel.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import io
import itertools
import json
from pathlib import Path
import sys
import traceback
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from sim.fencing_rules import BladeSection, FirstTouchScorer, RULES_VERSION
from sim.foul_attribution import ContactEvidence, attribute_contact, settle_tick
from sim.training_contacts import settle_training_tick, TrainingEnvironmentError, DEPLOYED_RULE_VERSION


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def blade(x, name):
    return BladeSection(np.array([x, 0., -.2]), np.array([x, 0., .2]), .01, name)


def moving_frame(now, entry_times, speed):
    # Target r=.06 plus cylinder r=.01: x=-.07 is exact side tangency.
    return {slot: (blade(-.07+speed*(now-entry), slot+"_synthetic_blade"),)
            for slot, entry in zip(("a1", "a2"), entry_times)}


def linear_sweep(entry_times, speed, dt, duration):
    scorer = FirstTouchScorer({"a1": [0., 0., 0.], "a2": [0., 0., 0.]}, .06)
    before = moving_frame(0., entry_times, speed)
    scorer.update(before, before, 0., 0.)
    count = round(duration/dt)
    steps = []
    for index in range(count):
        t0, t1 = index*dt, (index+1)*dt
        after = moving_frame(t1, entry_times, speed)
        entries = scorer.update(before, after, t0, t1)
        if entries:
            steps.append({"interval_s": [t0, t1], "entries": entries})
        before = after
    return {"speed_m_s": speed, "timestep_s": dt, "duration_s": duration,
        "analytic_tangent_entry_times_s": entry_times,
        "initial_overlap_events": scorer.initial_overlap_events,
        "failure": scorer.failure, "entry_steps": steps,
        "all_entries": scorer.all_entries, "round_result": scorer.round_result,
        "geometry_note": "Vertical .4m cylinder translating linearly through a .06m sphere. Actual strict-interior witness follows tangency by distance epsilon/speed. No rigid-body simulation."}


def evidence(*, identifier="forbidden", a=.2, b=0., prior=False, reliable=True,
             category="illegal_cross_robot_nonblade_contact"):
    return ContactEvidence(17, identifier, category, "a1", "a2", (a,0.,0.), (b,0.,0.),
                           (1.,0.,0.), reliable, prior)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "outputs/rules_review_v1/event_contract"))
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    test_names = ("test_fencing_rules.py", "test_foul_attribution.py",
                  "test_training_contacts.py", "test_training_contact_episodes.py")
    inputs = [Path(__file__).resolve(), *(ROOT / "tests" / name for name in test_names),
              ROOT / "sim/fencing_rules.py", ROOT / "sim/foul_attribution.py",
              ROOT / "sim/training_contacts.py"]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in inputs}
    summary = {"schema_version": 1, "status": "running", "rules_version": RULES_VERSION,
        "deployed_contact_rule_version": DEPLOYED_RULE_VERSION, "source_sha256": hashes,
        "scope": "Existing unit tests and synthetic scorer/settlement contract probes; not native robot dynamics, skill validation, self-play or new rule selection",
        "production_rules_changed": False, "admitted_to_training": False,
        "synthetic_cases": []}

    def save():
        (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")

    def case(name, function, check):
        row = {"name": name}
        try:
            value = function()
            row.update(result=value, passed=bool(check(value)))
        except Exception as exc:
            row.update(passed=False, unexpected_error={"type":type(exc).__name__, "message":str(exc),
                                                      "traceback":traceback.format_exc()})
        summary["synthetic_cases"].append(row)
        save()

    save()
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    for name in test_names:
        suite.addTests(loader.discover(str(ROOT / "tests"), pattern=name))
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    (output / "unit_tests.log").write_text(stream.getvalue())
    summary["existing_unit_tests"] = {"files":list(test_names), "tests_run":result.testsRun,
        "failures":[{"test":str(t), "traceback":tb} for t,tb in result.failures],
        "errors":[{"test":str(t), "traceback":tb} for t,tb in result.errors],
        "skipped":[{"test":str(t), "reason":reason} for t,reason in result.skipped],
        "successful":result.wasSuccessful(), "log":"unit_tests.log"}
    save()
    # One active cylinder, one remote cylinder whose entry is beyond the run.
    for label, speed in (("very_slow_0.1mm_s", 1e-4), ("normal_1m_s", 1.)):
        case("speed_qualification_"+label,
             lambda speed=speed: linear_sweep([.5, 1000.], speed, .01, 1.),
             lambda r: r["failure"] is None and not r["initial_overlap_events"]
             and r["round_result"]["winner"] == "a1" and len(r["all_entries"]) == 1)
    # Both phases preserve 0.7ms real entry separation. The second phase also
    # demonstrates that aligning the step-grid differently changes the bin.
    for label, times, expected in (
        ("phase_a", [.0008, .0015], ("double_touch", "win", "win")),
        ("phase_b", [.0012, .0019], ("double_touch", "double_touch", "win"))):
        for dt, status in zip((.002, .001, .0005), expected):
            case(f"same_continuous_motion_{label}_dt_{dt:g}",
                 lambda times=times, dt=dt: linear_sweep(times, 1., dt, .004),
                 lambda r, status=status: r["failure"] is None and not r["initial_overlap_events"]
                 and r["round_result"]["status"] == status and len(r["all_entries"]) == 2
                 and (status == "double_touch" or r["round_result"]["winner"] == "a1"))

    def settle_example(kind):
        if kind == "no_contact":
            return settle_training_tick(17)
        if kind == "single_hit":
            return settle_training_tick(17, hit_agents=["a1"])
        if kind == "double_hit":
            return settle_training_tick(17, hit_agents=["a1","a2"])
        changes = {"same_tick_foul_and_hit":{}, "double_foul_and_hit":{"b":-.2},
                   "prior_legal_contact_ambiguous":{"prior":True},
                   "invalid_contact_evidence":{"reliable":False}}
        if kind == "numeric_fault_and_hit":
            kwargs = {"numerical_faults":["excessive_proxy_penetration"], "hit_agents":["a1"]}
        else:
            kwargs = {"contacts":[attribute_contact(evidence(**changes[kind]))], "hit_agents":["a1"]}
        try:
            return settle_training_tick(17, **kwargs)
        except TrainingEnvironmentError as exc:
            return {"exception":"TrainingEnvironmentError", "details":exc.details}

    expectations = {
        "no_contact": ("ongoing", {"a1":0.,"a2":0.}, None),
        "single_hit": ("target_win", {"a1":1.,"a2":-1.}, "a1"),
        "double_hit": ("double_touch", {"a1":0.,"a2":0.}, None),
        "same_tick_foul_and_hit": ("foul_win", {"a1":-1.,"a2":1.}, "a2"),
        "double_foul_and_hit": ("double_foul", {"a1":-1.,"a2":-1.}, None),
        "prior_legal_contact_ambiguous": ("ambiguous_foul", {"a1":-1.,"a2":-1.}, None),
        "invalid_contact_evidence": ("evidence_error", None, None),
        "numeric_fault_and_hit": ("numerical_fault", None, None)}
    for name, expected in expectations.items():
        def check(r, expected=expected):
            value = r.get("details", r)
            return (value["status"], value["rl_reward"], value["match_winner"]) == expected
        case("settlement_"+name, lambda name=name:settle_example(name), check)

    def order_invariance():
        a = attribute_contact(evidence(identifier="foul_a"))
        b = attribute_contact(evidence(identifier="foul_b", a=0., b=-.2))
        neutral = attribute_contact(evidence(identifier="legal", category="cross_blade_guard"))
        outputs = [settle_tick(17, entries, hit_agents=["a1"]) for entries in itertools.permutations((a,b,neutral,a))]
        return {"input_permutations":len(outputs), "outputs_identical":all(r == outputs[0] for r in outputs),
                "representative":outputs[0]}
    case("duplicate_contact_input_order_invariance", order_invariance, lambda r:r["outputs_identical"])

    def mirror_invariance():
        base = evidence(a=.2, b=.03)
        yaw = np.diag([-1.,-1.,1.])
        mirrored = replace(base, velocity_a_world_m_s=tuple(yaw@base.velocity_a_world_m_s),
            velocity_b_world_m_s=tuple(yaw@base.velocity_b_world_m_s),
            normal_a_to_b_world=tuple(yaw@base.normal_a_to_b_world))
        swap = replace(base, owner_a="a2", owner_b="a1",
            velocity_a_world_m_s=base.velocity_b_world_m_s, velocity_b_world_m_s=base.velocity_a_world_m_s,
            normal_a_to_b_world=tuple(-np.asarray(base.normal_a_to_b_world)))
        rows = [attribute_contact(item) for item in (base,mirrored,swap)]
        return {"verdicts":rows, "status_and_responsible_owners_identical":all(
            (r["status"],r["responsible_owners"]) == (rows[0]["status"],rows[0]["responsible_owners"]) for r in rows)}
    case("yaw_and_geom_order_invariance", mirror_invariance, lambda r:r["status_and_responsible_owners_identical"])

    summary["synthetic_case_count"] = len(summary["synthetic_cases"])
    summary["synthetic_pass_count"] = sum(row["passed"] for row in summary["synthetic_cases"])
    summary["source_hashes_unchanged"] = all(digest(ROOT/p) == h for p,h in hashes.items())
    summary["status"] = "contract_audit_passed" if result.wasSuccessful() and all(
        row["passed"] for row in summary["synthetic_cases"]) and summary["source_hashes_unchanged"] else "contract_audit_failed"
    summary["conclusions"] = [
        "Current scorer has no minimum hit-speed gate: both tested speeds score under the same geometric rule.",
        "Same physical .7ms entry separation can produce double-touch or first-hit win when timestep/bin alignment changes.",
        "Same-tick foul overrides hit; mutual foul is both -1, not a zero-reward draw.",
        "Invalid physical/evidence transitions raise without a reward; no new temporal window or rule was selected."]
    save()
    print(json.dumps({k:summary[k] for k in ("status","existing_unit_tests","synthetic_case_count",
        "synthetic_pass_count","source_hashes_unchanged")}, indent=2))
    return 0 if summary["status"] == "contract_audit_passed" else 1


if __name__ == "__main__":
    sys.exit(main())
