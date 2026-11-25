"""Offline candidate command-governor contract; never imported by production.

No MuJoCo state, PD gains, force caps, physics or scoring is changed here.
The numeric defaults below belong only to the audit demonstration. A1 would
need the same command component and new physical validation before deployment.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.joint_constraints import load_joint_constraints, model_joint_ranges_rad

STATUS = "candidate_not_deployed"
# Public compatibility name: these are CAD-coordinate bounds, after applying
# the confirmed hardware zero/sign map, not the raw hardware-angle table.
DEMO_BOUNDS_DEG = tuple(map(tuple, np.rad2deg(model_joint_ranges_rad(load_joint_constraints()))))


def _real(value, shape, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu" or raw.shape != shape:
        raise ValueError(f"{name} must be a real array of shape {shape}")
    result = np.array(raw, dtype=float, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _parameter(value, name, *, allow_zero=False):
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu" or raw.shape not in ((), (7,)):
        raise ValueError(f"{name} must be a real scalar or seven-vector")
    result = np.broadcast_to(raw, (7,)).astype(float, copy=True)
    if not np.isfinite(result).all() or np.any(result < 0 if allow_zero else result <= 0):
        raise ValueError(f"{name} has invalid magnitude")
    return result


def _readonly(value):
    result = np.array(value, copy=True)
    result.flags.writeable = False
    return result


def bounded_command_step(q, q_ref, raw_action, bounds_rad, *, control_period_s,
                         max_reference_speed_rad_s, tracking_leash_rad, joint_margin_rad):
    """Pure seven-joint command step with directional tracking anti-windup.

    All inputs are copied. ``q`` is measured physical position; invalid physical
    state is rejected, NEVER clipped. ``q_ref`` is the previous accepted target,
    not a latent unclipped accumulator. Raw action is retained for policy data.
    The zero action holds the reference, even if contact moved q outside its
    tracking leash. In that situation only outward accumulation is blocked;
    inward commands are allowed at the usual target-rate bound, not teleported.
    This limits target speed, NOT actual joint velocity or contact penetration.
    """
    actual = _real(q, (7,), "q")
    reference = _real(q_ref, (7,), "q_ref")
    action = _real(raw_action, (7,), "raw_action")
    bounds = _real(bounds_rad, (7,2), "bounds_rad")
    if np.any(bounds[:,0] >= bounds[:,1]):
        raise ValueError("Joint bounds must be ordered")
    if np.any(actual < bounds[:,0]) or np.any(actual > bounds[:,1]):
        raise ValueError("Actual q is outside physical bounds; do not silently repair it")
    if np.any(action < -1.) or np.any(action > 1.):
        raise ValueError("Raw normalized action must already be in [-1,1]")
    tick = _real(control_period_s, (), "control_period_s")
    if tick <= 0:
        raise ValueError("Positive control period required")
    speed = _parameter(max_reference_speed_rad_s, "max_reference_speed_rad_s")
    leash = _parameter(tracking_leash_rad, "tracking_leash_rad")
    margin = _parameter(joint_margin_rad, "joint_margin_rad", allow_zero=True)
    with np.errstate(over="ignore", invalid="ignore"):
        interior_low, interior_high = bounds[:,0]+margin, bounds[:,1]-margin
        maximum_delta = speed*float(tick)
        requested_delta = action*maximum_delta
        desired = reference+requested_delta
        # Each interval contains reference. A displaced physical q therefore
        # cannot cause an empty intersection or a forced target snap-back.
        low = np.maximum.reduce((interior_low, reference-maximum_delta,
                                 np.minimum(reference, actual-leash)))
        high = np.minimum.reduce((interior_high, reference+maximum_delta,
                                  np.maximum(reference, actual+leash)))
    if not all(np.isfinite(x).all() for x in (interior_low,interior_high,maximum_delta,
                                             requested_delta,desired,low,high)):
        raise ValueError("Derived command values are not finite")
    if np.any(interior_low >= interior_high):
        raise ValueError("Margin leaves no joint interior")
    if np.any(reference < interior_low) or np.any(reference > interior_high):
        raise ValueError("Previous accepted q_ref must lie in the configured interior")
    if np.any(low > reference) or np.any(high < reference):
        raise ValueError("Governor invariant failed: interval must contain previous reference")
    accepted = np.clip(desired, low, high)
    return {"status": STATUS, "raw_action": _readonly(action),
        "requested_delta_rad": _readonly(requested_delta),
        "accepted_delta_rad": _readonly(accepted-reference),
        "accepted_q_ref_rad": _readonly(accepted),
        "admissible_low_rad": _readonly(low), "admissible_high_rad": _readonly(high),
        "governed_axes": _readonly((desired < low) | (desired > high)),
        "tracking_error_before_rad": _readonly(reference-actual)}


def jsonable(value):
    if isinstance(value, dict):
        return {k:jsonable(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def demonstration():
    constraints = load_joint_constraints()
    bounds = model_joint_ranges_rad(constraints)
    actual = bounds.mean(axis=1)
    reference, naive = actual.copy(), actual.copy()
    action = np.zeros(7); action[0] = 1.
    control, decision, duration = .01, .02, 1.
    speed, leash, margin = 2., .10, np.deg2rad(1.)
    history = []
    # Held action changes only at 20ms boundaries; both 10ms ticks use it.
    for index in range(round(duration/control)):
        update = bounded_command_step(actual, reference, action, bounds,
            control_period_s=control, max_reference_speed_rad_s=speed,
            tracking_leash_rad=leash, joint_margin_rad=margin)
        reference = update["accepted_q_ref_rad"].copy()
        naive += speed*control*action
        history.append({"time_s":(index+1)*control,
            "decision_index":index//round(decision/control),
            "naive_J0_delta_rad":float(naive[0]-actual[0]),
            "bounded_J0_delta_rad":float(reference[0]-actual[0]),
            "raw_action_J0":float(action[0]),
            "accepted_J0_delta_this_tick_rad":float(update["accepted_delta_rad"][0]),
            "J0_governed":bool(update["governed_axes"][0])})
    reverse = bounded_command_step(actual, reference, -action, bounds,
        control_period_s=control, max_reference_speed_rad_s=speed,
        tracking_leash_rad=leash, joint_margin_rad=margin)
    zero = bounded_command_step(actual, reference, np.zeros(7), bounds,
        control_period_s=control, max_reference_speed_rad_s=speed,
        tracking_leash_rad=leash, joint_margin_rad=margin)
    return {"parameters_status":"illustrative audit values, not approved or calibrated motor limits",
        "joint_coordinate_system":"CAD q radians; bounds converted from the current hardware zero/sign map",
        "joint_bounds_cad_rad":bounds.tolist(),
        "hardware_offset_deg":list(constraints.offset_deg),
        "parameters":{"control_period_s":control,"decision_period_s":decision,"duration_s":duration,
            "reference_speed_rad_s":speed,"tracking_leash_rad":leash,"command_margin_deg":1.},
        "physical_model":"q held constant numerically; no MuJoCo or contact dynamics",
        "naive_target_delta_after_1s_rad":float(naive[0]-actual[0]),
        "bounded_target_delta_after_1s_rad":float(reference[0]-actual[0]),
        "reverse_next_target_delta_from_actual_rad":float(reverse["accepted_q_ref_rad"][0]-actual[0]),
        "zero_action_holds_exactly":bool(np.array_equal(zero["accepted_q_ref_rad"],reference)),
        "history":history}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "outputs/scoring_design_v1/a2_command_contract"))
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__).resolve(), ROOT/"tests/test_a2_command_contract.py",
               *sorted((ROOT/"sim").glob("*.py")), *sorted((ROOT/"config").glob("*.json"))]
    hashes = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    summary = {"status":STATUS,"scope":"Offline pure command mapping and unit tests only",
        "source_sha256":hashes,"production_integration":False,"native_physics_tested":False,
        "admitted_to_training":False}
    suite = unittest.defaultTestLoader.discover(str(ROOT/"tests"),pattern="test_a2_command_contract.py")
    stream = io.StringIO()
    tests = unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    (output/"unit_tests.log").write_text(stream.getvalue())
    summary["tests"] = {"run":tests.testsRun,"successful":tests.wasSuccessful(),
        "failures":[{"test":str(t),"traceback":trace} for t,trace in tests.failures],
        "errors":[{"test":str(t),"traceback":trace} for t,trace in tests.errors],
        "skipped":[{"test":str(t),"reason":reason} for t,reason in tests.skipped]}
    summary["stalled_joint_demonstration"] = demonstration()
    summary["limitations"] = [
        "Only reference rate/interior and directional accumulation are constrained; actual speed, torque, contact, safety and stability are not proved.",
        "A1 must use the same component if adopted; current A1 physical action certificates would require revalidation.",
        "The governor may change task trajectories; no learned policy, scorer or deployed controller is modified.",
        "Raw policy action/log probability must remain separate from the many-to-one accepted target; do not rewrite sampled actions as governed values.",
        "Each agent owns its own accepted reference state; this pure function stores no hidden command accumulator."]
    summary["source_hashes_unchanged"] = all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    summary["audit_passed"] = bool(tests.wasSuccessful() and summary["source_hashes_unchanged"])
    (output/"summary.json").write_text(json.dumps(jsonable(summary),indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":STATUS,"audit_passed":summary["audit_passed"],"tests":summary["tests"],
        "naive_delta_rad":summary["stalled_joint_demonstration"]["naive_target_delta_after_1s_rad"],
        "bounded_delta_rad":summary["stalled_joint_demonstration"]["bounded_target_delta_after_1s_rad"],
        "source_hashes_unchanged":summary["source_hashes_unchanged"]},indent=2))
    return 0 if summary["audit_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
