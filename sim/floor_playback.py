"""Slow, finite-torque forward playback of a saved floor-matched skill.

This is a conservative simulation diagnostic, not source-speed playback,
duelling validation, or evidence of hardware feasibility. The sole qpos reset
places the model at Guard initially; travel from CAD zero is not validated.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from .assembly import ROOT, _finite_json, contact_records
from .floor import build_floor_assembly


TARGET_SPEED = .05
TARGET_ACCELERATION = .1
PHYSICS_DT = .002
CONTROL_DT = .01
HOLD_S = 1.


def segment_duration(delta, speed=TARGET_SPEED, acceleration=TARGET_ACCELERATION):
    """Stop-to-stop quintic bounds, rounded UP to a whole control period."""
    delta = np.asarray(delta, dtype=float)
    if not np.isfinite(delta).all() or not np.isfinite([speed, acceleration]).all() or min(speed, acceleration) <= 0:
        raise ValueError("Finite deltas and positive speed/acceleration are required")
    distance = float(np.max(np.abs(delta), initial=0.))
    seconds = max(1.875 * distance / speed, np.sqrt(5.774 * distance / acceleration), .02)
    return float(np.ceil(seconds / CONTROL_DT) * CONTROL_DT)


def quintic_position(start, end, fraction):
    u = np.clip(fraction, 0., 1.)
    return np.asarray(start) + (10*u**3 - 15*u**4 + 6*u**5) * (np.asarray(end) - np.asarray(start))


def _read_q(path):
    with np.load(path, allow_pickle=False) as source:
        q = np.asarray(source["qpos"], dtype=float).copy()
    if q.ndim != 2 or q.shape[1] != 7 or len(q) < 1 or not np.isfinite(q).all():
        raise ValueError(f"Invalid finite Nx7 qpos in {path}")
    return q


def load_sequence(input_dir, skill):
    if skill not in ("stab", "parry_up"):
        raise ValueError("skill must be stab or parry_up")
    input_dir = Path(input_dir).resolve()
    paths = {name: input_dir/f"{name}.npz" for name in
             ("guard", f"guard_to_{skill}", skill, f"{skill}_to_guard")}
    arrays = {name: _read_q(path) for name, path in paths.items()}
    guard = arrays["guard"][0]
    if not np.allclose(arrays["guard"], guard, atol=1e-10, rtol=0):
        raise ValueError("Guard must be a single fixed pose")
    pieces = [(f"guard_to_{skill}", arrays[f"guard_to_{skill}"]),
              (skill, arrays[skill]), (f"{skill}_to_guard", arrays[f"{skill}_to_guard"])]
    previous = guard
    starts, ends, labels, durations = [], [], [], []
    for label, values in pieces:
        if not np.allclose(previous, values[0], atol=1e-7, rtol=0):
            raise ValueError(f"Discontinuous saved path at {label}")
        for value in values[1:]:
            starts.append(previous.copy())
            ends.append(value.copy())
            labels.append(label)
            durations.append(segment_duration(value-previous))
            previous = value
    if not np.allclose(previous, guard, atol=1e-7, rtol=0):
        raise ValueError("Saved return path does not end at Guard")
    all_q = np.vstack([guard, ends])
    if np.any(np.abs(all_q) > np.pi + 1e-10):
        raise ValueError("Saved path leaves the explicitly hypothetical +/-pi ranges")
    cumulative = np.r_[HOLD_S, HOLD_S + np.cumsum(durations)]
    return {"guard": guard, "starts": np.asarray(starts), "ends": np.asarray(ends),
            "labels": labels, "durations": np.asarray(durations), "boundaries": cumulative,
            "duration_s": float(cumulative[-1] + HOLD_S),
            "data_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths.values()}}


def sequence_target(sequence, time_s):
    if time_s < HOLD_S:
        return sequence["guard"].copy(), "initial_guard_hold"
    if time_s >= sequence["boundaries"][-1]:
        return sequence["guard"].copy(), "final_guard_hold"
    i = int(np.searchsorted(sequence["boundaries"], time_s, side="right") - 1)
    fraction = (time_s - sequence["boundaries"][i]) / sequence["durations"][i]
    return quintic_position(sequence["starts"][i], sequence["ends"][i], fraction), sequence["labels"][i]


def apply_position_gravity_target(model, data, gravity_data, target):
    """Offset native position target; complete output still uses forcerange."""
    gravity_data.qpos[:] = data.qpos
    gravity_data.qvel[:] = 0
    mujoco.mj_forward(model, gravity_data)
    data.ctrl[:] = np.clip(np.asarray(target) + gravity_data.qfrc_bias/model.actuator_gainprm[:, 0],
                           model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])


def run_playback(input_dir, skill, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    summary_path = output/"summary.json"
    summary_path.write_text(json.dumps({"status": "running", "skill": skill}) + "\n", encoding="utf-8")
    result = {"status": "running", "skill": skill, "scope": "retimed_forward_stepped_proxy_collision_debug_test_not_hardware_or_duelling_validation",
              "initialization": "reset directly to saved Guard; CAD-zero-to-Guard travel not validated",
              "input": str(Path(input_dir).resolve()), "mujoco_version": mujoco.__version__}
    model = data = None
    peak_error, peak_speed, peak_torque = np.zeros(7), np.zeros(7), np.zeros(7)
    max_penetration, max_grip_error, max_tip_error = 0., 0., 0.
    saturation = np.zeros(7, dtype=int)
    failure, samples, completed, phase = None, 0, 0., "initialization"
    deepest_contacts = []
    try:
        sequence = load_sequence(input_dir, skill)
        assembly = build_floor_assembly(np.tile([-np.pi, np.pi], (7, 1)))
        model, data, config = assembly.model, assembly.data, assembly.config
        if not np.isclose(model.opt.timestep, PHYSICS_DT) or np.any(model.body_gravcomp) or np.any(data.qfrc_applied):
            raise ValueError("Expected .002s stepping, no gravcomp, and no applied generalized forces")
        gains = model.actuator_gainprm[:, 0]
        if np.any(gains <= 0) or not np.allclose(model.actuator_biasprm[:, 1], -gains):
            raise ValueError("Expected existing native position actuators with positive kp")
        caps = np.asarray(config["runtime"]["torque_caps_nm"])
        result.update({"source_sha256": assembly.source_hashes, "data_sha256": sequence["data_sha256"],
            "config": config, "caps_nm": caps.tolist(), "kp_nm_per_rad": model.actuator_gainprm[:, 0].tolist(),
            "planned_duration_s": sequence["duration_s"], "physics_dt_s": PHYSICS_DT, "control_dt_s": CONTROL_DT,
            "retiming": {"method": "each adjacent saved q pair: stop-to-stop minimum-jerk quintic; durations rounded up to .01s",
                "target_speed_cap_rad_s": TARGET_SPEED, "target_acceleration_cap_rad_s2": TARGET_ACCELERATION,
                "minimum_segment_s": .02, "initial_and_final_hold_s_each": HOLD_S, "source_timing_used": False,
                "segment_count": len(sequence["durations"])},
            "abort_thresholds": {"penetration_m": .001, "actual_speed_rad_s": .5, "tracking_error_rad": .05},
            "initial_qpos": sequence["guard"].tolist()})
        data.qpos[:], data.qvel[:] = sequence["guard"], 0
        mujoco.mj_forward(model, data)
        result["initial_contacts"] = contact_records(model, data)
        gravity, reference = mujoco.MjData(model), mujoco.MjData(model)
        site_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in ("shinai_grip", "shinai_tip")]
        steps = int(round(sequence["duration_s"]/PHYSICS_DT))
        control_steps, log_steps = int(round(CONTROL_DT/PHYSICS_DT)), 25
        target = sequence["guard"].copy()
        with (output/"trajectory.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["time_s", "phase"] + [f"J{i}_{field}" for field in ("q_rad", "dq_rad_s", "target_rad", "torque_nm") for i in range(7)]
                + [f"{name}_{axis}_m" for name in ("grip", "tip") for axis in "xyz"])
            for step in range(steps + 1):
                if step % control_steps == 0:
                    target, phase = sequence_target(sequence, data.time)
                    apply_position_gravity_target(model, data, gravity, target)
                mujoco.mj_forward(model, data)
                completed = float(data.time)
                if not all(np.isfinite(a).all() for a in (data.qpos, data.qvel, data.qacc, data.qfrc_actuator)):
                    raise RuntimeError("Nonfinite simulator state")
                error, speed, torque = np.abs(data.qpos-target), np.abs(data.qvel), np.abs(data.qfrc_actuator)
                peak_error, peak_speed, peak_torque = np.maximum(peak_error, error), np.maximum(peak_speed, speed), np.maximum(peak_torque, torque)
                saturation += torque >= caps*(1-1e-6)
                reference.qpos[:], reference.qvel[:] = target, 0
                mujoco.mj_forward(model, reference)
                site_error = np.linalg.norm(data.site_xpos[site_ids]-reference.site_xpos[site_ids], axis=1)
                max_grip_error, max_tip_error = max(max_grip_error, site_error[0]), max(max_tip_error, site_error[1])
                penetration = max([0.] + [-float(c.dist) for c in data.contact])
                if penetration > max_penetration:
                    max_penetration, deepest_contacts = penetration, contact_records(model, data)
                if step % log_steps == 0 or step == steps:
                    writer.writerow([data.time, phase] + np.r_[data.qpos, data.qvel, target, data.qfrc_actuator, data.site_xpos[site_ids].ravel()].tolist())
                    samples += 1
                if any(w.number for w in data.warning):
                    raise RuntimeError("MuJoCo warning")
                if penetration > .001:
                    raise RuntimeError(f"Proxy penetration {penetration:.6g}m exceeds .001m")
                if speed.max() > .5 or error.max() > .05:
                    raise RuntimeError("Actual joint speed or tracking error exceeds diagnostic threshold")
                if np.any(torque > caps + 1e-7):
                    raise RuntimeError("Complete native actuator output exceeds configured torque cap")
                if np.any(data.qpos < model.jnt_range[:, 0]-.005) or np.any(data.qpos > model.jnt_range[:, 1]+.005):
                    raise RuntimeError("Joint left hypothetical range")
                if step == steps:
                    break
                previous = data.time
                mujoco.mj_step(model, data)
                if data.time <= previous:
                    raise RuntimeError("Physics time did not advance")
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
    result.update({"status": "failed" if failure else "passed", "failure": failure,
        "completed_duration_s": completed, "failure_or_final_phase": phase, "csv_sample_count": samples,
        "peak_joint_tracking_error_rad": peak_error.tolist(), "peak_actual_joint_speed_rad_s": peak_speed.tolist(),
        "peak_complete_actuator_torque_nm": peak_torque.tolist(), "saturated_state_evaluations": saturation.tolist(),
        "maximum_proxy_penetration_m": max_penetration, "deepest_contacts": deepest_contacts,
        "maximum_grip_error_to_commanded_q_fk_m": max_grip_error,
        "maximum_tip_error_to_commanded_q_fk_m": max_tip_error,
        "final_qpos": None if data is None else data.qpos.tolist(),
        "final_contacts": [] if data is None else contact_records(model, data),
        "warnings": [] if data is None else [int(w.number) for w in data.warning],
        "limitations": ["A pass applies only to this slow retiming, proxy geometry and hypothetical +/-pi joint ranges",
            "Real motor limits and collision geometry remain uncalibrated; no hardware feasibility is established",
            "Tool tracking errors are relative to interpolated commanded joint FK, not the source motion targets",
            "The one initial Guard reset is not a validated motion from CAD zero; source timing is not reproduced"]})
    summary_path.write_text(json.dumps(_finite_json(result), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT/"outputs/floor_matching_round1_refined/hypothetical180")
    parser.add_argument("--skill", choices=("stab", "parry_up", "both"), default="both")
    parser.add_argument("--output", type=Path, required=True, help="New output directory; existing paths are never overwritten")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    skills = ("stab", "parry_up") if args.skill == "both" else (args.skill,)
    failed = False
    for skill in skills:
        result = run_playback(args.input, skill, args.output/skill)
        print(f"{skill}: {result['status']}; simulated {result['completed_duration_s']:.3f}s; {result['failure'] or 'no threshold violation'}", flush=True)
        failed |= result["status"] != "passed"
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
