"""Run the single-arm, force-limited joint demonstration and log its state."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from .model import DEFAULT_CONFIG, DebugModel, duration, joint_target, load_config, load_debug_model


class SimulationFailure(RuntimeError):
    """A run did not meet the numerical or configured debug acceptance checks."""


def check_state(robot: DebugModel, target: np.ndarray) -> None:
    data = robot.data
    if not all(np.all(np.isfinite(array)) for array in (data.qpos, data.qvel, data.qacc, data.qfrc_actuator)):
        raise SimulationFailure("Non-finite simulation state")
    if any(warning.number for warning in data.warning):
        raise SimulationFailure("MuJoCo reported a numerical/solver warning")
    q = data.qpos[robot.qpos_ids]
    dq = data.qvel[robot.dof_ids]
    # These are diagnostics around debug windows, not a hardware safety system.
    if np.any(q < robot.windows[:, 0] - 0.005) or np.any(q > robot.windows[:, 1] + 0.005):
        raise SimulationFailure("Actual joint position left a configured debug window")
    if np.max(np.abs(dq)) > robot.config["max_observed_speed_rad_s"]:
        raise SimulationFailure("Observed speed exceeded the debug monitoring threshold")
    if np.max(np.abs(target - q)) > robot.config["max_tracking_error_rad"]:
        raise SimulationFailure("Tracking error exceeded the debug monitoring threshold")
    if np.any(np.abs(data.qfrc_actuator[robot.dof_ids]) > robot.torque_caps + 1e-7):
        raise SimulationFailure("Actuator torque exceeded its configured cap")


def capture_snapshot(robot: DebugModel, path: str | Path) -> None:
    from PIL import Image

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [0.05, -0.25, 0.72]
    camera.distance = 2.4
    camera.azimuth = 135
    camera.elevation = -12
    robot.model.vis.global_.offwidth = max(robot.model.vis.global_.offwidth, 960)
    robot.model.vis.global_.offheight = max(robot.model.vis.global_.offheight, 720)
    with mujoco.Renderer(robot.model, height=720, width=960) as renderer:
        renderer.update_scene(robot.data, camera=camera)
        pixels = renderer.render()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels).save(path)


def run_demo(robot: DebugModel, output: str | Path, *, viewer=None, snapshot: Path | None = None) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    # A crash or interruption must not leave an older successful run as evidence.
    summary_path.write_text(json.dumps({"status": "running", "parameter_status": robot.config["parameter_status"]}) + "\n",
                            encoding="utf-8")
    final_time = duration(robot.config)
    total_steps = round(final_time / robot.model.opt.timestep)
    decimation = round(robot.config["control_period_s"] / robot.model.opt.timestep)
    fields = ["time_s"]
    for name in robot.names:
        fields += [f"{name}_q_rad", f"{name}_dq_rad_s", f"{name}_target_rad", f"{name}_torque_nm"]
    fields += ["ee_body_x_m", "ee_body_y_m", "ee_body_z_m"]
    errors = []
    peak_torque = np.zeros(len(robot.names))
    peak_speed = np.zeros(len(robot.names))
    peak_error = np.zeros(len(robot.names))
    saturated_steps = np.zeros(len(robot.names), dtype=int)
    wall_start = time.monotonic()
    status, failure = "passed", None
    failure_exception = None
    target = joint_target(robot.config, 0)
    try:
        with (output / "trajectory.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(fields)
            for step in range(total_steps + 1):
                if step % decimation == 0 or step == total_steps:
                    target = robot.command(joint_target(robot.config, robot.data.time))
                    mujoco.mj_forward(robot.model, robot.data)
                    q = robot.data.qpos[robot.qpos_ids]
                    dq = robot.data.qvel[robot.dof_ids]
                    torque = robot.data.qfrc_actuator[robot.dof_ids]
                    row = [robot.data.time]
                    for index in range(len(robot.names)):
                        row += [q[index], dq[index], target[index], torque[index]]
                    row += robot.data.xpos[robot.ee_body].tolist()
                    writer.writerow(row)
                    errors.append(target - q)
                    if viewer is not None:
                        if not viewer.is_running():
                            raise SimulationFailure("Viewer closed before the full demonstration completed")
                        viewer.sync()
                        time.sleep(max(0, wall_start + robot.data.time - time.monotonic()))
                torque = np.abs(robot.data.qfrc_actuator[robot.dof_ids])
                speed = np.abs(robot.data.qvel[robot.dof_ids])
                error = np.abs(target - robot.data.qpos[robot.qpos_ids])
                if all(np.all(np.isfinite(values)) for values in (torque, speed, error)):
                    peak_torque = np.maximum(peak_torque, torque)
                    peak_speed = np.maximum(peak_speed, speed)
                    peak_error = np.maximum(peak_error, error)
                    saturated_steps += torque >= robot.torque_caps * (1 - 1e-6)
                check_state(robot, target)
                if step == total_steps:
                    break
                old_time = robot.data.time
                mujoco.mj_step(robot.model, robot.data)
                if robot.data.time <= old_time:
                    raise SimulationFailure("Simulation time failed to advance")
    except Exception as exc:
        status, failure = "failed", f"{type(exc).__name__}: {exc}"
        failure_exception = exc
    finite_errors = [error for error in errors if np.all(np.isfinite(error))]
    rmse = np.sqrt(np.mean(np.square(finite_errors), axis=0)) if finite_errors else np.full(len(robot.names), np.nan)
    summary = {
        "status": status,
        "failure": failure,
        "mujoco_version": mujoco.__version__,
        "duration_s": float(robot.data.time) if np.isfinite(robot.data.time) else None,
        "planned_duration_s": final_time,
        "parameter_status": robot.config["parameter_status"],
        "config": robot.config,
        "scope": "single_arm_joint_control_only",
        "tool_frame": "end_effector_block body origin; NOT a calibrated TCP or sword tip",
        "gravity_compensation": "model gravity via position-target offset, inside actuator force caps"
        if robot.config["gravity_compensation"] else "disabled",
        "sample_count": len(errors),
        "warnings": [int(w.number) for w in robot.data.warning],
        "joints": {
            name: {
                "tracking_rmse_rad": float(rmse[index]) if finite_errors else None,
                "peak_tracking_error_rad": float(peak_error[index]),
                "peak_speed_rad_s": float(peak_speed[index]),
                "peak_actuator_torque_nm": float(peak_torque[index]),
                "effective_debug_torque_cap_nm": float(robot.torque_caps[index]),
                "effective_debug_window_rad": robot.windows[index].tolist(),
                "saturated_state_evaluations": int(saturated_steps[index]),
            } for index, name in enumerate(robot.names)
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if status != "passed":
        raise SimulationFailure(f"{failure}; details saved to {summary_path}") from failure_exception
    if snapshot:
        capture_snapshot(robot, snapshot)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--headless", action="store_true", help="Run physics without opening a viewer")
    parser.add_argument("--output", type=Path, default=Path("outputs/joint_demo"))
    parser.add_argument("--snapshot", type=Path, help="Optional offscreen PNG; requires a working rendering backend")
    args = parser.parse_args()
    robot = load_debug_model(load_config(args.config))
    print("SIMULATION DEBUG ONLY: torque caps, windows, and gains are not hardware specifications.")
    if args.headless:
        summary = run_demo(robot, args.output, snapshot=args.snapshot)
    else:
        import mujoco.viewer

        with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
            summary = run_demo(robot, args.output, viewer=viewer, snapshot=args.snapshot)
    max_error = max(joint["peak_tracking_error_rad"] for joint in summary["joints"].values())
    print(f"{summary['status'].upper()}: {summary['duration_s']:.3f} s, max tracking error {max_error:.6f} rad")
    print(f"Logs saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
