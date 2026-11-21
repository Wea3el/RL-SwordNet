"""Load the original CAD model with explicitly provisional debug parameters."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "debug.json"


def positive(value: float, name: str, *, allow_zero: bool = False) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    return value


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        config = json.load(stream)
    if config.get("schema_version") != 1:
        raise ValueError("Expected configuration schema_version 1")
    if config.get("parameter_status") != "simulation_debug_only_not_hardware_limits":
        raise ValueError("This tool requires an explicitly labelled simulation-debug configuration")
    if not isinstance(config.get("gravity_compensation"), bool):
        raise ValueError("gravity_compensation must be a boolean")
    dt = positive(config["timestep_s"], "timestep_s")
    period = positive(config["control_period_s"], "control_period_s")
    ratio = period / dt
    if ratio < 1 or not np.isclose(ratio, round(ratio)):
        raise ValueError("control_period_s must be an integer multiple of timestep_s")
    for key in ("max_tracking_error_rad", "max_observed_speed_rad_s"):
        positive(config[key], key)
    trajectory = config["trajectory"]
    amplitude = positive(trajectory["amplitude_rad"], "amplitude_rad")
    segment = positive(trajectory["segment_duration_s"], "segment_duration_s")
    for key in ("initial_hold_s", "final_hold_s"):
        positive(trajectory[key], key, allow_zero=True)
    for key in ("segment_duration_s", "initial_hold_s", "final_hold_s"):
        steps = trajectory[key] / period
        if not np.isclose(steps, round(steps)):
            raise ValueError(f"{key} must be an integer multiple of control_period_s")
    if not config["joints"]:
        raise ValueError("At least one joint must be configured")
    for name, entry in config["joints"].items():
        window = np.asarray(entry["debug_window_rad"], dtype=float)
        if window.shape != (2,) or not np.all(np.isfinite(window)) or window[0] >= window[1]:
            raise ValueError(f"Invalid debug_window_rad for {name}")
        home = float(entry["initial_q_rad"])
        if not np.isfinite(home) or not window[0] < home < window[1]:
            raise ValueError(f"{name}: initial position must lie strictly inside its debug window")
        if home + amplitude > window[1]:
            raise ValueError(f"{name}: trajectory would leave its debug window")
        positive(entry["debug_torque_cap_nm"], f"{name} torque cap")
        speed = positive(entry["max_target_speed_rad_s"], f"{name} target speed")
        if amplitude * np.pi / segment > speed:
            raise ValueError(f"{name}: trajectory would exceed target speed limit")
    return config


@dataclass
class DebugModel:
    model: mujoco.MjModel
    data: mujoco.MjData
    gravity_data: mujoco.MjData
    config: dict
    names: list[str]
    joint_ids: np.ndarray
    qpos_ids: np.ndarray
    dof_ids: np.ndarray
    actuator_ids: np.ndarray
    kp: np.ndarray
    home: np.ndarray
    windows: np.ndarray
    torque_caps: np.ndarray
    target_speed_caps: np.ndarray
    last_target: np.ndarray
    last_command_time: float
    ee_body: int

    def command(self, target: np.ndarray) -> np.ndarray:
        """Apply a position target with optional model-gravity feedforward.

        The native position actuator supplies implicit damping. The offset
        g(q)/kp adds feedforward THROUGH the same force-limited actuator;
        no external force or simulator body-gravity cancellation is used.
        """
        target = np.asarray(target, dtype=float)
        if target.shape != self.home.shape or not np.all(np.isfinite(target)):
            raise ValueError("Target must contain one finite angle per configured joint")
        if np.any(target < self.windows[:, 0]) or np.any(target > self.windows[:, 1]):
            raise ValueError("Target lies outside a configured debug window")
        elapsed = self.data.time - self.last_command_time
        if elapsed < 0:
            raise ValueError("Simulation time moved backwards; reload the controller after a reset")
        excursion = elapsed * self.target_speed_caps
        applied_target = np.clip(target, self.last_target - excursion, self.last_target + excursion)
        control = applied_target.copy()
        if self.config["gravity_compensation"]:
            self.gravity_data.qpos[:] = self.data.qpos
            self.gravity_data.qvel[:] = 0
            mujoco.mj_forward(self.model, self.gravity_data)
            control += self.gravity_data.qfrc_bias[self.dof_ids] / self.kp
        self.data.ctrl[self.actuator_ids] = np.clip(control, self.windows[:, 0], self.windows[:, 1])
        self.last_target = applied_target.copy()
        self.last_command_time = float(self.data.time)
        return applied_target


def load_debug_model(config: dict) -> DebugModel:
    scene = Path(config["source_scene"])
    if not scene.is_absolute():
        scene = PROJECT_ROOT / scene
    model = mujoco.MjModel.from_xml_path(str(scene))
    names = list(config["joints"])
    if model.nq != len(names) or model.nv != len(names) or model.nu != len(names):
        raise ValueError("The debug controller requires one scalar joint and actuator per configured DOF")
    joint_ids, actuator_ids = [], []
    for name in names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if jid < 0 or aid < 0:
            raise ValueError(f"Missing joint or actuator named {name}")
        if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError(f"{name} must be a hinge joint")
        if (model.actuator_trntype[aid] != mujoco.mjtTrn.mjTRN_JOINT
                or model.actuator_trnid[aid, 0] != jid
                or not np.allclose(model.actuator_gear[aid], [1, 0, 0, 0, 0, 0])):
            raise ValueError(f"{name} requires a direct, unit-gear joint actuator")
        kp = model.actuator_gainprm[aid, 0]
        if (kp <= 0 or model.actuator_gaintype[aid] != mujoco.mjtGain.mjGAIN_FIXED
                or model.actuator_biastype[aid] != mujoco.mjtBias.mjBIAS_AFFINE
                or model.actuator_dyntype[aid] != mujoco.mjtDyn.mjDYN_NONE
                or not np.isclose(model.actuator_biasprm[aid, 0], 0)
                or not np.isclose(model.actuator_biasprm[aid, 1], -kp)):
            raise ValueError(f"{name} requires a native position actuator")
        joint_ids.append(jid)
        actuator_ids.append(aid)
    joint_ids = np.asarray(joint_ids)
    actuator_ids = np.asarray(actuator_ids)
    windows = np.asarray([config["joints"][n]["debug_window_rad"] for n in names], dtype=float)
    # Debug windows must never expand any physical limits added to the source.
    for index, jid in enumerate(joint_ids):
        if model.jnt_limited[jid]:
            windows[index, 0] = max(windows[index, 0], model.jnt_range[jid, 0])
            windows[index, 1] = min(windows[index, 1], model.jnt_range[jid, 1])
    home = np.asarray([config["joints"][n]["initial_q_rad"] for n in names], dtype=float)
    if np.any(home <= windows[:, 0]) or np.any(home + config["trajectory"]["amplitude_rad"] >= windows[:, 1]):
        raise ValueError("Home/trajectory is incompatible with source-model joint limits")
    caps = np.asarray([config["joints"][n]["debug_torque_cap_nm"] for n in names], dtype=float)
    for index, aid in enumerate(actuator_ids):
        if model.actuator_forcelimited[aid]:
            caps[index] = min(caps[index], model.actuator_forcerange[aid, 1], -model.actuator_forcerange[aid, 0])
        if model.actuator_ctrllimited[aid]:
            windows[index, 0] = max(windows[index, 0], model.actuator_ctrlrange[aid, 0])
            windows[index, 1] = min(windows[index, 1], model.actuator_ctrlrange[aid, 1])
    if np.any(caps <= 0) or np.any(home <= windows[:, 0]) or np.any(home + config["trajectory"]["amplitude_rad"] >= windows[:, 1]):
        raise ValueError("Debug configuration is incompatible with source-model actuator limits")
    model.jnt_limited[joint_ids] = 1
    model.jnt_range[joint_ids] = windows
    model.actuator_ctrllimited[actuator_ids] = 1
    model.actuator_ctrlrange[actuator_ids] = windows
    model.actuator_forcelimited[actuator_ids] = 1
    model.actuator_forcerange[actuator_ids] = np.column_stack((-caps, caps))
    model.opt.timestep = config["timestep_s"]
    # A numerical failure must be reported rather than hidden by an automatic reset.
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_AUTORESET
    qpos_ids = model.jnt_qposadr[joint_ids].copy()
    dof_ids = model.jnt_dofadr[joint_ids].copy()
    data = mujoco.MjData(model)
    data.qpos[qpos_ids] = home
    mujoco.mj_forward(model, data)
    ee_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, config["end_effector_body"])
    if ee_body < 0:
        raise ValueError("Configured end-effector body is missing")
    return DebugModel(
        model=model, data=data, gravity_data=mujoco.MjData(model), config=config, names=names,
        joint_ids=joint_ids, qpos_ids=qpos_ids, dof_ids=dof_ids, actuator_ids=actuator_ids,
        kp=model.actuator_gainprm[actuator_ids, 0].copy(), home=home,
        windows=windows, torque_caps=caps,
        target_speed_caps=np.asarray([config["joints"][n]["max_target_speed_rad_s"] for n in names]),
        last_target=home.copy(), last_command_time=-config["control_period_s"], ee_body=ee_body,
    )


def duration(config: dict) -> float:
    trajectory = config["trajectory"]
    return (trajectory["initial_hold_s"] + len(config["joints"]) * trajectory["segment_duration_s"]
            + trajectory["final_hold_s"])


def joint_target(config: dict, time_s: float) -> np.ndarray:
    """One small cosine round trip per joint, starting and ending at rest."""
    target = np.asarray([entry["initial_q_rad"] for entry in config["joints"].values()], dtype=float)
    trajectory = config["trajectory"]
    elapsed = time_s - trajectory["initial_hold_s"]
    segment = trajectory["segment_duration_s"]
    if 0 <= elapsed < len(target) * segment:
        active = int(elapsed / segment)
        phase = (elapsed - active * segment) / segment
        target[active] += 0.5 * trajectory["amplitude_rad"] * (1 - np.cos(2 * np.pi * phase))
    return target
