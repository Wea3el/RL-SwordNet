"""Inspect the unchanged source model; no debug parameters are applied."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from .model import PROJECT_ROOT


def audit_model(scene: str | Path = PROJECT_ROOT / "scene.xml") -> dict:
    model = mujoco.MjModel.from_xml_path(str(Path(scene).resolve()))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    joints = []
    for jid in range(model.njnt):
        body = int(model.jnt_bodyid[jid])
        joints.append({
            "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid),
            "body": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body),
            "parent_body": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.body_parentid[body])),
            "joint_type": int(model.jnt_type[jid]),
            "axis_body": model.jnt_axis[jid].tolist(),
            "axis_world_at_cad_pose": data.xaxis[jid].tolist(),
            "anchor_world_m": data.xanchor[jid].tolist(),
            "range_rad": model.jnt_range[jid].tolist() if model.jnt_limited[jid] else None,
            "qpos_address": int(model.jnt_qposadr[jid]),
            "dof_address": int(model.jnt_dofadr[jid]),
        })
    actuators = []
    for aid in range(model.nu):
        actuators.append({
            "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid),
            "transmission_type": int(model.actuator_trntype[aid]),
            "transmission_target_ids": model.actuator_trnid[aid].tolist(),
            "gear": model.actuator_gear[aid].tolist(),
            "kp": float(model.actuator_gainprm[aid, 0]),
            "velocity_damping": float(-model.actuator_biasprm[aid, 2]),
            "control_range": model.actuator_ctrlrange[aid].tolist() if model.actuator_ctrllimited[aid] else None,
            "force_range": model.actuator_forcerange[aid].tolist() if model.actuator_forcelimited[aid] else None,
        })
    bodies = []
    for bid in range(1, model.nbody):
        inertia = model.body_inertia[bid]
        bodies.append({
            "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid),
            "mass_kg": float(model.body_mass[bid]),
            "com_body_m": model.body_ipos[bid].tolist(),
            "principal_inertia_kg_m2": inertia.tolist(),
            "inertial_quaternion_wxyz": model.body_iquat[bid].tolist(),
            "positive_inertia": bool(np.all(inertia > 0)),
            "inertia_triangle_valid": bool(2 * inertia.max() <= inertia.sum() + 1e-12),
        })
    robot_geoms = np.arange(model.ngeom)[model.geom_type != mujoco.mjtGeom.mjGEOM_PLANE]
    collision_geoms = robot_geoms[(model.geom_contype[robot_geoms] != 0) | (model.geom_conaffinity[robot_geoms] != 0)]
    return {
        "scene": str(Path(scene).resolve()),
        "mujoco_version": mujoco.__version__,
        "parameter_status": "source_CAD_model_not_hardware_calibrated",
        "nq": model.nq, "nv": model.nv, "nu": model.nu,
        "fixed_base": not bool(np.any(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)),
        "moving_mass_kg": float(model.body_mass[1:].sum()),
        "timestep_s": float(model.opt.timestep),
        "gravity_m_s2": model.opt.gravity.tolist(),
        "equality_constraints": model.neq,
        "robot_collision_geom_count": len(collision_geoms),
        "sites": [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(model.nsite)],
        "gravity_torque_at_cad_pose_nm": data.qfrc_bias.tolist(),
        "joints": joints, "actuators": actuators, "bodies": bodies,
        "known_limitations": [
            "Independent real-world DOFs and transmission coupling need confirmation.",
            "CAD masses and inertias need calibration against the printed assembly.",
            "Debug settings do not establish hardware torque, speed, or angle limits.",
            "Sword mounting, sword/TCP sites, and upright sword pose are not configured.",
            "No collision or scoring validation has been performed.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=PROJECT_ROOT / "scene.xml")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_model(args.scene)
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(f"Source model: {report['nq']} DOFs, {report['nu']} actuators, "
          f"moving mass {report['moving_mass_kg']:.6f} kg")
    print(f"Fixed base: {report['fixed_base']}; robot collision geoms: {report['robot_collision_geom_count']}")
    print("Joint windows and force caps below refer to the ORIGINAL model:")
    for joint, actuator in zip(report["joints"], report["actuators"]):
        print(f"  {joint['name']}: range={joint['range_rad']}, force_range={actuator['force_range']}")
    if args.output:
        print(f"Audit saved to {args.output}")


if __name__ == "__main__":
    main()
