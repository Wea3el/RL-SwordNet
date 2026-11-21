"""Fixed, original-size CAD arm and shinai on the floor, without a mannequin.

Building or viewing this scene is geometry inspection, not a dynamics pass or
hardware-feasibility claim. The clamp, inertias, collision proxies, debug torque
caps and any explicit exploratory joint ranges remain simulation assumptions.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .assembly import (
    ARM_BITS, ENV_BIT, TOOL_BIT, ROOT, Assembly, _build_tool, _finite_json,
    _hash_sources, _robot_proxies, contact_records, export_scene,
    load_assembly_config,
)
from .shinai import load_profile


CAD_TRANSLATION_M = np.array([0., 0., .008])
CAD_QUATERNION_WXYZ = np.array([1., 0., 0., 0.])


def _floor_scene(config):
    scene_xml = ET.parse(ROOT / config["source_scene"]).getroot()
    includes = list(scene_xml.findall("include"))
    if len(includes) != 1 or includes[0].get("file") != "robot.xml":
        raise ValueError("Expected the original scene with one robot.xml include")
    scene_xml.remove(includes[0])
    spec = mujoco.MjSpec.from_string(ET.tostring(scene_xml, encoding="unicode"))
    spec.modelname = "fixed_floor_original_size_arm_shinai"
    spec.meshdir = str(ROOT / "meshes")
    frame = spec.worldbody.add_frame(name="floor_CAD", pos=CAD_TRANSLATION_M,
                                     quat=CAD_QUATERNION_WXYZ)
    spec.worldbody.add_site(name="floor_mount", pos=[0, 0, 0], size=[.003]*3,
                           rgba=[.9, .4, .1, 1])
    robot = mujoco.MjSpec.from_file(str(ROOT / "robot.xml"))
    robot.meshdir = str(ROOT / "meshes")
    robot.default.name = "cad_robot"
    spec.option.integrator = robot.option.integrator
    # Same coarse, open-cage proxies and masks as the shoulder assembly. The
    # CAD visuals themselves are not silently changed into convex collisions.
    # The fixed base is welded to the world: floor friction is not its anchor.
    robot.worldbody.add_geom(name="base_collision", type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0, 0, .004], size=[.12, .12, .004], density=0, group=3,
        rgba=[.2, .8, .4, 0], contype=ENV_BIT, conaffinity=ARM_BITS | TOOL_BIT)
    for i, (x, y) in enumerate([(-.11, -.11), (-.11, .11), (.11, -.11), (.11, .11)]):
        robot.worldbody.add_geom(name=f"base_post_collision_{i}",
            type=mujoco.mjtGeom.mjGEOM_BOX, pos=[x, y, .075], size=[.01, .01, .075],
            density=0, group=3, rgba=[.2, .8, .4, 0],
            contype=ENV_BIT, conaffinity=ARM_BITS | TOOL_BIT)
    spec.attach(robot, prefix="", suffix="", frame=frame)
    return spec


def build_floor_assembly(joint_ranges=None):
    """Build a fixed floor arm with zero joint pose and unchanged source CAD.

    ``joint_ranges`` is an optional finite (7, 2) array in radians, J0--J6.
    Explicit ranges are exploratory assumptions, never identified hardware
    limits. They are intersected with any limits already present in the source
    XML and must contain zero; this function does not invent a new neutral pose.
    """
    config = copy.deepcopy(load_assembly_config())
    hashes = _hash_sources(config)
    half = np.asarray(config["runtime"]["joint_half_windows_rad"], dtype=float)
    requested = np.column_stack((-half, half)) if joint_ranges is None else np.asarray(joint_ranges, dtype=float)
    if requested.shape != (7, 2) or not np.isfinite(requested).all() or np.any(requested[:, 0] >= requested[:, 1]):
        raise ValueError("joint_ranges must be a finite (7, 2) array with lower < upper")
    spec = _floor_scene(config)
    original = spec.compile()
    if original.nq != 7 or original.nv != 7 or original.nu != 7:
        raise ValueError("Expected exactly the seven independent source joints and actuators")
    original_data = mujoco.MjData(original)
    mujoco.mj_forward(original, original_data)
    _robot_proxies(spec, original, original_data, config)
    effective = requested.copy()
    source_ranges = []
    for i in range(7):
        j = mujoco.mj_name2id(original, mujoco.mjtObj.mjOBJ_JOINT, f"J{i}")
        if j != i:
            raise ValueError("Expected source joint order J0 through J6")
        limited = bool(original.jnt_limited[j])
        interval = original.jnt_range[j].copy()
        source_ranges.append({"limited": limited, "range_rad": interval.tolist() if limited else None})
        if limited:
            effective[i] = [max(effective[i, 0], interval[0]), min(effective[i, 1], interval[1])]
    if np.any(effective[:, 0] >= effective[:, 1]) or np.any(effective[:, 0] > 0) or np.any(effective[:, 1] < 0):
        raise ValueError("Effective ranges must be nonempty and contain the CAD zero pose")
    del original_data, original
    _build_tool(spec, config)
    for geom in spec.geoms:
        if geom.type == mujoco.mjtGeom.mjGEOM_PLANE:
            geom.contype, geom.conaffinity = ENV_BIT, ARM_BITS | TOOL_BIT
    spec.option.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT) | int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)
    spec.option.timestep = config["runtime"]["timestep_s"]
    caps = np.asarray(config["runtime"]["torque_caps_nm"], dtype=float)
    for i in range(7):
        joint, actuator = spec.joint(f"J{i}"), spec.actuator(f"J{i}")
        joint.limited, joint.range = True, effective[i]
        actuator.ctrllimited, actuator.ctrlrange = True, effective[i]
        actuator.forcelimited, actuator.forcerange = True, [-caps[i], caps[i]]
    home = np.zeros(7)
    config["runtime"]["initial_pose"] = "floor_mounted_CAD_zero_reset_not_a_validated_motion_path"
    config["floor"] = {
        "status": "fixed_floor_mount_geometry_not_dynamics_or_hardware_validation",
        "translation_world_from_cad_m": CAD_TRANSLATION_M.tolist(),
        "quat_world_from_cad_wxyz": CAD_QUATERNION_WXYZ.tolist(),
        "source_base_contact_center_m": [0., 0., -.008],
        "active_humanoid": False,
        "inherited_torso_config_note": "torso settings are retained as provenance only; no torso is built",
        "joint_range_status": "original_debug_software_windows" if joint_ranges is None else "explicit_hypothetical_ranges_not_hardware_limits",
        "requested_joint_ranges_rad": requested.tolist(),
        "effective_joint_ranges_rad": effective.tolist(),
        "source_joint_ranges": source_ranges,
        "anchor": "fixed to world, not a free-standing friction-supported robot",
    }
    spec.add_key(name="floor_home", qpos=home.tolist(), ctrl=home.tolist())
    model = spec.compile()
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "floor_home")
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)
    # Native position-actuator target offset; the same forcerange caps its
    # complete output. Clipping can prevent a static hold: no hold is claimed.
    data.ctrl[:] = np.clip(home + data.qfrc_bias / model.actuator_gainprm[:, 0],
                           model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
    model.key_ctrl[key] = data.ctrl.copy()
    spec.key("floor_home").ctrl = data.ctrl.tolist()
    mujoco.mj_forward(model, data)
    return Assembly(model, data, config, home, spec, hashes)


def floor_build_report(assembly):
    """Describe the static build and all initial proxy contacts, without steps."""
    model, data, config = assembly.model, assembly.data, assembly.config
    contacts = contact_records(model, data)
    penetration = max([0.] + [-float(c.dist) for c in data.contact])
    threshold = config["collision"]["maximum_allowed_penetration_m"]
    sites = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
             for name in ("floor_mount", "mount_face", "shinai_grip", "shinai_tip", "shinai_pommel")}
    positions = {name: data.site_xpos[i].tolist() for name, i in sites.items()}
    original = mujoco.MjModel.from_xml_path(str(ROOT / "scene.xml"))
    unchanged = {}
    for i in range(1, original.nbody):
        name = mujoco.mj_id2name(original, mujoco.mjtObj.mjOBJ_BODY, i)
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        unchanged[name] = bool(j >= 0 and np.array_equal(original.body_mass[i], model.body_mass[j])
            and np.array_equal(original.body_ipos[i], model.body_ipos[j])
            and np.array_equal(original.body_inertia[i], model.body_inertia[j]))
    return _finite_json({
        "status": "geometry_built_initial_penetration" if penetration > threshold else "geometry_built",
        "physics_validation": "not_run",
        "scope": "fixed_floor_arm_and_rigid_shinai_static_geometry_only",
        "initialization": "CAD zero pose, no executed reset path or gravity-balance optimization",
        "source_sha256": assembly.source_hashes,
        "config": config,
        "mujoco_version": mujoco.__version__,
        "home_q_rad_J0_to_J6": assembly.home.tolist(),
        "initial_contacts": contacts,
        "maximum_initial_proxy_penetration_m": penetration,
        "initial_penetration_above_configured_threshold": penetration > threshold,
        "site_positions_world_m": positions,
        "J0_axis_world": data.xaxis[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "J0")].tolist(),
        "source_body_mass_COM_inertia_unchanged": unchanged,
        "source_dynamic_body_mass_kg": float(original.body_mass.sum()),
        "assembly_dynamic_body_mass_kg": float(model.body_mass.sum()),
        "added_sword_and_fixture_mass_kg": config["sword"]["mass_kg"] + config["fixture"]["mass_kg"],
        "shinai_total_length_m": float(np.linalg.norm(data.site_xpos[sites["shinai_tip"]] - data.site_xpos[sites["shinai_pommel"]])),
        "actual_grip_to_tip_m": float(np.linalg.norm(data.site_xpos[sites["shinai_tip"]] - data.site_xpos[sites["shinai_grip"]])),
        "shinai_profile_dimensions": load_profile()["dimensions"],
        "warnings": [int(w.number) for w in data.warning],
        "limitations": [
            "Original CAD visuals are collision-disabled; arm/base/tool collision geometry is approximate",
            "Parent filtering is disabled; adjacent arm proxy exclusions are inherited masks, not a full CAD clearance certificate",
            "The fixed base is welded to the world; stand-alone balance and floor friction are not simulated",
            "Real joint limits, motor limits, clamp strength, inertia and contact compliance are uncalibrated",
            "Building without initial deep proxy contacts does not establish collision-free motion or hardware feasibility",
        ],
    })


def configure_floor_camera(assembly, camera):
    """Frame this floor scene without depending on humanoid-only sites."""
    model, data = assembly.model, assembly.data
    positions = [np.zeros(3)]
    for name in ("shinai_tip", "shinai_pommel"):
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        positions.append(data.site_xpos[i].copy())
    positions.extend(data.xanchor.copy())
    points = np.asarray(positions)
    low, high = points.min(axis=0), points.max(axis=0)
    camera.lookat[:] = .5 * (low + high)
    camera.distance = max(2., 1.7 * float(np.linalg.norm(high - low)))
    camera.azimuth, camera.elevation = 135, -18


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/floor_assembly")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--build-only", action="store_true", help="Export a static build, without advancing physics (default)")
    mode.add_argument("--inspect", action="store_true", help="Open a static geometry viewer; no physics pass/fail is produced")
    args = parser.parse_args()
    assembly = build_floor_assembly()
    args.output.mkdir(parents=True, exist_ok=True)
    export_scene(assembly, args.output / "scene.xml")
    report = floor_build_report(assembly)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("STATIC FLOOR BUILD ONLY: original-size arm + shinai; fixed base; hypothetical fixture and debug caps.")
    print(f"Initial maximum proxy penetration: {report['maximum_initial_proxy_penetration_m']:.6f} m; no dynamics validation.")
    if args.inspect:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(assembly.model, assembly.data) as viewer:
            configure_floor_camera(assembly, viewer.cam)
            viewer.opt.geomgroup[3] = 0
            print("GEOMETRY VIEW ONLY: physics does not advance; close the viewer to exit.")
            while viewer.is_running():
                viewer.sync()
                time.sleep(.05)
    print(f"Artifacts: {args.output.resolve()}")


if __name__ == "__main__":
    main()
