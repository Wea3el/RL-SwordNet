"""Build and check a four-hole-mounted, single-arm shinai scene.

Original CAD XML and meshes remain unchanged. The adapter is a proposed rigid
fixture; masses, collision proxies and motion bounds are simulation assumptions.
"""
from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from .shinai import load_profile, revolved_mesh

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/assembly.json"
ARM_BITS = (1 << 7) - 1
ENV_BIT, TOOL_BIT = 1 << 8, 1 << 9


def _vector(value, length, name, positive=False):
    result = np.asarray(value, dtype=float)
    if result.shape != (length,) or not np.isfinite(result).all() or (positive and np.any(result <= 0)):
        raise ValueError(f"Invalid {name}")
    return result


def _positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def load_assembly_config(path=DEFAULT_CONFIG):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_config(config)
    return config


def validate_config(config):
    if config.get("schema_version") != 2 or config.get("parameter_status") != "simulation_fixture_and_debug_limits_not_hardware_specification":
        raise ValueError("Expected explicitly labelled assembly simulation configuration")
    mount, fixture, sword, runtime = (config[k] for k in ("mount", "fixture", "sword", "runtime"))
    _vector(mount["center_ee_m"], 3, "mount center")
    quat = _vector(mount["quat_ee_from_mount_wxyz"], 4, "mount quaternion")
    if not np.isclose(np.linalg.norm(quat), 1, rtol=0, atol=1e-7):
        raise ValueError("Mount quaternion must be unit length")
    for key in ("face_width_m", "hole_pitch_m", "hole_diameter_m"):
        _positive(mount[key], key)
    if mount["hole_pitch_m"] + mount["hole_diameter_m"] >= mount["face_width_m"]:
        raise ValueError("Mount holes must lie inside the face")
    for key in ("plate_thickness_m", "grip_from_pommel_m", "collar_width_m", "collar_radial_wall_m", "collar_clearance_m", "bridge_half_width_m"):
        _positive(fixture[key], key)
    dimensions = load_profile()["dimensions"]
    station = fixture["grip_from_pommel_m"]
    half = fixture["collar_width_m"] / 2
    if not half < station < dimensions["handle_length_m"]-half:
        raise ValueError("The clamp must lie within the handle")
    center = _vector(fixture["grip_center_mount_m"],3,"grip center")
    tool_quat = _vector(fixture["quat_mount_from_sword_wxyz"],4,"transverse grip quaternion")
    if not np.isclose(np.linalg.norm(tool_quat),1,atol=1e-7):
        raise ValueError("Grip rotation must be unit length")
    rotation = _quat_matrix(tool_quat)
    if abs(rotation[2,2])>1e-7 or center[2]<=fixture["plate_thickness_m"]+dimensions["grip_radius_m"]:
        raise ValueError("Sword must be transverse to, and outside, the mounting face")
    for part in (fixture, sword):
        _positive(part["mass_kg"], "mass")
        inertia = _vector(part["inertia_diagonal_kg_m2"], 3, "inertia", positive=True)
        if 2 * inertia.max() >= inertia.sum():
            raise ValueError("Principal inertias must satisfy strict triangle inequalities")
    _vector(fixture["com_from_mount_m"], 3, "fixture COM")
    if not 0 < sword["com_from_pommel_m"] < dimensions["total_length_m"]:
        raise ValueError("Sword COM must lie within its length")
    _vector(config["torso"]["center_m"], 3, "torso center")
    for key in ("right_shoulder_mount_local_m","source_base_contact_center_m"):
        _vector(config["torso"][key],3,key)
    torso_quat = _vector(config["torso"]["quat_world_from_cad_wxyz"],4,"shoulder rotation")
    if not np.isclose(np.linalg.norm(torso_quat),1,atol=1e-7):
        raise ValueError("Shoulder rotation must be unit length")
    for key in ("joint_half_windows_rad", "torque_caps_nm"):
        _vector(runtime[key], 7, key, positive=True)
    _vector(config["collision"]["arm_proxy_radii_m"], 7, "proxy radii", positive=True)
    _positive(config["collision"]["maximum_allowed_penetration_m"], "penetration threshold")
    for key in ("timestep_s", "control_period_s", "max_observed_speed_rad_s", "max_tracking_error_rad", "max_target_speed_rad_s", "motion_amplitude_rad", "motion_period_s", "default_check_duration_s"):
        _positive(runtime[key], key)
    ratio = runtime["control_period_s"] / runtime["timestep_s"]
    if ratio < 1 or not np.isclose(ratio, round(ratio)):
        raise ValueError("Control period must be an integer multiple of the physics timestep")
    if runtime["motion_amplitude_rad"] >= min(runtime["joint_half_windows_rad"]):
        raise ValueError("Motion amplitude must fit the joint windows")
    if np.pi * runtime["motion_amplitude_rad"] / runtime["motion_period_s"] > runtime["max_target_speed_rad_s"]:
        raise ValueError("Diagnostic trajectory exceeds target speed limit")


def _hash_sources(config):
    hashes = {}
    for relative, expected in config["expected_source_sha256"].items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError("Source path outside the repository")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"{relative}: source changed; remeasure the mount before rebuilding")
        hashes[relative] = actual
    return hashes


def _quat_matrix(quaternion):
    result = np.empty(9)
    mujoco.mju_quat2Mat(result, np.asarray(quaternion, dtype=float))
    return result.reshape(3, 3)


def _add_mesh(spec, name, vertices, faces):
    spec.add_mesh(name=name, uservert=np.asarray(vertices).ravel().tolist(), userface=np.asarray(faces).ravel().tolist())


def _collision(body, name, kind, *, contype, conaffinity, **geometry):
    return body.add_geom(name=name, type=kind, contype=contype, conaffinity=conaffinity,
                         density=0, group=3, rgba=[.2, .8, .4, 0.], friction=[.4, .005, .0001], **geometry)


def _interval_radius(points, a, b):
    """Maximum piecewise-linear radius, including knots missed by sampling."""
    internal = points[(points[:,0] > a) & (points[:,0] < b),1]
    return float(np.max(np.r_[np.interp([a,b],points[:,0],points[:,1]),internal]))


def _robot_proxies(spec, model, data, config):
    radii = config["collision"]["arm_proxy_radii_m"]
    for i in range(7):
        body_id = int(model.jnt_bodyid[i])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        inverse_rotation = data.xmat[body_id].reshape(3, 3).T
        start = inverse_rotation @ (data.xanchor[i] - data.xpos[body_id])
        end_world = data.xanchor[i+1] if i < 6 else data.xanchor[i] + data.xmat[body_id].reshape(3, 3) @ np.array([.025, 0, .005])
        end = inverse_rotation @ (end_world - data.xpos[body_id])
        adjacent = sum(1 << j for j in range(max(0, i-1), min(7, i+2)))
        affinity = ENV_BIT | TOOL_BIT | (ARM_BITS & ~adjacent)
        if np.linalg.norm(end-start) < .01:
            _collision(spec.body(name), f"arm_proxy_{i}", mujoco.mjtGeom.mjGEOM_SPHERE,
                       pos=.5*(start+end), size=[radii[i], 0, 0], contype=1 << i, conaffinity=affinity)
        else:
            # Endpoints lie on consecutive joint anchors of this rigid link.
            _collision(spec.body(name), f"arm_proxy_{i}", mujoco.mjtGeom.mjGEOM_CAPSULE,
                       fromto=np.r_[start, end], size=[radii[i], 0, 0], contype=1 << i, conaffinity=affinity)


def _build_tool(spec, config):
    from .fixture_mesh import collar_mesh, perforated_plate

    mount, fixture, sword = (config[k] for k in ("mount", "fixture", "sword"))
    profile = load_profile()
    dimensions = profile["dimensions"]
    ee = spec.body(config["end_effector_body"])
    face = ee.add_body(name="sword_fixture", pos=mount["center_ee_m"], quat=mount["quat_ee_from_mount_wxyz"])
    face.mass, face.ipos = fixture["mass_kg"], fixture["com_from_mount_m"]
    face.inertia, face.explicitinertial = fixture["inertia_diagonal_kg_m2"], True
    face.add_site(name="mount_face", size=[.002, .002, .002], rgba=[.8,.3,.1,1])
    thickness = fixture["plate_thickness_m"]
    v, f = perforated_plate(width_m=mount["face_width_m"], thickness_m=thickness,
                            pitch_m=mount["hole_pitch_m"], hole_radius_m=mount["hole_diameter_m"]/2)
    _add_mesh(spec, "adapter_plate_mesh", v, f)
    face.add_geom(name="adapter_plate_visual", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="adapter_plate_mesh",
                  density=0, contype=0, conaffinity=0, rgba=[.12,.23,.35,1], group=1)
    _collision(face, "adapter_plate_collision", mujoco.mjtGeom.mjGEOM_BOX, pos=[0,0,thickness/2],
               size=[mount["face_width_m"]/2]*2+[thickness/2], contype=TOOL_BIT, conaffinity=ENV_BIT|ARM_BITS)
    pitch = mount["hole_pitch_m"]/2
    for i, (x,y) in enumerate([(-pitch,-pitch),(pitch,-pitch),(-pitch,pitch),(pitch,pitch)]):
        face.add_site(name=f"mount_bolt_{i}", pos=[x,y,0], size=[.0015]*3, rgba=[.9,.7,.1,1])
        face.add_geom(name=f"mount_bolt_shank_{i}", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                      pos=[x,y,thickness/2], size=[.85*mount["hole_diameter_m"]/2,thickness/2,0],
                      density=0, contype=0, conaffinity=0, rgba=[.7,.72,.75,1], group=1)
        face.add_geom(name=f"mount_bolt_head_{i}", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                      pos=[x,y,thickness+.0015], size=[.0042,.0015,0], density=0,
                      contype=0, conaffinity=0, rgba=[.55,.57,.61,1], group=1)
    points = np.array(profile["profile_s_r_m"])
    grip_station = fixture["grip_from_pommel_m"]-dimensions["grip_from_pommel_m"]
    grip_center = np.array(fixture["grip_center_mount_m"])
    grip_quat = fixture["quat_mount_from_sword_wxyz"]
    rotation = _quat_matrix(grip_quat)
    # A short transverse collar grips immediately below the guard. No long
    # pommel-to-face rails or axial sword insertion are used.
    clamp = face.add_body(name="grip_clamp",pos=grip_center,quat=grip_quat)
    radius = float(np.interp(grip_station,points[:,0],points[:,1]))+fixture["collar_clearance_m"]
    v,f = collar_mesh(radius,radius+fixture["collar_radial_wall_m"],fixture["collar_width_m"])
    _add_mesh(spec,"handle_collar_mesh_0",v,f)
    clamp.add_geom(name="handle_collar_visual_0",type=mujoco.mjtGeom.mjGEOM_MESH,
                   meshname="handle_collar_mesh_0",density=0,contype=0,conaffinity=0,
                   rgba=[.12,.23,.35,1],group=1)
    for j in range(16):
        theta = j*2*np.pi/16
        r = radius+fixture["collar_radial_wall_m"]/2
        _collision(clamp,f"handle_collar_collision_0_{j}",mujoco.mjtGeom.mjGEOM_BOX,
                   pos=[r*np.cos(theta),r*np.sin(theta),0],
                   quat=[np.cos(theta/2),0,0,np.sin(theta/2)],
                   size=[fixture["collar_radial_wall_m"]/2,r*np.tan(np.pi/16),fixture["collar_width_m"]/2],
                   contype=TOOL_BIT,conaffinity=ENV_BIT|ARM_BITS)
    bridge_top = grip_center[2]-radius-.0015
    if bridge_top<=thickness:
        raise ValueError("No room for a short support below the transverse clamp")
    geometry = {"pos":[grip_center[0],grip_center[1],.5*(thickness+bridge_top)],
                "size":[fixture["collar_width_m"]/2,fixture["bridge_half_width_m"],.5*(bridge_top-thickness)]}
    face.add_geom(name="clamp_bridge_visual",type=mujoco.mjtGeom.mjGEOM_BOX,
                   density=0,contype=0,conaffinity=0,rgba=[.15,.27,.4,1],group=1,**geometry)
    _collision(face,"clamp_bridge_collision",mujoco.mjtGeom.mjGEOM_BOX,
               contype=TOOL_BIT,conaffinity=ENV_BIT|ARM_BITS,**geometry)
    blade = face.add_body(name="shinai",pos=grip_center-rotation@np.array([0,0,grip_station]),quat=grip_quat)
    blade.mass = sword["mass_kg"]
    blade.ipos = [0,0,sword["com_from_pommel_m"]-dimensions["grip_from_pommel_m"]]
    blade.inertia, blade.explicitinertial = sword["inertia_diagonal_kg_m2"], True
    v, f = revolved_mesh(profile)
    _add_mesh(spec, "shinai_profile_visual_mesh", v, f)
    blade.add_geom(name="shinai_visual", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="shinai_profile_visual_mesh",
                   contype=0, conaffinity=0, density=0, rgba=[.80,.62,.32,1], group=1)
    near, far = dimensions["guard_near_m"], dimensions["guard_far_m"]
    for name, z, color in [("shinai_profile_origin",0,[.6,.3,.9,1]),("shinai_grip",grip_station,[.9,.2,.1,1]),("shinai_tip",dimensions["tip_from_grip_m"],[.2,.8,.3,1]),
                           ("shinai_pommel",-dimensions["grip_from_pommel_m"],[.6,.3,.9,1]),
                           ("shinai_guard",.5*(near+far),[.1,.6,.9,1]),
                           ("shinai_guard_near",near,[.1,.6,.9,1]),("shinai_guard_far",far,[.1,.6,.9,1])]:
        blade.add_site(name=name, pos=[0,0,z], size=[.003]*3, rgba=color)
    # Separate the handle-side flare from the guard's duplicate-s radial step.
    handle_end = np.flatnonzero(np.isclose(points[:,0],near,rtol=0,atol=1e-10))[0]
    handle_points = points[:handle_end+1]
    flare_start = float(handle_points[-2,0])
    handle_bounds = [-dimensions["grip_from_pommel_m"],
                     -dimensions["grip_from_pommel_m"]+.01,0.,flare_start,near]
    for i, (a,b) in enumerate(zip(handle_bounds[:-1],handle_bounds[1:])):
        radius = _interval_radius(handle_points,a,b)
        _collision(blade, f"shinai_handle_collision_{i}", mujoco.mjtGeom.mjGEOM_CYLINDER,
                   pos=[0,0,.5*(a+b)],size=[radius,.5*(b-a),0],
                   contype=TOOL_BIT,conaffinity=ENV_BIT|ARM_BITS)
    _collision(blade, "shinai_guard_collision", mujoco.mjtGeom.mjGEOM_CYLINDER,
               pos=[0,0,.5*(near+far)], size=[dimensions["guard_radius_m"],.5*(far-near),0],
               contype=TOOL_BIT, conaffinity=ENV_BIT|ARM_BITS)
    # Flat-ended adjacent cylinders cover each complete axial interval. Capsules
    # shortened by one radius at each end would touch only at their poles and
    # leave an artificial narrow neck at every segment boundary.
    blade_end = dimensions["tip_from_grip_m"]
    blade_start = np.flatnonzero(np.isclose(points[:,0],far,rtol=0,atol=1e-10))[-1]
    blade_points = points[blade_start:]
    for i, (a,b) in enumerate(zip(np.linspace(far,blade_end,4)[:-1], np.linspace(far,blade_end,4)[1:])):
        radius = _interval_radius(blade_points,a,b)
        _collision(blade, f"shinai_blade_collision_{i}", mujoco.mjtGeom.mjGEOM_CYLINDER,
                   pos=[0,0,.5*(a+b)], size=[radius,.5*(b-a),0],
                   contype=TOOL_BIT, conaffinity=ENV_BIT|ARM_BITS)


def _balanced_home(model):
    """Select a static J0 roll balance; not an executed CAD-to-home path."""
    data = mujoco.MjData(model)
    q = np.zeros(model.nq)
    for _ in range(20):
        data.qpos[:] = q
        mujoco.mj_forward(model,data)
        torque = float(data.qfrc_bias[0])
        if abs(torque)<1e-7:
            return q
        data.qpos[0]+=1e-4
        mujoco.mj_forward(model,data)
        derivative = (data.qfrc_bias[0]-torque)/1e-4
        if abs(derivative)<1e-6:
            break
        q[0] = np.clip(q[0]-np.clip(torque/derivative,-.15,.15),-.6,.6)
    raise ValueError("Could not find a J0 static balance within the initial-pose search window")


@dataclass
class Assembly:
    model: mujoco.MjModel
    data: mujoco.MjData
    config: dict
    home: np.ndarray
    spec: mujoco.MjSpec
    source_hashes: dict


def build_assembly(config=None):
    config = copy.deepcopy(load_assembly_config() if config is None else config)
    validate_config(config)
    hashes = _hash_sources(config)
    from .avatar import shoulder_scene
    spec = shoulder_scene(ROOT,config,ENV_BIT,ARM_BITS,TOOL_BIT)
    original = spec.compile()
    original_data = mujoco.MjData(original)
    mujoco.mj_forward(original,original_data)
    _robot_proxies(spec, original, original_data, config)
    original_limits = [(bool(original.jnt_limited[i]),original.jnt_range[i].copy()) for i in range(7)]
    del original_data, original
    _build_tool(spec,config)
    for geom in spec.geoms:
        if geom.type == mujoco.mjtGeom.mjGEOM_PLANE:
            geom.contype, geom.conaffinity = ENV_BIT, ARM_BITS|TOOL_BIT
    spec.option.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT) | int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)
    spec.option.timestep = config["runtime"]["timestep_s"]
    temporary = spec.compile()
    home = _balanced_home(temporary)
    del temporary
    half = np.array(config["runtime"]["joint_half_windows_rad"])
    caps = np.array(config["runtime"]["torque_caps_nm"])
    for i in range(7):
        joint, actuator = spec.joint(f"J{i}"),spec.actuator(f"J{i}")
        window = np.array([home[i]-half[i],home[i]+half[i]])
        if original_limits[i][0]:
            window = np.array([max(window[0],original_limits[i][1][0]),min(window[1],original_limits[i][1][1])])
        if not window[0] < home[i] < window[1]:
            raise ValueError(f"J{i}: shoulder pose incompatible with existing source limits")
        joint.limited,joint.range = True,window
        actuator.ctrllimited,actuator.ctrlrange = True,window
        actuator.forcelimited,actuator.forcerange = True,[-caps[i],caps[i]]
    spec.add_key(name="assembly_home",qpos=home.tolist(),ctrl=home.tolist())
    model = spec.compile()
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_KEY,"assembly_home")
    mujoco.mj_resetDataKeyframe(model,data,key)
    mujoco.mj_forward(model,data)
    data.ctrl[:] = np.clip(home+data.qfrc_bias/model.actuator_gainprm[:,0],model.actuator_ctrlrange[:,0],model.actuator_ctrlrange[:,1])
    model.key_ctrl[key] = data.ctrl.copy()
    spec.key("assembly_home").ctrl = data.ctrl.tolist()
    mujoco.mj_forward(model,data)
    return Assembly(model,data,config,home,spec,hashes)


def contact_records(model,data):
    return [{"geom1":mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,int(c.geom1)),
             "geom2":mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,int(c.geom2)),
             "distance_m":float(c.dist)} for c in data.contact]


def _finite_json(value):
    """Keep failure evidence serializable even when simulator output is NaN."""
    if isinstance(value, dict):
        return {key:_finite_json(item) for key,item in value.items()}
    if isinstance(value, (list,tuple)):
        return [_finite_json(item) for item in value]
    if isinstance(value, (float,np.floating)) and not np.isfinite(value):
        return None
    return value


def export_scene(assembly,path):
    path = Path(path).resolve()
    if path in {(ROOT/"robot.xml").resolve(),(ROOT/"scene.xml").resolve()} or path.is_relative_to(ROOT/"meshes"):
        raise ValueError("Never overwrite source CAD XML or mesh assets")
    if path.suffix.lower() != ".xml":
        raise ValueError("Export filename must end in .xml")
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(assembly.spec.to_xml(),encoding="utf-8")


def configure_camera(assembly,camera,view="overall"):
    model,data = assembly.model,assembly.data
    if view == "mount":
        site = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,"shinai_grip")
        camera.lookat[:] = data.site_xpos[site]+np.array([0,0,.04])
        camera.distance,camera.azimuth,camera.elevation = .55,135,-12
    elif view == "shoulder":
        site = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,"right_shoulder_mount")
        camera.lookat[:] = data.site_xpos[site]+np.array([0,-.05,0])
        camera.distance,camera.azimuth,camera.elevation = .85,135,-8
    else:
        camera.lookat[:] = [.15,-.65,1.0]
        camera.distance,camera.azimuth,camera.elevation = 3.3,135,-12


def capture_assembly(assembly,path,view="overall"):
    from PIL import Image

    model,data = assembly.model,assembly.data
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    configure_camera(assembly,camera,view)
    model.vis.global_.offwidth,model.vis.global_.offheight = 1100,900
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
    with mujoco.Renderer(model,height=900,width=1100) as renderer:
        renderer.update_scene(data,camera=camera,scene_option=option)
        path = Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        Image.fromarray(renderer.render()).save(path)


def run_assembly_check(assembly,output,duration_s=10.,viewer=None):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    summary_path = output/"summary.json"
    summary_path.write_text(json.dumps({"status":"running"})+"\n")
    model,data,config,home = assembly.model,assembly.data,assembly.config,assembly.home
    runtime = config["runtime"]
    dt,period = model.opt.timestep,runtime["control_period_s"]
    planned_duration = None
    gravity = mujoco.MjData(model)
    kp = model.actuator_gainprm[:,0]
    caps = np.array(runtime["torque_caps_nm"])
    initial_contacts = contact_records(model,data)
    peak_error,peak_speed,peak_torque = np.zeros(7),np.zeros(7),np.zeros(7)
    max_penetration = 0.
    saturation = np.zeros(7,dtype=int)
    failure = None
    sampled = 0
    wall = time.monotonic()
    target = home.copy()
    motion_duration, motion_start, amplitude = None, None, None
    try:
        duration_s = _positive(duration_s,"duration")
        planned_duration = duration_s
        steps,decimation = round(duration_s/dt),round(period/dt)
        if steps < 2 or not np.isclose(steps*dt,duration_s,rtol=0,atol=dt/100):
            raise ValueError("Duration must span at least two steps and be an integer multiple of the timestep")
        if not np.isclose(data.time,0) or not np.allclose(data.qpos,home):
            raise ValueError("Build a fresh assembly before running the check")
        # Short checks still complete an out-and-back; scale amplitude when
        # necessary so shortening a check never raises target speed above its cap.
        motion_start = min(1.,.1*duration_s)
        motion_duration = min(runtime["motion_period_s"],duration_s-2*motion_start)
        amplitude = min(runtime["motion_amplitude_rad"],
                        runtime["max_target_speed_rad_s"]*motion_duration/np.pi)
        with (output/"trajectory.csv").open("w",newline="",encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["time_s"]+[f"J{i}_{field}" for field in ["q_rad","target_rad","torque_nm"] for i in range(7)])
            for step in range(steps+1):
                if step%decimation == 0:
                    phase = np.clip((data.time-motion_start)/motion_duration,0,1)
                    target = home.copy()
                    target[6] += .5*amplitude*(1-np.cos(2*np.pi*phase))
                    gravity.qpos[:],gravity.qvel[:] = data.qpos,0
                    mujoco.mj_forward(model,gravity)
                    data.ctrl[:] = np.clip(target+gravity.qfrc_bias/kp,model.actuator_ctrlrange[:,0],model.actuator_ctrlrange[:,1])
                mujoco.mj_forward(model,data)
                if not all(np.isfinite(a).all() for a in [data.qpos,data.qvel,data.qacc,data.qfrc_actuator]):
                    raise RuntimeError("Nonfinite simulator state")
                if any(w.number for w in data.warning):
                    raise RuntimeError("MuJoCo warning")
                error,speed,torque = np.abs(data.qpos-target),np.abs(data.qvel),np.abs(data.qfrc_actuator)
                peak_error,peak_speed,peak_torque = np.maximum(peak_error,error),np.maximum(peak_speed,speed),np.maximum(peak_torque,torque)
                saturation += torque>=caps*(1-1e-6)
                for contact in data.contact:
                    max_penetration = max(max_penetration,-float(contact.dist))
                if max_penetration>config["collision"]["maximum_allowed_penetration_m"]:
                    raise RuntimeError(f"Collision penetration exceeds configured threshold: {contact_records(model,data)}")
                if error.max()>runtime["max_tracking_error_rad"] or speed.max()>runtime["max_observed_speed_rad_s"]:
                    raise RuntimeError("Tracking error or joint speed exceeds configured threshold")
                if np.any(data.qpos<model.jnt_range[:,0]-.005) or np.any(data.qpos>model.jnt_range[:,1]+.005):
                    raise RuntimeError("Joint left the configured window")
                if np.any(torque>caps+1e-7):
                    raise RuntimeError("Actuator torque cap violated")
                if step%decimation == 0:
                    writer.writerow(np.r_[data.time,data.qpos,target,data.qfrc_actuator])
                    sampled += 1
                    if viewer is not None:
                        if not viewer.is_running():
                            raise RuntimeError("Viewer closed before the check completed")
                        viewer.sync()
                        time.sleep(max(0,wall+data.time-time.monotonic()))
                if step == steps:
                    break
                old_time = data.time
                mujoco.mj_step(model,data)
                if data.time<=old_time:
                    raise RuntimeError("Simulation time failed to advance")
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
    dims = load_profile()["dimensions"]
    site_ids = {name:mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,name) for name in ["mount_face","shinai_grip","shinai_tip","shinai_pommel","shinai_guard"]}
    result = {"status":"failed" if failure else "passed","failure":failure,"scope":"single_arm_fixture_hold_and_small_wrist_motion_not_ASE_or_fencing_validation",
              "parameter_status":config["parameter_status"],"mujoco_version":mujoco.__version__,
              "planned_duration_s":planned_duration,"completed_duration_s":float(data.time),"sample_count":sampled,
              "diagnostic_motion":{"joint":"J6","start_s":motion_start,"duration_s":motion_duration,"amplitude_rad":amplitude},
              "source_sha256":assembly.source_hashes,"config":config,"home_q_rad_J0_to_J6":home.tolist(),
              "initialization":"complete CAD base side-mounted at right shoulder; reset J0 to a model-gravity-balanced angle; CAD-zero-to-home path not validated",
              "shinai_profile_dimensions":dims,
              "actual_grip_from_pommel_m":config["fixture"]["grip_from_pommel_m"],
              "actual_grip_to_tip_m":dims["total_length_m"]-config["fixture"]["grip_from_pommel_m"],
              "sword_and_fixture_added_mass_kg":config["sword"]["mass_kg"]+config["fixture"]["mass_kg"],
              "site_positions_world_m":{name:data.site_xpos[index].tolist() for name,index in site_ids.items()},
              "peak_tracking_error_rad":peak_error.tolist(),"peak_joint_speed_rad_s":peak_speed.tolist(),
              "peak_actuator_torque_nm":peak_torque.tolist(),"saturated_state_evaluations":saturation.tolist(),
              "maximum_collision_penetration_m":max_penetration,"initial_contacts":initial_contacts,
              "warnings":[int(w.number) for w in data.warning],
              "collision_scope":"fixed humanoid primitives, side-mounted arm/base proxies, short hollow clamp and segmented sword; original CAD visuals collision-disabled; parent filtering disabled; adjacent arm proxies excluded by masks only",
              "limitations":["Fixture is not a structural or manufacturing design","Shinai visual is an axial STEP-profile approximation, not a BRep tessellation","Real limits, actuator parameters, inertia, compliance and grip friction are uncalibrated","No ASE replay, opponent, scoring or RL is run by this command"]}
    summary_path.write_text(json.dumps(_finite_json(result),indent=2,allow_nan=False)+"\n",encoding="utf-8")
    if failure:
        raise RuntimeError(f"{failure}; details in {summary_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=DEFAULT_CONFIG)
    parser.add_argument("--output",type=Path,default=ROOT/"outputs/assembly")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--headless",action="store_true")
    mode.add_argument("--build-only",action="store_true",help="Export scene without advancing physics")
    mode.add_argument("--inspect",action="store_true",help="Keep a static inspection viewer open until closed; no physics check")
    parser.add_argument("--duration",type=float)
    parser.add_argument("--snapshot",action="store_true",help="Render overall and close-up PNGs after the check")
    args = parser.parse_args()
    assembly = build_assembly(load_assembly_config(args.config))
    args.output.mkdir(parents=True,exist_ok=True)
    export_scene(assembly,args.output/"scene.xml")
    print("SIMULATION ASSEMBLY: right-shoulder CAD base + transverse grip below guard; unverified fixture and hardware limits.")
    if args.build_only:
        print(f"Built {args.output/'scene.xml'}; load assembly_home keyframe before standalone simulation.")
    elif args.headless:
        result = run_assembly_check(assembly,args.output,args.duration if args.duration is not None else assembly.config["runtime"]["default_check_duration_s"])
        print(f"{result['status'].upper()}: {result['completed_duration_s']:.3f}s")
    else:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(assembly.model,assembly.data) as viewer:
            configure_camera(assembly,viewer.cam)
            viewer.opt.geomgroup[3] = 0
            if args.inspect:
                print("STATIC INSPECTION ONLY: close the viewer to exit; no new physics pass/fail evidence.")
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(.05)
            else:
                run_assembly_check(assembly,args.output,args.duration if args.duration is not None else assembly.config["runtime"]["default_check_duration_s"],viewer=viewer)
    if args.snapshot:
        capture_assembly(assembly,args.output/"overview.png")
        capture_assembly(assembly,args.output/"mount_closeup.png",view="mount")
        capture_assembly(assembly,args.output/"shoulder_closeup.png",view="shoulder")
    print(f"Artifacts: {args.output.resolve()}")


if __name__ == "__main__":
    main()
