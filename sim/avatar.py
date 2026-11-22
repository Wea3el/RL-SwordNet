"""Fixed mannequin torso and a side-mounted, unmodified CAD robot subtree."""
from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


def shoulder_scene(root: Path, config: dict, env_bit: int, arm_bits: int, tool_bit: int):
    """Attach *all* original base geometry and joints to the right shoulder.

    The source's bottom ring is at CAD z=-8 mm, not at the J0 joint origin.
    Its center lands exactly on the outer face of the shoulder adapter plate.
    World gravity, floor and lights are deliberately not transformed.
    """
    scene_xml = ET.parse(root/config["source_scene"]).getroot()
    includes = list(scene_xml.findall("include"))
    if len(includes) != 1 or includes[0].get("file") != "robot.xml":
        raise ValueError("Expected the original scene with one robot.xml include")
    scene_xml.remove(includes[0])
    spec = mujoco.MjSpec.from_string(ET.tostring(scene_xml,encoding="unicode"))
    spec.modelname = "right_shoulder_arm_transverse_shinai"
    spec.meshdir = str(root/"meshes")
    settings = config["torso"]
    torso = spec.worldbody.add_body(name="torso",pos=settings["center_m"])
    scale = settings["body_scale"]

    def shape(name, kind, pos, size, color, *, person=True, **kwargs):
        if person:
            pos, size = np.asarray(pos)*scale, np.asarray(size)*scale
        return torso.add_geom(name=name,type=kind,pos=pos,size=size,rgba=color,
                              density=0,contype=env_bit,conaffinity=arm_bits|tool_bit,
                              group=1,friction=[.4,.005,.0001],**kwargs)

    skin, suit, dark = [.69,.72,.75,1], [.24,.36,.45,1], [.13,.18,.22,1]
    shape("torso_target",mujoco.mjtGeom.mjGEOM_ELLIPSOID,[0,0,.13],[.14,.23,.25],suit)
    shape("torso_abdomen",mujoco.mjtGeom.mjGEOM_ELLIPSOID,[0,0,-.13],[.12,.17,.22],suit)
    shape("torso_hips",mujoco.mjtGeom.mjGEOM_ELLIPSOID,[0,0,-.30],[.13,.18,.10],dark)
    shape("neck",mujoco.mjtGeom.mjGEOM_CYLINDER,[0,0,.425],[.055,.065,0],skin)
    shape("head",mujoco.mjtGeom.mjGEOM_ELLIPSOID,[0,0,.61],[.105,.115,.145],skin)
    # A small front-facing mark makes +X (the fencer's forward direction) clear.
    torso.add_geom(name="face_front",type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                   pos=np.array([.099,0,.63])*scale,size=np.array([.012,.055,.035])*scale,rgba=dark,
                   density=0,contype=0,conaffinity=0,group=1)
    for sign,label in [(-1,"right"),(1,"left")]:
        shape(f"{label}_shoulder",mujoco.mjtGeom.mjGEOM_ELLIPSOID,
              [0,sign*.20,.30],[.11,.08,.11],suit)
    for name, endpoints, radius in [
        ("left_upper_arm",[0,.25,.29,.015,.29,-.025],.055),
        ("left_forearm",[.015,.29,-.025,.05,.29,-.29],.045),
    ]:
        torso.add_geom(name=name,type=mujoco.mjtGeom.mjGEOM_CAPSULE,fromto=np.array(endpoints)*scale,
                       size=[radius*scale,0,0],rgba=suit,density=0,group=1,
                       contype=env_bit,conaffinity=arm_bits|tool_bit)
    shape("left_hand",mujoco.mjtGeom.mjGEOM_ELLIPSOID,[.05,.29,-.34],[.04,.045,.065],skin)
    # Resize the person, not the arm or its mating plates. The pedestal is sized
    # independently to stay on the ground and overlap the bottom of the hips.
    base_z = float(settings["center_m"][2])
    stand_low, stand_high = .08, base_z-.40*scale+.04
    if stand_high<=stand_low:
        raise ValueError("Torso height leaves no room for the ground-mounted stand")
    shape("mannequin_stand",mujoco.mjtGeom.mjGEOM_CYLINDER,
          [0,0,.5*(stand_low+stand_high)-base_z],
          [settings["stand_shaft_radius_m"],.5*(stand_high-stand_low),0],dark,person=False)
    shape("mannequin_foot",mujoco.mjtGeom.mjGEOM_CYLINDER,
          [0,0,.04-base_z],[settings["stand_foot_radius_m"],.04,0],dark,person=False)
    # Put the adapter *outside* the enlarged shoulder instead of letting the
    # larger rounded shoulder swallow the fixed CAD cage and bearing hardware.
    plate_thickness = settings["shoulder_adapter_thickness_m"]
    shoulder = np.array(settings["right_shoulder_mount_local_m"],dtype=float)*scale
    shoulder[1] -= plate_thickness
    shape("shoulder_adapter_plate",mujoco.mjtGeom.mjGEOM_BOX,
          shoulder+[0,plate_thickness/2,0],[.13,plate_thickness/2,.13],[.16,.21,.26,1],person=False)
    torso.add_site(name="right_shoulder_mount",pos=shoulder,
                   quat=settings["quat_world_from_cad_wxyz"],size=[.003]*3,rgba=[.9,.4,.1,1])
    torso.add_site(name="torso_target_center",pos=[0,0,.13*scale],size=[.004]*3,rgba=[1,.2,.2,1])
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation,np.array(settings["quat_world_from_cad_wxyz"]))
    translation = shoulder-rotation.reshape(3,3)@np.array(settings["source_base_contact_center_m"])
    frame = torso.add_frame(name="shoulder_CAD",pos=translation,quat=settings["quat_world_from_cad_wxyz"])
    robot = mujoco.MjSpec.from_file(str(root/"robot.xml"))
    robot.meshdir = str(root/"meshes")
    # An attached unnamed "main" default serializes as an invalid empty class
    # in MuJoCo 3.4. Give the imported defaults a real name before attachment.
    robot.default.name = "cad_robot"
    spec.option.integrator = robot.option.integrator
    # Open base proxy in CAD coordinates, transformed with the real fixed base.
    robot.worldbody.add_geom(name="base_collision",type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0,0,.004],size=[.12,.12,.004],density=0,group=3,rgba=[.2,.8,.4,0],
        contype=env_bit,conaffinity=arm_bits|tool_bit)
    for i,(x,y) in enumerate([(-.11,-.11),(-.11,.11),(.11,-.11),(.11,.11)]):
        robot.worldbody.add_geom(name=f"base_post_collision_{i}",type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[x,y,.075],size=[.01,.01,.075],density=0,group=3,rgba=[.2,.8,.4,0],
            contype=env_bit,conaffinity=arm_bits|tool_bit)
    spec.attach(robot,prefix="",suffix="",frame=frame)
    return spec
