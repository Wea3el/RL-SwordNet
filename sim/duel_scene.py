"""Two unchanged, fixed-floor CAD arms with real cross-robot proxy contacts.

No actions, scoring logic, scripted deflection, or hardware limits are supplied
here. The visible target sites are virtual test fixtures, not physical bodies.
The default +/-pi ranges and inherited motor caps are exploratory debug
assumptions. A contact-enabled scene is not a validated attack/parry sequence.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import mujoco
import numpy as np

from .assembly import ARM_BITS, ENV_BIT, TOOL_BIT, ROOT
from .floor import build_floor_assembly


AGENT_BIT_STRIDE = 10
FLOOR_BIT = 1 << 20
SOURCE_COLLISION_BITS = ARM_BITS | ENV_BIT | TOOL_BIT
SITE_NAMES = ("floor_mount", "mount_face", "shinai_grip", "shinai_tip",
              "shinai_pommel", "shinai_guard", "shinai_guard_near", "shinai_guard_far")


@dataclass
class DuelScene:
    model: mujoco.MjModel
    data: mujoco.MjData
    spec: mujoco.MjSpec
    config: dict
    source_hashes: dict
    agents: dict


def _id(model, kind, name):
    result = mujoco.mj_name2id(model, kind, name)
    if result < 0:
        raise ValueError(f"Missing {kind.name}: {name}")
    return int(result)


def _agent_spec(floor, index):
    child = floor.spec.copy()
    for geom in list(child.geoms):
        if geom.type == mujoco.mjtGeom.mjGEOM_PLANE:
            child.delete(geom)
    for light in list(child.lights):
        child.delete(light)
    for key in list(child.keys):
        child.delete(key)
    # Namespaces keep the original self-adjacency mask policy independent for
    # each arm. All OTHER-agent collidable proxies are added to each affinity,
    # including blade-vs-blade, rather than prescribing a contact response.
    offset = index * AGENT_BIT_STRIDE
    other_bits = SOURCE_COLLISION_BITS << ((1-index) * AGENT_BIT_STRIDE)
    for geom in child.geoms:
        if not (geom.contype or geom.conaffinity):
            continue  # original visual CAD meshes remain visual-only
        if (int(geom.contype) | int(geom.conaffinity)) & ~SOURCE_COLLISION_BITS:
            raise ValueError("Unexpected source collision bits")
        geom.contype = int(geom.contype) << offset
        geom.conaffinity = (int(geom.conaffinity) << offset) | other_bits | FLOOR_BIT
    # Attached main defaults must have an actual name in MuJoCo 3.4.
    child.default.name = "floor_scene_defaults"
    return child


def build_duel_scene(base_distance_m=1.6, joint_ranges=None):
    """Build a1 at x=0,yaw=0 and a2 at x=D,yaw=pi, no human torso.

    The floor builder already lifts the source CAD by 8mm, so the frames added
    here have z=0. Both bases are welded to the world. ``joint_ranges`` is a
    common (7,2) radian array, intersected with source limits by floor.py; None
    deliberately requests explicitly hypothetical +/-pi, not real motor limits.
    """
    if isinstance(base_distance_m, (bool, str)) or not np.isfinite(base_distance_m) or base_distance_m <= .24:
        raise ValueError("base_distance_m must be finite and greater than the .24m base width")
    ranges = np.tile([-np.pi, np.pi], (7, 1)) if joint_ranges is None else joint_ranges
    floor = build_floor_assembly(ranges)
    spec = mujoco.MjSpec.from_string('''<mujoco model="two_fixed_floor_arms_shinai">
      <visual><headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3"/></visual>
      <worldbody><light name="duel_light" pos="0 0 3" dir="0 0 -1" directional="true"/>
      <geom name="floor" type="plane" size="0 0 .05" rgba=".22 .25 .28 1"/></worldbody>
    </mujoco>''')
    spec.meshdir = str(ROOT / "meshes")
    spec.option.integrator = floor.spec.option.integrator
    spec.option.timestep = floor.spec.option.timestep
    spec.option.disableflags = floor.spec.option.disableflags
    spec.option.gravity = floor.spec.option.gravity
    ground = spec.geom("floor")
    ground.contype = FLOOR_BIT
    ground.conaffinity = SOURCE_COLLISION_BITS | (SOURCE_COLLISION_BITS << AGENT_BIT_STRIDE)
    placements = {"a1": {"position_m": [0., 0., 0.], "quaternion_wxyz": [1., 0., 0., 0.]},
                  "a2": {"position_m": [float(base_distance_m), 0., 0.], "quaternion_wxyz": [0., 0., 0., 1.]}}
    for index, (name, placement) in enumerate(placements.items()):
        # Attach the native CAD frame to a fixed BODY, not to a second frame:
        # MuJoCo 3.4's spec.attach through nested imported frames can fail to
        # propagate the outer placement. No XML serialization/reparse is used,
        # because that would round source inertial and geometry parameters.
        fixed = spec.worldbody.add_body(name=f"{name}_fixed_base", pos=placement["position_m"],
                                       quat=placement["quaternion_wxyz"])
        child = _agent_spec(floor, index)
        fixed.attach_frame(child.frame("floor_CAD"), prefix=f"{name}_", suffix="")
        # The only non-CAD floor child needed from outside that frame is this
        # virtual base-reference site. Ground and lights are deliberately not
        # duplicated. It adds neither mass nor a collision geom.
        fixed.add_site(name=f"{name}_floor_mount", pos=[0, 0, 0], size=[.003]*3,
                       rgba=[.9, .4, .1, 1])
    for name, x in (("a1", .32), ("a2", float(base_distance_m)-.32)):
        spec.worldbody.add_site(name=f"{name}_target", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                               pos=[x, 0, 1.05], size=[.06, 0, 0], rgba=[.9, .25, .2, .35])
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    agents = {}
    for name in placements:
        prefix = name + "_"
        joints = np.array([_id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix+f"J{i}") for i in range(7)], dtype=int)
        actuators = np.array([_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, prefix+f"J{i}") for i in range(7)], dtype=int)
        agents[name] = {"prefix": prefix, "joint_ids": joints,
            "qpos_ids": model.jnt_qposadr[joints].copy(), "dof_ids": model.jnt_dofadr[joints].copy(),
            "actuator_ids": actuators,
            "sites": {site: _id(model, mujoco.mjtObj.mjOBJ_SITE, prefix+site) for site in SITE_NAMES},
            "placement": copy.deepcopy(placements[name])}
    # Initial CAD zero only; this does not validate a route to a future guard.
    for mapping in agents.values():
        ids, dofs = mapping["actuator_ids"], mapping["dof_ids"]
        data.ctrl[ids] = np.clip(data.qfrc_bias[dofs]/model.actuator_gainprm[ids, 0],
                                model.actuator_ctrlrange[ids, 0], model.actuator_ctrlrange[ids, 1])
    mujoco.mj_forward(model, data)
    config = {"status": "dual_floor_proxy_contact_scene_not_validated_actions_or_hardware",
        "base_distance_m": float(base_distance_m), "base_placements": placements,
        "floor_assembly_config": copy.deepcopy(floor.config),
        "initial_pose": "both CAD zero reset; guard and reset travel not validated",
        "joint_range_status": "hypothetical_ranges_not_identified_hardware_limits",
        "virtual_targets": {"status": "scoring_and_visual_test_fixtures_only_no_collision_or_mass",
            "a1": {"center_world_m": [.32, 0, 1.05], "radius_m": .06, "site_name": "a1_target"},
            "a2": {"center_world_m": [float(base_distance_m)-.32, 0, 1.05], "radius_m": .06, "site_name": "a2_target"}},
        "collision": {"per_agent_bit_stride": AGENT_BIT_STRIDE, "floor_bit": FLOOR_BIT,
            "self_policy": "original floor proxy adjacency masks unchanged within each independent namespace",
            "cross_policy": "all collidable proxies across agents, including both blades; original CAD visuals remain collision-disabled",
            "contact_response": "native MuJoCo rigid contact solver only; no scripted deflection",
            "legal_contact_class": "cross_blade_blade", "scoring_targets": "virtual sites only; scoring logic not implemented"},
        "limitations": ["Debug torque caps and actuator gains are inherited unchanged, not measured motor specifications",
            "Joint ranges, rigid fixture, sword inertias and coarse collision geometry remain assumptions",
            "No attack, successful strike, block, control speed, or safety is established by building this scene"]}
    return DuelScene(model, data, spec, config, copy.deepcopy(floor.source_hashes), agents)


def classify_contact(model, contact, *, rules_version="tip_attack_v1"):
    """Classify physical geom pairs; this does not award points or alter forces."""
    if rules_version not in {"tip_attack_v1", "whole_blade_first_touch_v2"}:
        raise ValueError("Unknown contact rules version")
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(i)) or "" for i in (contact.geom1, contact.geom2)]
    if "floor" in names:
        return "illegal_floor_contact"
    owners = [name.split("_", 1)[0] if name.startswith(("a1_", "a2_")) else None for name in names]
    if owners[0] != owners[1] and all(owner in ("a1", "a2") for owner in owners):
        if all("_shinai_blade_collision_" in name for name in names):
            return "cross_blade_blade"
        if rules_version == "whole_blade_first_touch_v2" and (
            any("_shinai_blade_collision_" in name for name in names)
            and any(name.endswith("_shinai_guard_collision") for name in names)
        ):
            return "cross_blade_guard"
        return "illegal_cross_robot_nonblade_contact"
    return "illegal_self_contact"


def duel_contact_records(scene, *, rules_version="tip_attack_v1"):
    """Include solver contact forces as evidence, not an invented collision flag."""
    model, data = scene.model, scene.data
    result = []
    for i, contact in enumerate(data.contact):
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force)
        result.append({"geom1": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)),
            "geom2": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)),
            "category": classify_contact(model, contact, rules_version=rules_version), "distance_m": float(contact.dist),
            "normal_force_n": float(force[0]), "contact_force_local": force.tolist()})
    return result
