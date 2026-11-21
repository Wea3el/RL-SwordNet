"""Geometry, collision eligibility, and reversible assembly checks.

These checks validate the configured simulation assembly. They do not establish
measured motor limits, real clamping strength, or safe human contact.
"""

from __future__ import annotations

import hashlib
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np

from sim.assembly import (
    DEFAULT_CONFIG,
    build_assembly,
    export_scene,
    load_assembly_config,
    run_assembly_check,
)


ROOT = Path(__file__).resolve().parents[1]
MEASURED_MOUNT_POS = np.array([.025, -.01091223666, .02249289146])
MEASURED_MOUNT_QUAT = np.array([.97460038477, .22395108842, 0., 0.])
MEASURED_HOLE_PITCH = .02828427
DEBUG_CAPS = np.array([5., 80., 15., 20., 3., 3., 2.])


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quaternion_matrix(q):
    result = np.empty(9)
    q = np.asarray(q, dtype=float)
    mujoco.mju_quat2Mat(result, q / np.linalg.norm(q))
    return result.reshape(3, 3)


def object_id(model, kind, name):
    index = mujoco.mj_name2id(model, kind, name)
    if index < 0:
        raise AssertionError(f"Missing {kind.name} named {name}")
    return index


class AssemblyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source_hashes_before = {name: checksum(ROOT / name) for name in ["robot.xml", "scene.xml"]}
        cls.config = load_assembly_config()
        cls.assembly = build_assembly(cls.config)
        cls.model = cls.assembly.model
        cls.data = cls.assembly.data
        cls.dimensions = json.loads((ROOT / "assets/shinai_profile.json").read_text())["dimensions"]
        cls.original_model = mujoco.MjModel.from_xml_path(str(ROOT / "scene.xml"))
        cls.original_mass = float(cls.original_model.body_mass.sum())

    def site_position(self, name, data=None):
        site = object_id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        return (self.data if data is None else data).site_xpos[site].copy()

    def test_measured_mount_frame_and_four_hole_square(self):
        m, d = self.model, self.data
        ee = object_id(m, mujoco.mjtObj.mjOBJ_BODY, "end_effector_block")
        face = object_id(m, mujoco.mjtObj.mjOBJ_SITE, "mount_face")
        ee_rotation = d.xmat[ee].reshape(3, 3)
        face_rotation = d.site_xmat[face].reshape(3, 3)
        local_position = ee_rotation.T @ (d.site_xpos[face] - d.xpos[ee])
        local_rotation = ee_rotation.T @ face_rotation
        np.testing.assert_allclose(local_position, MEASURED_MOUNT_POS, atol=1e-10)
        # The measured full-precision frame has tiny (~1e-8) out-of-plane
        # quaternion components; the independently recorded values are rounded.
        np.testing.assert_allclose(local_rotation, quaternion_matrix(MEASURED_MOUNT_QUAT), atol=1e-7)
        holes = np.array([self.site_position(f"mount_bolt_{i}") for i in range(4)])
        local_holes = (holes - d.site_xpos[face]) @ face_rotation
        np.testing.assert_allclose(local_holes.mean(axis=0), 0., atol=1e-9)
        np.testing.assert_allclose(local_holes[:, 2], 0., atol=1e-9)
        np.testing.assert_allclose(np.abs(local_holes[:, :2]), MEASURED_HOLE_PITCH / 2, atol=1e-8)
        pair_lengths = sorted(np.linalg.norm(holes[i] - holes[j]) for i in range(4) for j in range(i))
        np.testing.assert_allclose(pair_lengths, [MEASURED_HOLE_PITCH] * 4 + [MEASURED_HOLE_PITCH * np.sqrt(2)] * 2, atol=2e-8)
        j6 = object_id(m, mujoco.mjtObj.mjOBJ_JOINT, "J6")
        self.assertLess(abs(float(face_rotation[:, 2] @ d.xaxis[j6])), 1e-7)

    def test_sword_dimensions_and_actual_clamp_below_the_guard(self):
        dims = self.dimensions
        grip, tip = self.site_position("shinai_grip"), self.site_position("shinai_tip")
        pommel = self.site_position("shinai_pommel")
        profile_origin = self.site_position("shinai_profile_origin")
        axis = (tip-pommel)/np.linalg.norm(tip-pommel)
        self.assertAlmostEqual(float(np.linalg.norm(tip-pommel)), dims["total_length_m"], places=9)
        np.testing.assert_allclose(profile_origin-pommel, dims["grip_from_pommel_m"]*axis, atol=1e-9)
        self.assertGreater(float((grip-profile_origin) @ axis), .1,
                           "Physical grip incorrectly remains at the CAD handle midpoint")
        self.assertAlmostEqual(float((grip-pommel) @ axis), .29, places=9)
        np.testing.assert_allclose(grip-pommel, .29*axis, atol=1e-9)
        near = self.site_position("shinai_guard_near")
        far = self.site_position("shinai_guard_far")
        middle = self.site_position("shinai_guard")
        np.testing.assert_allclose(near-pommel, dims["handle_length_m"]*axis, atol=1e-9)
        np.testing.assert_allclose(far-profile_origin, dims["guard_far_m"]*axis, atol=1e-9)
        np.testing.assert_allclose(middle, .5 * (near + far), atol=1e-9)
        self.assertGreater(float((near-grip) @ axis), .0125,
                           "The physical collar reaches or crosses the guard")
        face = object_id(self.model, mujoco.mjtObj.mjOBJ_SITE, "mount_face")
        frame = self.data.site_xmat[face].reshape(3, 3)
        self.assertLess(abs(float(frame[:, 2] @ axis)), 1e-9)
        self.assertGreater(float(frame[:, 0] @ axis), 1-1e-9)
        actual_center = frame.T @ (grip-self.data.site_xpos[face])
        np.testing.assert_allclose(actual_center, self.config["fixture"]["grip_center_mount_m"], atol=1e-9)

    def test_short_fixture_has_one_near_guard_collar_and_no_long_rails(self):
        m, d = self.model, self.data
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(m.ngeom)]
        self.assertFalse(any(name.startswith("clamp_rail_") for name in names))
        self.assertEqual([name for name in names if name.startswith("handle_collar_visual_")],
                         ["handle_collar_visual_0"])
        collar = object_id(m, mujoco.mjtObj.mjOBJ_GEOM, "handle_collar_visual_0")
        self.assertEqual(m.geom_type[collar], mujoco.mjtGeom.mjGEOM_MESH)
        grip = self.site_position("shinai_grip")
        tip = self.site_position("shinai_tip")
        axis = (tip-grip)/np.linalg.norm(tip-grip)
        mesh = int(m.geom_dataid[collar])
        begin, count = int(m.mesh_vertadr[mesh]), int(m.mesh_vertnum[mesh])
        vertices = m.mesh_vert[begin:begin+count]
        world = vertices @ d.geom_xmat[collar].reshape(3, 3).T + d.geom_xpos[collar]
        collar_s = (world-grip) @ axis
        half_width = self.config["fixture"]["collar_width_m"]/2
        np.testing.assert_allclose([collar_s.min(), collar_s.max()], [-half_width, half_width], atol=1e-8)
        guard_s = float((self.site_position("shinai_guard_near")-grip) @ axis)
        self.assertGreater(guard_s-float(collar_s.max()), .001)

        fixture = object_id(m, mujoco.mjtObj.mjOBJ_BODY, "sword_fixture")
        sword = object_id(m, mujoco.mjtObj.mjOBJ_BODY, "shinai")

        def descendant(body, ancestor):
            while body:
                if body == ancestor:
                    return True
                body = int(m.body_parentid[body])
            return False

        geoms = [i for i in range(m.ngeom) if descendant(int(m.geom_bodyid[i]), fixture)
                 and not descendant(int(m.geom_bodyid[i]), sword)]
        self.assertTrue(geoms)
        # Conservative bounding spheres suffice for this compactness check.
        # The prior 30cm longitudinal rails cannot pass it, even if renamed.
        centers = np.array([float((d.geom_xpos[i]-grip) @ axis) for i in geoms])
        bounds = m.geom_rbound[geoms]
        self.assertLess(float((centers+bounds).max()-(centers-bounds).min()), .14)

        face = object_id(m, mujoco.mjtObj.mjOBJ_SITE, "mount_face")
        frame, origin = d.site_xmat[face].reshape(3, 3), d.site_xpos[face]

        def box_bounds(name):
            geom = object_id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
            self.assertEqual(m.geom_type[geom], mujoco.mjtGeom.mjGEOM_BOX)
            rotation = frame.T @ d.geom_xmat[geom].reshape(3, 3)
            np.testing.assert_allclose(np.sort(np.abs(rotation), axis=0), [[0.]*3, [0.]*3, [1.]*3], atol=1e-9)
            center = frame.T @ (d.geom_xpos[geom]-origin)
            extent = np.abs(rotation) @ m.geom_size[geom]
            return center-extent, center+extent

        bridge_low, bridge_high = box_bounds("clamp_bridge_collision")
        visual_low, visual_high = box_bounds("clamp_bridge_visual")
        np.testing.assert_allclose(bridge_low, visual_low, atol=1e-12)
        np.testing.assert_allclose(bridge_high, visual_high, atol=1e-12)
        plate_low, plate_high = box_bounds("adapter_plate_collision")
        # A shared mounting face is allowed, but no unsupported normal gap.
        self.assertLessEqual(float(bridge_low[2]), float(plate_high[2])+1e-9)
        self.assertTrue(np.all(np.minimum(bridge_high[:2], plate_high[:2])
                               - np.maximum(bridge_low[:2], plate_low[:2]) > 1e-5))
        collar_low, collar_high = box_bounds("handle_collar_collision_0_0")
        overlap = np.minimum(bridge_high, collar_high)-np.maximum(bridge_low, collar_low)
        self.assertTrue(np.all(overlap > 1e-5), "Short bridge is disconnected from the collar")

    def test_right_shoulder_mount_transforms_the_entire_original_CAD(self):
        m, d, original = self.model, self.data, self.original_model
        self.assertEqual(self.config["schema_version"], 2)
        torso = object_id(m, mujoco.mjtObj.mjOBJ_BODY, "torso")
        shoulder = object_id(m, mujoco.mjtObj.mjOBJ_SITE, "right_shoulder_mount")
        chest = object_id(m, mujoco.mjtObj.mjOBJ_GEOM, "torso_target")
        self.assertEqual(int(m.body_weldid[torso]), 0)
        self.assertEqual(int(m.site_bodyid[shoulder]), torso)
        self.assertEqual(m.geom_type[chest], mujoco.mjtGeom.mjGEOM_ELLIPSOID)
        for name in ["head", "neck", "left_upper_arm", "left_forearm", "left_hand",
                     "mannequin_stand", "mannequin_foot"]:
            geom = object_id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
            self.assertEqual(int(m.body_weldid[m.geom_bodyid[geom]]), 0)
            self.assertTrue(m.geom_contype[geom] or m.geom_conaffinity[geom])
        head = object_id(m, mujoco.mjtObj.mjOBJ_GEOM, "head")
        left_arm = object_id(m, mujoco.mjtObj.mjOBJ_GEOM, "left_upper_arm")
        foot = object_id(m, mujoco.mjtObj.mjOBJ_GEOM, "mannequin_foot")
        self.assertGreater(float(d.geom_xpos[head, 2]), float(d.geom_xpos[chest, 2]))
        self.assertGreater(float(d.geom_xpos[left_arm, 1]), float(d.geom_xpos[chest, 1]))
        self.assertAlmostEqual(float(d.geom_xpos[foot, 2]-m.geom_size[foot, 1]), 0., places=9)
        shoulder_world = d.site_xpos[shoulder]
        self.assertLess(float(shoulder_world[1]), float(d.geom_xpos[chest, 1]-m.geom_size[chest, 1]))
        self.assertGreater(float(shoulder_world[2]), float(d.geom_xpos[chest, 2]))
        np.testing.assert_allclose(d.xmat[torso].reshape(3, 3).T @ (shoulder_world-d.xpos[torso]),
                                   self.config["torso"]["right_shoulder_mount_local_m"], atol=1e-9)
        rotation = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
        np.testing.assert_allclose(quaternion_matrix(self.config["torso"]["quat_world_from_cad_wxyz"]), rotation, atol=1e-9)
        source_contact = np.array([0., 0., -.008])
        np.testing.assert_allclose(self.config["torso"]["source_base_contact_center_m"], source_contact, atol=1e-12)
        translation = shoulder_world-rotation @ source_contact
        source_data = mujoco.MjData(original)
        for i in range(7):
            joint = object_id(original, mujoco.mjtObj.mjOBJ_JOINT, f"J{i}")
            source_data.qpos[original.jnt_qposadr[joint]] = self.assembly.home[i]
        mujoco.mj_forward(original, source_data)
        fixed = [i for i in range(original.ngeom) if int(original.geom_bodyid[i]) == 0
                 and original.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH]
        self.assertEqual(len(fixed), 25)
        source_mesh_names = {mujoco.mj_id2name(original, mujoco.mjtObj.mjOBJ_MESH, int(original.geom_dataid[i])) for i in fixed}
        candidates = [i for i in range(m.ngeom) if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH
                      and int(m.body_weldid[m.geom_bodyid[i]]) == 0
                      and mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, int(m.geom_dataid[i])) in source_mesh_names]
        self.assertEqual(len(candidates), 25, "Original fixed CAD parts were lost or duplicated")
        for old_geom in fixed:
            mesh_name = mujoco.mj_id2name(original, mujoco.mjtObj.mjOBJ_MESH, int(original.geom_dataid[old_geom]))
            expected_position = translation+rotation @ source_data.geom_xpos[old_geom]
            matching = [i for i in candidates
                        if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, int(m.geom_dataid[i])) == mesh_name]
            self.assertTrue(matching, f"Missing fixed CAD mesh {mesh_name}")
            selected = min(matching, key=lambda i: np.linalg.norm(d.geom_xpos[i]-expected_position))
            np.testing.assert_allclose(d.geom_xpos[selected], expected_position, rtol=0, atol=1e-7)
            np.testing.assert_allclose(d.geom_xmat[selected].reshape(3, 3),
                                       rotation @ source_data.geom_xmat[old_geom].reshape(3, 3), atol=1e-6)
            candidates.remove(selected)
        self.assertFalse(candidates)
        for i in range(7):
            old_joint = object_id(original, mujoco.mjtObj.mjOBJ_JOINT, f"J{i}")
            new_joint = object_id(m, mujoco.mjtObj.mjOBJ_JOINT, f"J{i}")
            old_body, new_body = int(original.jnt_bodyid[old_joint]), int(m.jnt_bodyid[new_joint])
            np.testing.assert_allclose(d.xpos[new_body], translation+rotation @ source_data.xpos[old_body], rtol=0, atol=1e-7)
            np.testing.assert_allclose(d.xmat[new_body].reshape(3, 3), rotation @ source_data.xmat[old_body].reshape(3, 3), atol=1e-6)
            np.testing.assert_allclose(d.xanchor[new_joint], translation+rotation @ source_data.xanchor[old_joint], rtol=0, atol=1e-7)
            np.testing.assert_allclose(d.xaxis[new_joint], rotation @ source_data.xaxis[old_joint], atol=1e-6)
            ancestor = new_body
            while ancestor and ancestor != torso:
                ancestor = int(m.body_parentid[ancestor])
            self.assertEqual(ancestor, torso, "A robot link is not attached below the fixed torso")

    def test_blade_cylinders_are_gapless_and_end_at_the_cad_tip(self):
        m, d, dims = self.model, self.data, self.dimensions
        origin, tip = self.site_position("shinai_profile_origin"), self.site_position("shinai_tip")
        axis = (tip-origin)/np.linalg.norm(tip-origin)
        profile = np.asarray(json.loads((ROOT/"assets/shinai_profile.json").read_text())["profile_s_r_m"])
        # guard_far has two radii at its discontinuity. The final row is the
        # blade side; the larger guard-side radius is not part of the blade.
        start = np.flatnonzero(np.isclose(profile[:, 0], dims["guard_far_m"], rtol=0, atol=1e-12))[-1]
        blade_profile = profile[start:]
        intervals, radii = [], []
        for i in range(3):
            geom = object_id(m, mujoco.mjtObj.mjOBJ_GEOM, f"shinai_blade_collision_{i}")
            self.assertEqual(m.geom_type[geom], mujoco.mjtGeom.mjGEOM_CYLINDER,
                             "Rounded capsule endpoints leave a zero-area neck between segments")
            direction = d.geom_xmat[geom].reshape(3, 3)[:, 2]
            self.assertGreater(abs(float(direction @ axis)), 1-1e-9)
            displacement = d.geom_xpos[geom]-origin
            center = float(displacement @ axis)
            np.testing.assert_allclose(displacement, center*axis, atol=1e-9)
            half_length = float(m.geom_size[geom, 1])
            self.assertGreater(half_length, 0.)
            a, b = center-half_length, center+half_length
            radius = float(m.geom_size[geom, 0])
            self.assertGreater(radius, 1e-4)
            interior = blade_profile[(blade_profile[:, 0] > a+1e-10) & (blade_profile[:, 0] < b-1e-10), 1]
            expected_radius = max(np.max(np.interp([a, b], blade_profile[:, 0], blade_profile[:, 1])),
                                  float(interior.max()) if interior.size else 0.)
            self.assertGreaterEqual(radius, expected_radius-1e-9,
                                    "Collision cylinder misses a profile peak between uniform samples")
            intervals.append((a, b))
            radii.append(radius)
        self.assertAlmostEqual(intervals[0][0], dims["guard_far_m"], places=9)
        for i in range(2):
            self.assertAlmostEqual(intervals[i][1], intervals[i+1][0], places=9)
            self.assertGreater(min(radii[i], radii[i+1]), 1e-4)
        self.assertAlmostEqual(intervals[-1][1], dims["tip_from_grip_m"], places=9)

    def test_fixed_tool_and_target_add_mass_without_new_dofs(self):
        m, d = self.model, self.data
        self.assertEqual((m.nq, m.nv, m.nu), (7, 7, 7))
        self.assertEqual(np.asarray(self.assembly.home).shape, (7,))
        np.testing.assert_allclose(d.qpos, self.assembly.home, atol=1e-12)
        self.assertFalse(np.any(m.jnt_type == mujoco.mjtJoint.mjJNT_FREE))
        ee = object_id(m, mujoco.mjtObj.mjOBJ_BODY, "end_effector_block")
        shinai = object_id(m, mujoco.mjtObj.mjOBJ_BODY, "shinai")
        fixture = object_id(m, mujoco.mjtObj.mjOBJ_BODY, "sword_fixture")
        for body in [shinai, fixture]:
            self.assertEqual(int(m.body_jntnum[body]), 0)
            self.assertEqual(int(m.body_weldid[body]), int(m.body_weldid[ee]))
        self.assertAlmostEqual(float(m.body_mass[shinai]), .436, places=9)
        self.assertGreater(float(m.body_mass[fixture]), 0.)
        self.assertGreater(float(m.body_mass.sum()), self.original_mass + .436)
        target = object_id(m, mujoco.mjtObj.mjOBJ_GEOM, "torso_target")
        self.assertEqual(int(m.body_weldid[m.geom_bodyid[target]]), 0)
        self.assertTrue(m.geom_contype[target] or m.geom_conaffinity[target])
        for joint in range(m.njnt):
            self.assertTrue(m.jnt_limited[joint])
            self.assertTrue(np.isfinite(m.jnt_range[joint]).all())
            self.assertLess(m.jnt_range[joint, 0], m.jnt_range[joint, 1])
            self.assertGreaterEqual(d.qpos[m.jnt_qposadr[joint]], m.jnt_range[joint, 0] - 1e-10)
            self.assertLessEqual(d.qpos[m.jnt_qposadr[joint]], m.jnt_range[joint, 1] + 1e-10)
        for jid in range(7):
            aid = object_id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"J{jid}")
            self.assertTrue(m.actuator_forcelimited[aid])
            self.assertTrue(m.actuator_ctrllimited[aid])
            np.testing.assert_allclose(m.actuator_forcerange[aid], [-DEBUG_CAPS[jid], DEBUG_CAPS[jid]], atol=1e-12)
            self.assertEqual(int(m.actuator_trnid[aid, 0]), object_id(m, mujoco.mjtObj.mjOBJ_JOINT, f"J{jid}"))

    def test_forward_home_and_rigid_tool_transform_after_joint_motion(self):
        m, home = self.model, np.asarray(self.assembly.home)
        vector = self.site_position("shinai_tip") - self.site_position("shinai_grip")
        vector /= np.linalg.norm(vector)
        self.assertGreater(float(vector[0]), .8)
        self.assertLess(abs(float(vector[2])), .6)
        ee = object_id(m, mujoco.mjtObj.mjOBJ_BODY, "end_effector_block")
        reference = {}
        for name in ["shinai_profile_origin", "shinai_grip", "shinai_guard", "shinai_tip"]:
            reference[name] = self.data.xmat[ee].reshape(3, 3).T @ (self.site_position(name) - self.data.xpos[ee])
        probe = mujoco.MjData(m)
        for offset in [-.02, .02]:
            probe.qpos[:] = home
            for name in ["J5", "J6"]:
                joint = object_id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
                probe.qpos[m.jnt_qposadr[joint]] += offset
            mujoco.mj_forward(m, probe)
            for name, local in reference.items():
                measured = probe.xmat[ee].reshape(3, 3).T @ (self.site_position(name, probe) - probe.xpos[ee])
                np.testing.assert_allclose(measured, local, atol=1e-9)

    def test_collision_proxies_enabled_and_home_has_no_penetration(self):
        m, d = self.model, self.data
        proxies = [i for i in range(m.ngeom)
                   if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("arm_proxy_")]
        self.assertGreaterEqual(len(proxies), 2)
        self.assertTrue(all(m.geom_contype[i] or m.geom_conaffinity[i] for i in proxies))
        for contact in d.contact:
            self.assertGreaterEqual(float(contact.dist), -1e-7,
                                    f"Initial penetration between geom {contact.geom1} and {contact.geom2}")

    def test_tool_wrist_parent_contact_is_not_silently_filtered(self):
        # A deliberately overlapping pair in a fresh in-memory test model
        # verifies actual pair eligibility, including exclusions/parent filters.
        a = build_assembly()
        m, d = a.model, a.data
        ee = object_id(m, mujoco.mjtObj.mjOBJ_BODY, "end_effector_block")
        tool_weld = int(m.body_weldid[ee])
        wrist_weld = int(m.body_weldid[m.body_parentid[ee]])
        enabled = lambda i: bool(m.geom_contype[i] or m.geom_conaffinity[i])
        tool = [i for i in range(m.ngeom)
                if int(m.body_weldid[m.geom_bodyid[i]]) == tool_weld and enabled(i)
                and not (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("arm_proxy_")]
        wrist = [i for i in range(m.ngeom)
                 if int(m.body_weldid[m.geom_bodyid[i]]) == wrist_weld and enabled(i)
                 and (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("arm_proxy_")]
        self.assertTrue(tool, "No enabled collision geom on the rigid sword/tool group")
        self.assertTrue(wrist, "No arm_proxy collision geom on the J6 parent/wrist group")
        tool_geom, wrist_geom = tool[0], wrist[0]
        body = int(m.geom_bodyid[tool_geom])
        m.geom_pos[tool_geom] = d.xmat[body].reshape(3, 3).T @ (d.geom_xpos[wrist_geom] - d.xpos[body])
        mujoco.mj_forward(m, d)
        expected = {tool_geom, wrist_geom}
        self.assertTrue(any({int(c.geom1), int(c.geom2)} == expected for c in d.contact),
                        "Tool versus wrist-parent overlap was silently removed by collision filtering")

    def test_torso_contact_with_the_moving_arm_and_sword_is_enabled(self):
        a = build_assembly()
        m, d = a.model, a.data
        chest = object_id(m, mujoco.mjtObj.mjOBJ_GEOM, "torso_target")
        self.assertEqual(int(m.body_weldid[m.geom_bodyid[chest]]), 0)
        for name in ["arm_proxy_3", "shinai_blade_collision_1"]:
            geom = object_id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
            body = int(m.geom_bodyid[geom])
            self.assertNotEqual(int(m.body_weldid[body]), 0)
            # Recompile the probe so its body's BVH reflects the new geom pose.
            # Avoid exactly coincident centers: convex GJK's degenerate initial
            # simplex can report zero distance for a deeply embedded primitive.
            location = (d.geom_xpos[chest]+d.geom_xmat[chest].reshape(3, 3)
                        @ np.array([.8*m.geom_size[chest, 0], 0., 0.]))
            probe_spec = a.spec.copy()
            probe_geom = probe_spec.geom(name)
            probe_geom.fromto = np.full(6, np.nan)
            probe_geom.pos = d.xmat[body].reshape(3, 3).T @ (location-d.xpos[body])
            probe_geom.quat, probe_geom.size = m.geom_quat[geom], m.geom_size[geom]
            probe_model = probe_spec.compile()
            probe_data = mujoco.MjData(probe_model)
            probe_data.qpos[:] = a.home
            mujoco.mj_forward(probe_model, probe_data)
            probe_id = object_id(probe_model, mujoco.mjtObj.mjOBJ_GEOM, name)
            chest_id = object_id(probe_model, mujoco.mjtObj.mjOBJ_GEOM, "torso_target")
            np.testing.assert_allclose(probe_data.geom_xpos[probe_id], location, atol=1e-9)
            expected = {probe_id, chest_id}
            self.assertTrue(any({int(c.geom1), int(c.geom2)} == expected for c in probe_data.contact),
                            f"{name} versus torso overlap was silently filtered")

    def test_headless_hold_and_small_j6_motion_writes_passed_evidence(self):
        a = build_assembly()
        with tempfile.TemporaryDirectory(prefix="assembly-check-") as tmp:
            with patch("mujoco.Renderer", side_effect=AssertionError("Headless check requested rendering")):
                result = run_assembly_check(a, Path(tmp), duration_s=5.)
            persisted = json.loads((Path(tmp) / "summary.json").read_text())
            with (Path(tmp) / "trajectory.csv").open(newline="") as stream:
                trajectory = list(csv.DictReader(stream))
        self.assertEqual(result["status"], "passed")
        self.assertEqual(persisted["status"], "passed")
        self.assertAlmostEqual(float(a.data.time), 5., places=6)
        self.assertFalse(any(w.number for w in a.data.warning))
        self.assertTrue(np.isfinite(a.data.qpos).all())
        self.assertTrue(np.isfinite(a.data.qvel).all())
        self.assertTrue(np.all(np.abs(a.data.qfrc_actuator) <= DEBUG_CAPS + 1e-7))
        target_j6 = np.array([float(row["J6_target_rad"]) for row in trajectory])
        self.assertGreater(float(np.ptp(target_j6)), .001)
        self.assertAlmostEqual(float(target_j6[0]), float(a.home[6]), places=9)
        self.assertAlmostEqual(float(target_j6[-1]), float(a.home[6]), places=9)

    def test_step_failure_replaces_stale_success_summary(self):
        a = build_assembly()
        with tempfile.TemporaryDirectory(prefix="assembly-failure-") as tmp:
            output = Path(tmp)
            (output / "summary.json").write_text('{"status":"passed"}')
            with patch("mujoco.mj_step", side_effect=RuntimeError("injected assembly step failure")):
                with self.assertRaises(Exception):
                    run_assembly_check(a, output, duration_s=.05)
            persisted = json.loads((output / "summary.json").read_text())
        self.assertEqual(persisted["status"], "failed")
        self.assertIn("injected assembly step failure", json.dumps(persisted))

    def test_invalid_duration_records_failed_summary_before_raising(self):
        a = build_assembly()
        with tempfile.TemporaryDirectory(prefix="assembly-invalid-duration-") as tmp:
            output = Path(tmp)
            (output / "summary.json").write_text('{"status":"passed"}')
            with self.assertRaises(Exception):
                run_assembly_check(a, output, duration_s=float("nan"))
            persisted = json.loads((output / "summary.json").read_text())
        self.assertEqual(persisted["status"], "failed")
        self.assertIn("duration", json.dumps(persisted).lower())

    def test_nonfresh_assembly_records_failed_summary_before_raising(self):
        a = build_assembly()
        mujoco.mj_step(a.model, a.data)
        self.assertGreater(float(a.data.time), 0.)
        with tempfile.TemporaryDirectory(prefix="assembly-nonfresh-") as tmp:
            output = Path(tmp)
            (output / "summary.json").write_text('{"status":"passed"}')
            with self.assertRaises(Exception):
                run_assembly_check(a, output, duration_s=.05)
            persisted = json.loads((output / "summary.json").read_text())
        self.assertEqual(persisted["status"], "failed")
        self.assertIn("fresh", json.dumps(persisted).lower())

    def test_export_reloads_outside_repo_with_same_home_tool_pose(self):
        a = build_assembly()
        with tempfile.TemporaryDirectory(prefix="assembly-export-") as tmp:
            exported = Path(tmp) / "assembly.xml"
            export_scene(a, exported)
            self.assertTrue(exported.is_file())
            reloaded = mujoco.MjModel.from_xml_path(str(exported))
            data = mujoco.MjData(reloaded)
            key = mujoco.mj_name2id(reloaded, mujoco.mjtObj.mjOBJ_KEY, "assembly_home")
            if key >= 0:
                mujoco.mj_resetDataKeyframe(reloaded, data, key)
            else:
                np.testing.assert_allclose(reloaded.qpos0, a.home, atol=1e-12)
            mujoco.mj_forward(reloaded, data)
            self.assertEqual((reloaded.nq, reloaded.nv, reloaded.nu), (7, 7, 7))
            # MjSpec's textual serializer rounds floats to six significant
            # digits; this tolerance is <0.001 degree rather than bit identity.
            np.testing.assert_allclose(data.qpos, a.home, rtol=0, atol=1e-5)
            for name in ["right_shoulder_mount", "mount_face", "shinai_profile_origin", "shinai_grip", "shinai_tip"]:
                old_site = object_id(a.model, mujoco.mjtObj.mjOBJ_SITE, name)
                new_site = object_id(reloaded, mujoco.mjtObj.mjOBJ_SITE, name)
                np.testing.assert_allclose(data.site_xpos[new_site], a.data.site_xpos[old_site], rtol=0, atol=1e-5)
            old_chest = object_id(a.model, mujoco.mjtObj.mjOBJ_GEOM, "torso_target")
            new_chest = object_id(reloaded, mujoco.mjtObj.mjOBJ_GEOM, "torso_target")
            np.testing.assert_allclose(data.geom_xpos[new_chest], a.data.geom_xpos[old_chest], rtol=0, atol=1e-5)

    def test_original_source_xml_and_reported_hashes_unchanged(self):
        self.assertEqual(self.config, load_assembly_config(DEFAULT_CONFIG))
        for name, before in self.source_hashes_before.items():
            self.assertEqual(checksum(ROOT / name), before)
            self.assertIn(before, self.assembly.source_hashes.values())
        self.assertIsInstance(self.assembly.spec, mujoco.MjSpec)


if __name__ == "__main__":
    unittest.main()
