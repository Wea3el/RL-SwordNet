"""Dual-scene preservation and contact eligibility, not action validation."""
from __future__ import annotations

from types import SimpleNamespace
import unittest

import mujoco
import numpy as np

from sim.duel_scene import FLOOR_BIT, build_duel_scene, classify_contact, duel_contact_records
from sim.floor import build_floor_assembly


def gid(model, name):
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if i < 0:
        raise AssertionError(name)
    return i


def eligible(model, first, second):
    return bool((int(model.geom_contype[first]) & int(model.geom_conaffinity[second]))
                or (int(model.geom_contype[second]) & int(model.geom_conaffinity[first])))


class DuelSceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.duel = build_duel_scene()
        cls.single = build_floor_assembly(np.tile([-np.pi, np.pi], (7, 1)))

    def test_fourteen_independent_fixed_base_joints_and_one_floor(self):
        scene, m = self.duel, self.duel.model
        self.assertEqual((m.nq, m.nv, m.nu), (14, 14, 14))
        self.assertFalse(np.any(m.jnt_type == mujoco.mjtJoint.mjJNT_FREE))
        self.assertEqual(int(np.sum(m.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)), 1)
        self.assertEqual(m.nkey, 0)
        self.assertEqual(m.nlight, 1)
        self.assertEqual(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso"), -1)
        for name, mapping in scene.agents.items():
            self.assertEqual(len(mapping["actuator_ids"]), 7)
            pos = scene.data.site_xpos[mapping["sites"]["floor_mount"]]
            np.testing.assert_allclose(pos, [0, 0, 0] if name == "a1" else [1.6, 0, 0], atol=1e-12)
        self.assertFalse(set(scene.agents["a1"]["qpos_ids"]) & set(scene.agents["a2"]["qpos_ids"]))

    def test_original_size_mass_inertia_mesh_scales_gains_and_caps(self):
        scene, old = self.duel, self.single.model
        m, d, od = scene.model, scene.data, self.single.data
        rotation2 = np.diag([-1., -1., 1.])
        for agent, mapping in scene.agents.items():
            prefix = mapping["prefix"]
            for i in range(1, old.nbody):
                name = mujoco.mj_id2name(old, mujoco.mjtObj.mjOBJ_BODY, i)
                j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, prefix+name)
                for field in ("body_mass", "body_ipos", "body_inertia"):
                    np.testing.assert_array_equal(getattr(m, field)[j], getattr(old, field)[i])
            for i in range(old.nmesh):
                name = mujoco.mj_id2name(old, mujoco.mjtObj.mjOBJ_MESH, i)
                j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MESH, prefix+name)
                np.testing.assert_array_equal(m.mesh_scale[j], old.mesh_scale[i])
                a, n = old.mesh_vertadr[i], old.mesh_vertnum[i]
                b = m.mesh_vertadr[j]
                np.testing.assert_array_equal(m.mesh_vert[b:b+n], old.mesh_vert[a:a+n])
            for field in ("actuator_gainprm", "actuator_forcerange", "actuator_ctrlrange"):
                np.testing.assert_allclose(getattr(m, field)[mapping["actuator_ids"]], getattr(old, field))
            np.testing.assert_allclose(m.jnt_range[mapping["joint_ids"]], old.jnt_range)
            for name, site in mapping["sites"].items():
                oldsite = mujoco.mj_name2id(old, mujoco.mjtObj.mjOBJ_SITE, name)
                expected = od.site_xpos[oldsite] if agent == "a1" else rotation2@od.site_xpos[oldsite]+[1.6, 0, 0]
                np.testing.assert_allclose(d.site_xpos[site], expected, atol=1e-12)
            positions = {name: d.site_xpos[i] for name, i in mapping["sites"].items()}
            self.assertAlmostEqual(np.linalg.norm(positions["shinai_tip"]-positions["shinai_pommel"]), 1.08, places=9)
            self.assertAlmostEqual(np.linalg.norm(positions["shinai_tip"]-positions["shinai_grip"]), .79, places=9)
        self.assertAlmostEqual(float(m.body_mass.sum()), 2*float(old.body_mass.sum()), places=9)

    def test_cross_blades_arm_base_floor_and_self_adjacency_masks(self):
        m = self.duel.model
        for a in ("a1", "a2"):
            b = "a2" if a == "a1" else "a1"
            blade = gid(m, f"{a}_shinai_blade_collision_0")
            for other in (f"{b}_shinai_blade_collision_0", f"{b}_arm_proxy_0", f"{b}_base_collision", "floor"):
                self.assertTrue(eligible(m, blade, gid(m, other)), other)
            for i in range(7):
                first = gid(m, f"{a}_arm_proxy_{i}")
                for j in range(max(0, i-1), min(7, i+2)):
                    self.assertFalse(eligible(m, first, gid(m, f"{a}_arm_proxy_{j}")))
                for j in range(7):
                    self.assertTrue(eligible(m, first, gid(m, f"{b}_arm_proxy_{j}")))
        self.assertEqual(int(m.geom_contype[gid(m, "floor")]), FLOOR_BIT)
        self.assertTrue(m.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT))

    def test_contact_classifier_does_not_confuse_handle_or_floor_with_blades(self):
        m = self.duel.model
        def category(a, b):
            return classify_contact(m, SimpleNamespace(geom1=gid(m, a), geom2=gid(m, b)))
        self.assertEqual(category("a1_shinai_blade_collision_0", "a2_shinai_blade_collision_1"), "cross_blade_blade")
        self.assertEqual(category("a1_shinai_blade_collision_0", "a2_shinai_handle_collision_0"), "illegal_cross_robot_nonblade_contact")
        self.assertEqual(category("a1_shinai_blade_collision_0", "floor"), "illegal_floor_contact")

    def test_v2_guard_contact_is_neutral_without_changing_solver_or_v1(self):
        m = self.duel.model
        contact = SimpleNamespace(geom1=gid(m, "a1_shinai_blade_collision_0"),
                                  geom2=gid(m, "a2_shinai_guard_collision"))
        self.assertEqual(classify_contact(m, contact), "illegal_cross_robot_nonblade_contact")
        self.assertEqual(classify_contact(m, contact, rules_version="whole_blade_first_touch_v2"), "cross_blade_guard")
        contact.geom2 = gid(m, "a2_shinai_handle_collision_0")
        self.assertEqual(classify_contact(m, contact, rules_version="whole_blade_first_touch_v2"), "illegal_cross_robot_nonblade_contact")
        with self.assertRaises(ValueError):
            classify_contact(m, contact, rules_version="silent_rule_change")

    def test_native_solver_generates_force_for_deliberate_blade_overlap_probe(self):
        # This intentionally INVALID placement overlaps the two robots solely
        # to probe the collision engine. It is not an allowed pose, action,
        # force calibration, or a successful parry, and is never used by build.
        scene = build_duel_scene()
        fixed = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, "a2_fixed_base")
        scene.model.body_pos[fixed] = [0, .01, 0]
        scene.model.body_quat[fixed] = [1, 0, 0, 0]
        mujoco.mj_setConst(scene.model, scene.data)
        mujoco.mj_forward(scene.model, scene.data)
        blades = [r for r in duel_contact_records(scene) if r["category"] == "cross_blade_blade"]
        self.assertTrue(blades, "Mask eligibility must result in actual native blade contacts")
        self.assertTrue(any(r["normal_force_n"] > 0 for r in blades))
        self.assertGreater(float(np.linalg.norm(scene.data.qfrc_constraint)), 0)
        np.testing.assert_array_equal(scene.data.qfrc_applied, np.zeros(14))

    def test_virtual_targets_are_sites_only_and_add_no_collision_bodies(self):
        scene = self.duel
        for name, expected in (("a1_target", [.32, 0, 1.05]), ("a2_target", [1.28, 0, 1.05])):
            site = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_SITE, name)
            np.testing.assert_allclose(scene.data.site_xpos[site], expected, atol=1e-12)
            self.assertEqual(mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM, name), -1)

    def test_invalid_spacing_rejected_and_export_compiles(self):
        for value in (0, .2, float("nan"), True):
            with self.assertRaises(ValueError):
                build_duel_scene(value)
        rebuilt = mujoco.MjModel.from_xml_string(self.duel.spec.to_xml())
        self.assertEqual((rebuilt.nq, rebuilt.nu), (14, 14))


if __name__ == "__main__":
    unittest.main()
