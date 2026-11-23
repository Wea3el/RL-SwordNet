"""Confirmed-angle constructor checks; no integrated physics rollouts."""
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from sim.floor import build_floor_assembly
from sim.joint_constraints import JointConstraints, load_joint_constraints, model_joint_ranges_rad
from sim.limited_model import build_limited_floor


class LimitedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.constraints = load_joint_constraints()
        cls.ranges = model_joint_ranges_rad(cls.constraints)
        with patch("mujoco.mj_step", side_effect=AssertionError("constructor must not integrate")):
            cls.limited = build_limited_floor(cls.constraints)
            cls.legacy = build_floor_assembly()

    def test_confirmed_ranges_and_legal_default_reset(self):
        a, m, d = self.limited, self.limited.model, self.limited.data
        self.assertEqual((m.nq, m.nv, m.nu), (7, 7, 7))
        np.testing.assert_array_equal(m.jnt_range, self.ranges)
        np.testing.assert_array_equal(m.actuator_ctrlrange, self.ranges)
        self.assertTrue(np.all(m.jnt_limited))
        self.assertTrue(np.all(m.actuator_ctrllimited))
        np.testing.assert_array_equal(a.home, d.qpos)
        self.assertTrue(np.all(d.qpos > self.ranges[:, 0]))
        self.assertTrue(np.all(d.qpos < self.ranges[:, 1]))
        self.assertAlmostEqual(np.rad2deg(d.qpos[5]), 16.)
        self.assertEqual(d.time, 0.)
        np.testing.assert_array_equal(d.qvel, np.zeros(7))
        np.testing.assert_array_equal(d.qfrc_applied, np.zeros(7))
        self.assertFalse(a.config["joint_constraints"]["automatic_angle_wrapping"])

    def test_exportable_spec_matches_model_and_home_key(self):
        a = self.limited
        rebuilt = a.spec.compile()
        for field in ("jnt_range", "actuator_ctrlrange", "actuator_forcerange", "key_qpos", "key_ctrl"):
            np.testing.assert_array_equal(getattr(rebuilt, field), getattr(a.model, field))
        key = mujoco.mj_name2id(rebuilt, mujoco.mjtObj.mjOBJ_KEY, "floor_home")
        state = mujoco.MjData(rebuilt)
        mujoco.mj_resetDataKeyframe(rebuilt, state, key)
        np.testing.assert_array_equal(state.qpos, a.home)
        np.testing.assert_array_equal(state.ctrl, a.data.ctrl)
        self.assertTrue(np.all(state.ctrl >= self.ranges[:, 0]))
        self.assertTrue(np.all(state.ctrl <= self.ranges[:, 1]))

    def test_explicit_legal_pose_is_not_projected(self):
        pose = self.ranges.mean(axis=1)
        a = build_limited_floor(self.constraints, pose)
        np.testing.assert_array_equal(a.home, pose)
        np.testing.assert_array_equal(a.data.qpos, pose)
        # Boundary input is legal; no hidden default interior margin is added.
        b = build_limited_floor(self.constraints, self.ranges[:, 0])
        np.testing.assert_array_equal(b.home, self.ranges[:, 0])

    def test_rejects_explicit_illegal_or_malformed_pose(self):
        for pose in (np.zeros(7), np.zeros(6), np.full(7, np.nan), [True] * 7):
            with self.subTest(pose=pose), self.assertRaises(ValueError):
                build_limited_floor(self.constraints, pose)
        pose = self.ranges.mean(axis=1)
        pose[0] = self.ranges[0, 1] + 2 * np.pi
        with self.assertRaises(ValueError):
            build_limited_floor(self.constraints, pose)

    def test_unconfirmed_mapping_is_not_enforced(self):
        c = JointConstraints("test", self.constraints.bounds_deg, False, None, None, "")
        with self.assertRaises(ValueError):
            build_limited_floor(c)

    def test_original_geometry_inertias_caps_and_masks_preserved(self):
        a, b = self.limited.model, self.legacy.model
        for field in ("body_pos", "body_quat", "body_mass", "body_ipos", "body_inertia",
                      "geom_pos", "geom_quat", "geom_size", "geom_contype", "geom_conaffinity",
                      "mesh_scale", "mesh_vert", "mesh_face", "site_pos", "site_quat",
                      "actuator_gainprm", "actuator_biasprm", "actuator_forcerange",
                      "actuator_forcelimited", "actuator_gear", "dof_damping", "dof_armature"):
            with self.subTest(field=field):
                np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
        caps = np.array([5, 80, 15, 20, 3, 3, 2])
        np.testing.assert_array_equal(a.actuator_forcerange, np.column_stack((-caps, caps)))
        self.assertEqual(self.limited.source_hashes, self.legacy.source_hashes)

    def test_legacy_constructor_defaults_unchanged(self):
        a = self.legacy
        half = np.asarray(a.config["runtime"]["joint_half_windows_rad"])
        np.testing.assert_array_equal(a.home, np.zeros(7))
        np.testing.assert_array_equal(a.data.qpos, np.zeros(7))
        np.testing.assert_array_equal(a.model.jnt_range, np.column_stack((-half, half)))
        self.assertEqual(a.config["floor"]["joint_range_status"], "original_debug_software_windows")


if __name__ == "__main__":
    unittest.main()
