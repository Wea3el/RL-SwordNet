"""Construction-only tests for the opt-in confirmed-angle dual scene."""
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from sim.duel_scene import build_duel_scene
from sim.joint_constraints import load_joint_constraints, model_joint_ranges_rad
from sim.limited_duel import build_limited_duel


class LimitedDuelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.constraints = load_joint_constraints()
        cls.ranges = model_joint_ranges_rad(cls.constraints)
        with patch("mujoco.mj_step", side_effect=AssertionError("No construction-time integration")):
            cls.limited = build_limited_duel(constraints=cls.constraints)
            cls.old = build_duel_scene(1.75)

    def test_both_slot_ranges_control_and_legal_keyframe(self):
        scene = self.limited
        self.assertEqual((scene.model.nq, scene.model.nv, scene.model.nu), (14, 14, 14))
        key = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_KEY, "limited_duel_home")
        for slot, mapping in scene.agents.items():
            np.testing.assert_array_equal(scene.model.jnt_range[mapping["joint_ids"]], self.ranges)
            np.testing.assert_array_equal(scene.model.actuator_ctrlrange[mapping["actuator_ids"]], self.ranges)
            q = scene.data.qpos[mapping["qpos_ids"]]
            np.testing.assert_array_equal(q, scene.model.key_qpos[key, mapping["qpos_ids"]])
            np.testing.assert_allclose(np.rad2deg(q), [1, 1, 1, 0, 1, 16, 0], atol=1e-12)
            self.assertTrue(np.all(scene.model.jnt_limited[mapping["joint_ids"]]))
            self.assertTrue(np.all(scene.model.actuator_ctrllimited[mapping["actuator_ids"]]))
        self.assertEqual(scene.data.time, 0.)
        np.testing.assert_array_equal(scene.data.qvel, np.zeros(14))
        np.testing.assert_array_equal(scene.data.qfrc_applied, np.zeros(14))
        np.testing.assert_array_equal(scene.data.xfrc_applied, np.zeros_like(scene.data.xfrc_applied))

    def test_explicit_shared_and_distinct_poses_preserved(self):
        midpoint = self.ranges.mean(axis=1)
        same = build_limited_duel(base_distance_m=1.85, initial_qpos=midpoint)
        self.assertEqual(same.config["base_distance_m"], 1.85)
        self.assertAlmostEqual(same.config["virtual_targets"]["a2"]["center_world_m"][0], 1.53)
        for mapping in same.agents.values():
            np.testing.assert_array_equal(same.data.qpos[mapping["qpos_ids"]], midpoint)
        poses = {"a1": self.ranges[:, 0].copy(), "a2": self.ranges[:, 1].copy()}
        distinct = build_limited_duel(initial_qpos=poses)
        for slot, mapping in distinct.agents.items():
            np.testing.assert_array_equal(distinct.data.qpos[mapping["qpos_ids"]], poses[slot])

    def test_explicit_invalid_rejected_before_geometry_construction(self):
        valid = self.ranges.mean(axis=1)
        invalid = (np.zeros(7), np.zeros(14), [True]*7,
            {"a1": valid, "a2": np.zeros(7)}, {"a1": valid},
            {"a1": valid, "a2": valid, "extra": valid})
        with patch("sim.limited_duel.build_duel_scene", side_effect=AssertionError("Construction must not start")):
            for pose in invalid:
                with self.subTest(pose=pose), self.assertRaises(ValueError):
                    build_limited_duel(initial_qpos=pose)

    def test_exported_spec_matches_model_key_and_does_not_rebase_joints(self):
        scene = self.limited
        compiled = scene.spec.compile()
        for field in ("jnt_range", "actuator_ctrlrange", "actuator_forcerange", "key_qpos", "key_ctrl", "qpos0", "qpos_spring"):
            np.testing.assert_array_equal(getattr(compiled, field), getattr(scene.model, field))
        np.testing.assert_array_equal(scene.model.qpos0, self.old.model.qpos0)
        np.testing.assert_array_equal(scene.model.qpos_spring, self.old.model.qpos_spring)
        for slot in ("a1", "a2"):
            for i in range(7):
                name = f"{slot}_J{i}"
                self.assertEqual(scene.spec.joint(name).ref, self.old.spec.joint(name).ref)
        key = mujoco.mj_name2id(compiled, mujoco.mjtObj.mjOBJ_KEY, "limited_duel_home")
        data = mujoco.MjData(compiled)
        mujoco.mj_resetDataKeyframe(compiled, data, key)
        np.testing.assert_array_equal(data.qpos, scene.data.qpos)
        np.testing.assert_array_equal(data.ctrl, scene.data.ctrl)

    def test_geometry_caps_masks_and_targets_preserved(self):
        current, previous = self.limited, self.old
        for field in ("body_pos", "body_quat", "body_mass", "body_ipos", "body_inertia",
                      "geom_pos", "geom_quat", "geom_size", "geom_contype", "geom_conaffinity",
                      "mesh_scale", "mesh_vert", "mesh_face", "site_pos", "site_quat", "site_size",
                      "actuator_gainprm", "actuator_biasprm", "actuator_forcerange",
                      "actuator_forcelimited", "actuator_gear", "dof_damping", "dof_armature",
                      "jnt_axis", "jnt_pos"):
            with self.subTest(field=field):
                np.testing.assert_array_equal(getattr(current.model, field), getattr(previous.model, field))
        self.assertEqual(current.config["virtual_targets"], previous.config["virtual_targets"])
        self.assertEqual(current.config["base_placements"], previous.config["base_placements"])
        self.assertEqual(current.source_hashes, previous.source_hashes)

    def test_legacy_limits_and_new_default_distance_unchanged(self):
        np.testing.assert_array_equal(self.old.model.jnt_range, np.tile([-np.pi, np.pi], (14, 1)))
        np.testing.assert_array_equal(self.old.data.qpos, np.zeros(14))
        self.assertEqual(self.old.config["joint_range_status"], "hypothetical_ranges_not_identified_hardware_limits")
        self.assertEqual(self.limited.config["base_distance_m"], 1.75)
        self.assertFalse(self.limited.config["geometry_staging"]["integrated"])


if __name__ == "__main__":
    unittest.main()
