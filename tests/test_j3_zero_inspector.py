"""Display-only J3 zero conversion: actual FK and servo checks, no stepping."""
from dataclasses import replace
from contextlib import nullcontext
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from sim.j3_zero_inspector import (
    CONTROL_MODE, STATUS, StaticPoseControls, build_inspector, cad_to_display,
    compare_coordinates, display_to_cad, inspect, pose_record, rebase_scene,
    write_inspector,
)
from sim.joint_constraints import load_joint_constraints


class J3ZeroInspectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Explicit test fixture remains meaningful before/after the production
        # configuration update and does not edit the shared configuration.
        cls.constraints = replace(load_joint_constraints(),
            offset_deg=(0., 0., 0., 12.49194366, 0., 0., 0.),
            confirmation_source="test: hardware J3 zero is a straight arm")
        with patch("mujoco.mj_step", side_effect=AssertionError("Never integrate")):
            cls.source, cls.display = build_inspector(cls.constraints)

    def test_default_straight_zero_both_slots_and_exact_ranges(self):
        for slot, ids in self.display.agents.items():
            qid, jid = ids["qpos_ids"][3], ids["joint_ids"][3]
            self.assertEqual(self.display.data.qpos[qid], 0.)
            np.testing.assert_allclose(np.rad2deg(self.display.model.jnt_range[jid]), [-20, 120], atol=1e-12)
            self.assertAlmostEqual(self.source.data.qpos[qid], -np.deg2rad(12.49194366))
        self.assertEqual(self.display.data.time, 0.)
        self.assertEqual(self.display.config["status"], STATUS)

    def test_reversible_coordinate_mapping_with_negative_and_multiturn(self):
        q = np.array([[.1, .2, 6.1, -.4, 6.2, .5, -.8],
                      [6., 2., .2, 1.7, .3, 1., .1]])
        displayed = cad_to_display(q, self.constraints)
        np.testing.assert_allclose(display_to_cad(displayed, self.constraints), q, atol=1e-15)
        np.testing.assert_array_equal(displayed[:, [0, 1, 2, 4, 5, 6]], q[:, [0, 1, 2, 4, 5, 6]])
        # Conversion itself does not clip illegal posed angles.
        self.assertLess(displayed[0, 3], 0.)
        with self.assertRaises(ValueError):
            display_to_cad([float("nan")] * 7, self.constraints)

    def test_nonzero_fk_actuator_bias_and_force_equivalence_both_slots(self):
        result = compare_coordinates(self.source, self.display, self.constraints)
        self.assertTrue(result["tested_both_slots"])
        self.assertLess(max(result["maximum_absolute_errors"].values()), 1e-8)

    def test_wrong_unshifted_control_is_not_equivalent(self):
        a, b = mujoco.MjData(self.source.model), mujoco.MjData(self.display.model)
        a.qpos[:] = self.source.model.jnt_range.mean(axis=1)
        a.ctrl[:] = a.qpos + .0002
        b.qpos[:], b.ctrl[:] = a.qpos, a.ctrl
        for ids in self.source.agents.values():
            b.qpos[ids["qpos_ids"][3]] += np.deg2rad(12.49194366)
        mujoco.mj_forward(self.source.model, a)
        mujoco.mj_forward(self.display.model, b)
        for ids in self.source.agents.values():
            aid = ids["actuator_ids"][3]
            self.assertGreater(abs(a.actuator_force[aid] - b.actuator_force[aid]), 1.)

    def test_source_unmodified_and_keys_rebased_not_only_initial_state(self):
        original_qpos0 = self.source.model.qpos0.copy()
        original_keys = self.source.model.key_qpos.copy()
        original_refs = [self.source.spec.joint(f"{s}_J3").ref for s in ("a1", "a2")]
        again = rebase_scene(self.source, self.constraints)
        np.testing.assert_array_equal(self.source.model.qpos0, original_qpos0)
        np.testing.assert_array_equal(self.source.model.key_qpos, original_keys)
        self.assertEqual([self.source.spec.joint(f"{s}_J3").ref for s in ("a1", "a2")], original_refs)
        for ids in self.source.agents.values():
            qid, aid = ids["qpos_ids"][3], ids["actuator_ids"][3]
            np.testing.assert_allclose(again.model.key_qpos[:, qid], original_keys[:, qid] + np.deg2rad(12.49194366))
            np.testing.assert_allclose(again.model.key_ctrl[:, aid], self.source.model.key_ctrl[:, aid] + np.deg2rad(12.49194366))
        for field in ("body_mass", "body_inertia", "geom_size", "geom_contype", "geom_conaffinity",
                      "actuator_gainprm", "actuator_biasprm", "actuator_forcerange"):
            np.testing.assert_array_equal(getattr(again.model, field), getattr(self.source.model, field))

    def test_pose_export_recovers_original_coordinates(self):
        record = pose_record(self.display, self.constraints)
        for slot, ids in self.source.agents.items():
            np.testing.assert_allclose(record["slots"][slot]["original_cad_rad"], self.source.data.qpos[ids["qpos_ids"]], atol=1e-15)
            self.assertEqual(record["slots"][slot]["hardware_deg"][3], 0.)

    def test_fresh_export_and_no_training_claim(self):
        with tempfile.TemporaryDirectory() as parent:
            path = Path(parent) / "fresh"
            with patch("mujoco.mj_step", side_effect=AssertionError("Never integrate")):
                _, _, report = write_inspector(path, constraints=self.constraints)
            self.assertTrue((path / "scene.xml").is_file())
            saved = json.loads((path / "mapping.json").read_text())
            self.assertFalse(saved["for_training"])
            self.assertFalse(saved["motion_or_hold_certified"])
            self.assertEqual(saved["status"], report["status"])
            self.assertEqual(saved["interactive_control_mode"], CONTROL_MODE)
            before = (path / "scene.xml").read_bytes()
            with self.assertRaises(FileExistsError):
                write_inspector(path, constraints=self.constraints)
            self.assertEqual(before, (path / "scene.xml").read_bytes())

    def test_rejects_unconfirmed_or_other_changed_coordinates(self):
        with self.assertRaises(ValueError):
            cad_to_display(np.zeros(7), replace(self.constraints, offset_deg=(0.,) * 7))
        with self.assertRaises(ValueError):
            cad_to_display(np.zeros(7), replace(self.constraints, offset_deg=(1., 0., 0., 12., 0., 0., 0.)))
        with self.assertRaises(ValueError):
            cad_to_display(np.zeros(7), replace(self.constraints, confirmed=False))

    def test_wrong_source_mapping_and_double_rebase_rejected(self):
        with self.assertRaises(ValueError):
            rebase_scene(self.display, self.constraints)
        wrong = replace(self.constraints, offset_deg=(0., 0., 0., 20., 0., 0., 0.))
        with self.assertRaises(ValueError):
            rebase_scene(self.source, wrong)


class StaticPoseControlsTests(unittest.TestCase):
    def setUp(self):
        self.constraints = load_joint_constraints()
        self.source, self.scene = build_inspector(self.constraints)

    def test_initial_control_is_pose_not_gravity_feedforward(self):
        pose = self.scene.data.qpos.copy()
        self.assertGreater(np.max(np.abs(self.scene.data.ctrl - pose)), 1e-5)
        with patch("mujoco.mj_step", side_effect=AssertionError("Never integrate")):
            controls = StaticPoseControls(self.scene)
            self.assertEqual(controls.apply(), [])
        np.testing.assert_array_equal(self.scene.data.qpos, pose)
        np.testing.assert_array_equal(self.scene.data.ctrl, pose)
        self.assertEqual(self.scene.data.time, 0.)

    def test_actual_fk_changes_independently_for_each_arm(self):
        controls = StaticPoseControls(self.scene)
        original_pose = self.scene.data.qpos.copy()
        positions = self.scene.data.site_xpos.copy()
        ids = self.scene.agents["a2"]
        self.scene.data.ctrl[ids["actuator_ids"][3]] = np.deg2rad(40)
        self.scene.data.qvel[:] = 1
        with patch("mujoco.mj_step", side_effect=AssertionError("Never integrate")):
            self.assertEqual(controls.apply(), [])
        expected = original_pose.copy()
        expected[ids["qpos_ids"][3]] = np.deg2rad(40)
        np.testing.assert_array_equal(self.scene.data.qpos, expected)
        np.testing.assert_array_equal(self.scene.data.qvel, 0)
        self.assertEqual(self.scene.data.time, 0.)
        a1sites = list(self.scene.agents["a1"]["sites"].values())
        a2sites = list(ids["sites"].values())
        np.testing.assert_array_equal(self.scene.data.site_xpos[a1sites], positions[a1sites])
        self.assertGreater(np.max(np.abs(self.scene.data.site_xpos[a2sites] - positions[a2sites])), .01)
        record = pose_record(self.scene, self.constraints)
        self.assertAlmostEqual(record["slots"]["a2"]["hardware_deg"][3], 40)
        self.assertAlmostEqual(record["slots"]["a2"]["original_cad_rad"][3],
                               np.deg2rad(40 - self.constraints.offset_deg[3]))

    def test_j3_exact_endpoints_and_zero_in_both_slots(self):
        controls = StaticPoseControls(self.scene)
        for slot, ids in self.scene.agents.items():
            for degrees in (-20., 0., 120.):
                self.scene.data.ctrl[ids["actuator_ids"][3]] = np.deg2rad(degrees)
                self.assertEqual(controls.apply(), [])
                self.assertAlmostEqual(self.scene.data.qpos[ids["qpos_ids"][3]], np.deg2rad(degrees))

    def test_reject_invalid_fields_without_clipping_and_accept_valid_fields(self):
        controls = StaticPoseControls(self.scene)
        ids = self.scene.agents["a1"]
        previous = self.scene.data.qpos[ids["qpos_ids"]].copy()
        self.scene.data.ctrl[ids["actuator_ids"]] = [1., np.nan, np.inf, -1., 7., 0., 0.]
        self.assertEqual(controls.apply(), ["a1_J1", "a1_J2", "a1_J3", "a1_J4", "a1_J5"])
        expected = previous.copy()
        expected[[0, 6]] = [1., 0.]
        np.testing.assert_array_equal(self.scene.data.qpos[ids["qpos_ids"]], expected)
        np.testing.assert_array_equal(self.scene.data.ctrl[ids["actuator_ids"]], expected)
        # A GUI Reset/Clear-all cannot force J5 to zero or keep invalid qpos.
        self.scene.data.qpos[:] = 0
        self.scene.data.ctrl[:] = 0
        self.assertEqual(controls.apply(), ["a1_J5", "a2_J5"])
        self.assertEqual(self.scene.data.qpos[ids["qpos_ids"][5]], previous[5])

    def test_refuses_shared_training_scene_before_mutation(self):
        before = self.source.data.ctrl.copy()
        with self.assertRaisesRegex(ValueError, "dedicated inspection"):
            StaticPoseControls(self.source)
        np.testing.assert_array_equal(self.source.data.ctrl, before)

    def test_interactive_loop_applies_gui_edits_and_saves_actual_pose(self):
        scene = self.scene

        class FakeViewer:
            def __init__(self):
                self.cam = type("Cam", (), {"lookat": np.zeros(3)})()
                self.ticks = 0

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def is_running(self):
                return self.ticks < 2

            def lock(self):
                return nullcontext()

            def set_texts(self, texts):
                self.texts = texts

            def sync(self):
                # Simulate the edits entering data through viewer.sync().
                slot = ("a1", "a2")[self.ticks]
                scene.data.ctrl[scene.agents[slot]["actuator_ids"][3]] = np.deg2rad(-15 if self.ticks == 0 else 35)
                self.ticks += 1

        viewer = FakeViewer()
        with tempfile.TemporaryDirectory() as parent:
            with patch("mujoco.viewer.launch_passive", return_value=viewer), \
                    patch("sim.j3_zero_inspector.time.sleep"), \
                    patch("mujoco.mj_step", side_effect=AssertionError("Never integrate")):
                inspect(scene, self.constraints, parent)
            record = json.loads((Path(parent) / "pose_on_close.json").read_text())
        self.assertEqual(viewer.ticks, 2)
        self.assertIn("Control = joint angle", viewer.texts[2])
        self.assertEqual(record["interactive_control_mode"], CONTROL_MODE)
        self.assertEqual(record["time_s"], 0)
        self.assertAlmostEqual(record["slots"]["a1"]["hardware_deg"][3], -15)
        self.assertAlmostEqual(record["slots"]["a2"]["hardware_deg"][3], 35)


if __name__ == "__main__":
    unittest.main()
