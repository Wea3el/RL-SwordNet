"""Manual guard input is an approximate reference, never an implicit repair.

These tests cover only configuration/coordinate semantics. They deliberately
do not create a native hold, claim a useful parry, or infer joints from images.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from sim.a1_runtime import apply_runtime_profile, load_runtime_profile
from sim.assembly import ROOT
from sim.joint_constraints import JOINT_NAMES, load_joint_constraints
from sim.limited_duel import build_limited_duel
from sim.manual_guard_reference import (
    DEFAULT_REFERENCE, SCORING_PATH, choose_pose, load_reference, static_screen,
)
from sim.triangle_task import TriangleGeometry


class ManualGuardReferenceTests(unittest.TestCase):
    def setUp(self):
        self.constraints = load_joint_constraints()
        self.raw = {
            "schema_version": 1,
            "status": "approximate_user_guard_reference_only",
            "joint_names": list(JOINT_NAMES),
            "display_coordinate_system": "hardware_zero_radians_J3_zero_means_straight",
            "display_rad": [.7, .8, 3.1, .5, 2.9, .9, -.2],
            "mapping_offset_deg": list(self.constraints.offset_deg),
        }

    def load(self, raw):
        with tempfile.TemporaryDirectory() as parent:
            path = Path(parent) / "reference.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            return load_reference(path, constraints=self.constraints)

    def test_load_preserves_reference_and_inverts_j3_offset_once(self):
        raw, displayed, cad = self.load(self.raw)
        self.assertEqual(raw, self.raw)
        np.testing.assert_array_equal(displayed, self.raw["display_rad"])
        expected = np.asarray(self.raw["display_rad"]).copy()
        expected[3] -= np.deg2rad(self.constraints.offset_deg[3])
        np.testing.assert_allclose(cad, expected, atol=1e-15, rtol=0)
        # Returning to the display coordinate must exactly undo one offset,
        # not introduce the historical double-offset migration mistake.
        reconstructed = cad.copy()
        reconstructed[3] += np.deg2rad(self.constraints.offset_deg[3])
        np.testing.assert_allclose(reconstructed, displayed, atol=1e-15, rtol=0)

    def test_joint_name_order_is_required(self):
        for names in (list(reversed(JOINT_NAMES)), list(JOINT_NAMES[:-1]),
                      [*JOINT_NAMES[:-1], "J5"]):
            raw = copy.deepcopy(self.raw)
            raw["joint_names"] = names
            with self.subTest(names=names), self.assertRaises(ValueError):
                self.load(raw)

    def test_display_vector_rejects_bad_shapes_nonfinite_and_booleans(self):
        invalid = ([.5] * 6, [.5] * 8, [[.5] * 7],
                   [True, .8, 3.1, .5, 2.9, .9, -.2],
                   [float("nan")] * 7, [float("inf")] * 7,
                   ["0.5"] * 7)
        for values in invalid:
            raw = copy.deepcopy(self.raw)
            raw["display_rad"] = values
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.load(raw)

    def test_stale_mapping_is_rejected_instead_of_silently_reinterpreted(self):
        for offset in ([0.] * 7, [*self.constraints.offset_deg[:3],
                self.constraints.offset_deg[3] + .001, *self.constraints.offset_deg[4:]],
                list(self.constraints.offset_deg[:-1]),
                [True, *self.constraints.offset_deg[1:]],
                [float("nan")] * 7):
            raw = copy.deepcopy(self.raw)
            raw["mapping_offset_deg"] = offset
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                self.load(raw)

    def test_out_of_bounds_display_reference_is_not_clipped(self):
        for index, value in ((0, -.001), (0, 2 * np.pi + .001),
                             (1, np.pi + .001), (3, np.deg2rad(-20) - .001),
                             (3, np.deg2rad(120) + .001), (5, 0.)):
            raw = copy.deepcopy(self.raw)
            raw["display_rad"][index] = value
            with self.subTest(joint=index, value=value), self.assertRaises(ValueError):
                self.load(raw)

    def test_schema_and_reference_status_are_not_training_claims(self):
        for field, value in (("schema_version", 2), ("schema_version", True),
                             ("status", "training_ready")):
            raw = copy.deepcopy(self.raw)
            raw[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.load(raw)

    def test_choose_plus_pi_changes_only_j0_and_does_not_mutate_input(self):
        displayed = np.array(self.raw["display_rad"])
        before = displayed.copy()
        original = choose_pose(displayed, self.constraints)
        turned = choose_pose(displayed, self.constraints, yaw_delta_rad=np.pi)
        self.assertAlmostEqual(turned[0] - original[0], np.pi)
        np.testing.assert_array_equal(turned[1:], original[1:])
        np.testing.assert_array_equal(displayed, before)
        self.assertAlmostEqual(turned[3], displayed[3] - np.deg2rad(self.constraints.offset_deg[3]))

    def test_choose_rejects_yaw_outside_unwrapped_range(self):
        displayed = np.array(self.raw["display_rad"])
        # Rejection is for the resulting unwrapped angle. A legal small
        # negative delta is meaningful and must not be mistaken for wrapping.
        legal = choose_pose(displayed, self.constraints, yaw_delta_rad=-.1)
        self.assertAlmostEqual(legal[0], .6)
        for delta in (-1., 2 * np.pi, -2 * np.pi):
            with self.subTest(delta=delta), self.assertRaises(ValueError):
                choose_pose(displayed, self.constraints, yaw_delta_rad=delta)
        np.testing.assert_array_equal(displayed, self.raw["display_rad"])

    def test_choose_rejects_nonfinite_yaw_and_bad_input(self):
        for delta in (True, float("nan"), float("inf")):
            with self.subTest(delta=delta), self.assertRaises(ValueError):
                choose_pose(self.raw["display_rad"], self.constraints, yaw_delta_rad=delta)
        invalid = self.raw["display_rad"].copy()
        invalid[3] = np.deg2rad(120) + .01
        with self.assertRaises(ValueError):
            choose_pose(invalid, self.constraints)


class ManualGuardStaticScreenTests(unittest.TestCase):
    """Genuine compiled FK screens; no stepping or inherited parry claim."""

    @classmethod
    def setUpClass(cls):
        cls.constraints = load_joint_constraints()
        raw, displayed, original = load_reference(DEFAULT_REFERENCE, cls.constraints)
        facing = choose_pose(displayed, cls.constraints,
                             yaw_delta_rad=raw["facing_variant"]["base_yaw_delta_rad"])
        scoring = json.loads(SCORING_PATH.read_text())
        cls.geometry = TriangleGeometry.from_mapping(
            {k: scoring[k] for k in ("base_distance", "regions")})
        profile = load_runtime_profile(ROOT / "config/a1_runtime_rk4_candidate.json")
        with patch("mujoco.mj_step", side_effect=AssertionError("Static screens must not integrate")):
            cls.original_scene = build_limited_duel(
                cls.geometry.base_distance, initial_qpos=original, constraints=cls.constraints)
            cls.facing_scene = build_limited_duel(
                cls.geometry.base_distance, initial_qpos=facing, constraints=cls.constraints)
            for scene in (cls.original_scene, cls.facing_scene):
                apply_runtime_profile(scene, profile)
            cls.original_report = static_screen(cls.original_scene, cls.geometry)
            cls.facing_report = static_screen(cls.facing_scene, cls.geometry)

    def test_explicit_half_turn_preserves_height_and_upright_but_changes_facing(self):
        self.assertFalse(self.original_report["candidate_static_gate"])
        self.assertTrue(self.facing_report["candidate_static_gate"])
        for slot in ("a1", "a2"):
            original = self.original_report["slots"][slot]
            facing = self.facing_report["slots"][slot]
            self.assertLess(original["grip_self_frame_m"][0], 0.)
            self.assertGreater(facing["grip_self_frame_m"][0], 0.)
            self.assertAlmostEqual(original["grip_self_frame_m"][2],
                                   facing["grip_self_frame_m"][2], places=12)
            self.assertAlmostEqual(original["sword_elevation_deg"],
                                   facing["sword_elevation_deg"], places=11)
        for report in (self.original_report, self.facing_report):
            self.assertFalse(report["native_motion_tested"])
            self.assertFalse(report["defence_certified"])
            self.assertFalse(report["admitted_to_training"])

    def test_two_slots_have_equal_self_frame_geometry(self):
        for report in (self.original_report, self.facing_report):
            for field in ("grip_self_frame_m", "sword_axis_self_frame",
                          "nearest_angle_margin_deg"):
                np.testing.assert_allclose(report["slots"]["a1"][field],
                                           report["slots"]["a2"][field], atol=1e-12, rtol=0)
            self.assertAlmostEqual(report["slots"]["a1"]["sword_elevation_deg"],
                                   report["slots"]["a2"]["sword_elevation_deg"], places=12)

    def test_actual_out_of_range_state_fails_without_clipping(self):
        scene = self.facing_scene
        before = scene.data.qpos.copy()
        ids = scene.agents["a1"]
        qid, jid = ids["qpos_ids"][1], ids["joint_ids"][1]
        illegal = scene.model.jnt_range[jid, 1] + .01
        try:
            scene.data.qpos[qid] = illegal
            with patch("mujoco.mj_step", side_effect=AssertionError("No integration")):
                report = static_screen(scene, self.geometry)
            self.assertFalse(report["candidate_static_gate"])
            self.assertFalse(report["slots"]["a1"]["actual_joint_ranges_pass"])
            self.assertTrue(report["slots"]["a2"]["actual_joint_ranges_pass"])
            self.assertEqual(scene.data.qpos[qid], illegal)
            self.assertEqual(report["slots"]["a1"]["original_cad_rad"][1], illegal)
        finally:
            scene.data.qpos[:] = before
            mujoco.mj_forward(scene.model, scene.data)


if __name__ == "__main__":
    unittest.main()
