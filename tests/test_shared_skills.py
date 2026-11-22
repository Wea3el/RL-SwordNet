"""Shared finite-action geometry and own-clock seam tests, no dual rollout."""
from __future__ import annotations

import inspect
import unittest
from unittest.mock import patch

import numpy as np

from plan1_skills.matching import MatchResult
from sim.shared_skills import DEFAULT_CONFIG, _screen, build_shared_library, compose_skill, task_segment


class SharedPathMathTests(unittest.TestCase):
    def test_task_interpolates_grip_and_spherical_axis_with_exact_endpoints(self):
        phase, grip, axis = task_segment([.5, -.25, .75], [.3, 0., 1.05], [0., 0., 1.], [1., 0., 0.], 9)
        np.testing.assert_allclose(grip[[0, -1]], [[.5, -.25, .75], [.3, 0., 1.05]])
        np.testing.assert_allclose(axis[[0, -1]], [[0., 0., 1.], [1., 0., 0.]])
        np.testing.assert_allclose(axis[4], [np.sqrt(.5), 0., np.sqrt(.5)])
        np.testing.assert_allclose(np.linalg.norm(axis, axis=1), 1.)
        self.assertEqual(len(phase), 9)

    def test_invalid_task_and_antipodal_axes_are_explicit(self):
        with self.assertRaises(ValueError):
            task_segment([0., 0., 0.], [1., 0., 0.], [1., 0., 0.], [-1., 0., 0.])
        with self.assertRaises(ValueError):
            task_segment([0., 0., 0.], [1., 0., 0.], [2., 0., 0.], [1., 0., 0.])

    def test_composition_retains_rows_and_rejects_discontinuous_joint_seams(self):
        a = np.array([[0., 0.], [.3, -.1], [.6, .2]])
        b = np.array([[.6, .2], [.4, .1], [0., 0.]])
        path = compose_skill("test", 1, [("out", a, .5), ("back", b, .5)], .01, .5)
        np.testing.assert_array_equal(path.qpos, np.vstack([a, b[1:]]))
        self.assertEqual(len(path.segment_labels), len(path.qpos)-1)
        np.testing.assert_allclose(path.time_s/.01, np.round(path.time_s/.01), atol=1e-9)
        speed = np.abs(np.diff(path.qpos, axis=0)/np.diff(path.time_s)[:, None]).max()
        self.assertLessEqual(speed, .5)
        broken = b.copy(); broken[0] += 1e-8
        with self.assertRaisesRegex(ValueError, "exact common joint seam"):
            compose_skill("broken", 1, [("out", a, .5), ("back", broken, .5)])


class SharedLibraryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library = build_shared_library()

    def test_identical_three_skills_and_one_guard_for_both_agents(self):
        library = self.library
        self.assertEqual(list(library.skills), ["guard", "attack", "parry"])
        self.assertEqual([s.action_id for s in library.skills.values()], [0, 1, 2])
        for name, skill in library.skills.items():
            np.testing.assert_array_equal(skill.qpos[0], library.guard_q)
            np.testing.assert_array_equal(skill.qpos[-1], library.guard_q)
            np.testing.assert_array_equal(skill.sample(-1.), library.guard_q)
            np.testing.assert_array_equal(skill.sample(skill.duration_s+1.), library.guard_q)
            self.assertTrue(library.reports[name]["candidate_passes_skill_geometry"])
        # All nine legal switches share an exact G seam, with no role reset.
        for first in library.skills.values():
            for second in library.skills.values():
                np.testing.assert_array_equal(first.sample(first.duration_s), second.sample(0.))

    def test_manual_recovery_is_exact_path_reverse_to_same_guard(self):
        for name, first, second in (("attack", "prepare", "unprepare"),
                                    ("attack", "thrust", "retract"),
                                    ("parry", "sweep", "recover")):
            path = self.library.skills[name]
            seg = {s["name"]: s for s in path.segments}
            a, b = seg[first], seg[second]
            forward = path.qpos[a["start_row"]:a["end_row"]+1]
            backward = path.qpos[b["start_row"]:b["end_row"]+1]
            np.testing.assert_array_equal(backward, forward[::-1])

    def test_actions_have_fixed_own_hold_and_cooldown_without_future_inputs(self):
        for name in ("attack", "parry"):
            skill = self.library.skills[name]
            self.assertEqual(skill.segments[-1]["name"], "cooldown")
            self.assertAlmostEqual(skill.segments[-1]["end_s"]-skill.segments[-1]["start_s"], self.library.config["cooldown_s"])
            self.assertEqual(list(inspect.signature(skill.sample).parameters), ["elapsed"])
            for segment in skill.segments:
                middle = .5*(segment["start_s"]+segment["end_s"])
                self.assertEqual(skill.phase_at(middle), segment["name"])
            self.assertEqual(skill.phase_at(skill.duration_s), "ready")
            with self.assertRaises(ValueError):
                skill.sample(np.nan)
        self.assertFalse(self.library.reports["library"]["opponent_future_inputs"])

    def test_all_raw_contacts_margins_and_command_clock_are_reported(self):
        library = self.library
        self.assertTrue(library.reports["library"]["all_skills_pass_geometry"])
        for name, path in library.skills.items():
            report = library.reports[name]
            self.assertEqual(report["all_raw_contacts"], [])
            self.assertGreater(report["proxy5_adapter_minimum_clearance_m"], 0.)
            self.assertGreaterEqual(report["minimum_true_joint_margin_rad"], .05-1e-8)
            self.assertEqual(report["static_gravity_exceeded_screen_indices"], [])
            self.assertGreaterEqual(report["screened_configurations"], len(path.qpos))
            self.assertTrue(np.all(np.diff(path.time_s) > 0))
            self.assertLessEqual(report["provisional_peak_command_speed_rad_s"], library.config["provisional_command_speed_rad_s"])

    def test_failed_ik_rows_are_retained_and_block_action_gate(self):
        def unreachable(model, grip, axes, bounds, *, initial_q, **kwargs):
            count = len(grip)
            q = np.tile(initial_q, (count, 1))
            return MatchResult(q, np.ones(count), np.zeros(count), np.ones(count),
                               np.zeros_like(q), np.zeros(count, dtype=bool),
                               np.ones(count, dtype=bool), np.zeros(count, dtype=bool),
                               [[] for _ in range(count)], np.ones(count, dtype=int),
                               np.ones(count, dtype=int))
        with patch("sim.shared_skills.build_floor_assembly", return_value=self.library.single_assembly), patch("sim.shared_skills.match_sequence", side_effect=unreachable):
            failed = build_shared_library({"samples_per_motion": 5})
        self.assertFalse(failed.reports["library"]["all_skills_pass_geometry"])
        for name in ("prepare", "thrust", "sweep"):
            self.assertEqual(failed.reports[name]["frames"], 5)
            self.assertEqual(failed.reports[name]["failed_frame_indices"], list(range(5)))
            self.assertTrue(failed.reports[name]["all_target_rows_retained"])
        self.assertFalse(failed.reports["attack"]["candidate_passes_skill_geometry"])
        self.assertFalse(failed.reports["parry"]["candidate_passes_skill_geometry"])

    def test_debug_motor_gravity_exceedance_is_diagnostic_not_geometry_rejection(self):
        assembly = self.library.single_assembly
        saved_caps = assembly.model.actuator_forcerange.copy()
        try:
            # Temporary in-memory diagnostic fixture, restored below. No model
            # asset or production cap is edited by this test.
            assembly.model.actuator_forcerange[:] = [-1e-6, 1e-6]
            report = _screen(assembly, np.tile(self.library.guard_q, (2, 1)), .05)
        finally:
            assembly.model.actuator_forcerange[:] = saved_caps
        self.assertFalse(report["static_gravity_within_debug_caps"])
        self.assertGreater(len(report["static_gravity_exceeded_screen_indices"]), 0)
        self.assertTrue(report["candidate_passes_geometry_screen"])
        self.assertTrue(report["candidate_passes_static_screen"])


class UprightThenTiltTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        offset = .20
        cls.library = build_shared_library({
            "attack_prepare_mode": "upright_then_tilt",
            "attack_ready_grip_local_m": [.45, offset, 1.05],
            "attack_extended_grip_local_m": [.65, offset, 1.05],
            "attack_axis_local": [np.sqrt(1-(offset/.79)**2), -offset/.79, 0.],
        })

    def test_optional_mode_keeps_default_and_has_separate_fixed_grip_tilt(self):
        self.assertEqual(DEFAULT_CONFIG["attack_prepare_mode"], "simultaneous")
        self.assertEqual(DEFAULT_CONFIG["attack_ready_grip_local_m"], [.30, 0., 1.05])
        library = self.library
        self.assertTrue(library.reports["library"]["all_skills_pass_geometry"])
        self.assertEqual(library.skills["attack"].qpos.shape, (483, 7))
        segments = [s["name"] for s in library.skills["attack"].segments]
        self.assertEqual(segments, ["prepare_upright", "prepare_tilt", "thrust", "extended_hold", "retract", "unprepare_tilt", "unprepare_upright", "cooldown"])
        tilt = np.asarray(library.reports["prepare_tilt"]["target_grip_local_m"])
        np.testing.assert_array_equal(tilt, np.tile([.45, .20, 1.05], (81, 1)))
        upright_axis = np.asarray(library.reports["prepare_upright"]["target_axis_local"])
        np.testing.assert_allclose(upright_axis, np.tile([0., 0., 1.], (81, 1)), atol=1e-5)

    def test_both_preparation_parts_recover_exactly_and_keep_shared_guard(self):
        library = self.library
        path = library.skills["attack"]
        segments = {s["name"]: s for s in path.segments}
        for name in ("upright", "tilt"):
            a, b = segments["prepare_"+name], segments["unprepare_"+name]
            np.testing.assert_array_equal(path.qpos[a["start_row"]:a["end_row"]+1][::-1],
                                          path.qpos[b["start_row"]:b["end_row"]+1])
        for skill in library.skills.values():
            np.testing.assert_array_equal(skill.qpos[0], library.guard_q)
            np.testing.assert_array_equal(skill.qpos[-1], library.guard_q)
        self.assertEqual(library.reports["attack"]["all_raw_contacts"], [])

    def test_unknown_preparation_mode_rejected_before_model_work(self):
        with patch("sim.shared_skills.build_floor_assembly") as model:
            with self.assertRaisesRegex(ValueError, "attack_prepare_mode"):
                build_shared_library({"attack_prepare_mode": "hidden-reset"})
            model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
