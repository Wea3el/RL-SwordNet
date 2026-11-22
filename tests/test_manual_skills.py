"""Deterministic, source-dataset-independent manual trajectory screening."""
import unittest
from unittest.mock import patch

import numpy as np

from sim.manual_skills import build_manual_library, smoothstep5, task_reference


class ManualReferenceTests(unittest.TestCase):
    def test_quintic_endpoints_and_constant_axis(self):
        phase, grip, axis = task_reference([.3, 0., 1.05], [.65, 0., 1.05], [1., 0., 0.], 21)
        np.testing.assert_array_equal(grip[[0, -1]], [[.3, 0., 1.05], [.65, 0., 1.05]])
        np.testing.assert_array_equal(axis, np.tile([1., 0., 0.], (21, 1)))
        np.testing.assert_allclose(smoothstep5([0., .5, 1.]), [0., .5, 1.])
        self.assertTrue(np.all(np.diff(phase) > 0))

    def test_invalid_reference_is_rejected(self):
        for samples in (1, True, 2.5):
            with self.assertRaises(ValueError):
                task_reference([0., 0., 0.], [1., 0., 0.], [1., 0., 0.], samples)
        with self.assertRaises(ValueError):
            task_reference([0., 0., 0.], [1., 0., 0.], [2., 0., 0.])


class ManualModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # These fail loudly if a future implementation calls data conversion.
        with patch("plan1_skills.library.build_library", side_effect=AssertionError("No datasets allowed")), patch("plan1_skills.library.load_manifest", side_effect=AssertionError("No datasets allowed")):
            cls.library = build_manual_library()

    def test_forward_paths_all_frames_pass_single_arm_screen(self):
        library = self.library
        self.assertEqual(library.attack_q.shape, (81, 7))
        self.assertEqual(library.parry_q.shape, (81, 7))
        for name in ("manual_thrust", "manual_side_parry"):
            report = library.reports[name]
            self.assertTrue(report["all_frames_retained"])
            self.assertTrue(report["candidate_passes_single_arm_screen"], report)
            self.assertEqual(report["failed_frame_indices"], [])
            self.assertTrue(report["strict_static_contacts_free"])
            self.assertEqual(report["all_raw_static_contacts"], [])
            self.assertGreater(report["proxy5_adapter_minimum_clearance_m"], 0.)
        np.testing.assert_array_equal(library.guard_attack_q, library.attack_q[0])
        np.testing.assert_array_equal(library.guard_defend_q, library.parry_q[0])
        # Keep the redundant axial-roll solution far from the former J6~1.588
        # branch, which caused a 0.997 mm self-contact accepted by old tolerance.
        self.assertTrue(np.all(np.abs(library.parry_q[:, 6]) < 1.))

    def test_geometry_has_clear_initial_guards_and_intercept_before_target(self):
        library = self.library
        distance = library.config["base_distance_m"]
        # A2 yaw pi flips X/Y but leaves Z unchanged.
        a2_guard = np.array([distance, 0., 0.])+library.parry_grip_local_m[0]*[-1, -1, 1]
        barrier = np.array([distance, 0., 0.])+library.parry_grip_local_m[-1]*[-1, -1, 1]
        self.assertGreater(abs(a2_guard[1]), .2)
        # The parry crosses the attack line instead of parking a thin blade on
        # it. Round 1's static centred barrier contacted but did not protect.
        self.assertGreater(a2_guard[1], 0.)
        self.assertLess(barrier[1], 0.)
        self.assertLess(barrier[0], library.config["target_world_m"][0])
        self.assertLess(barrier[2], library.config["target_world_m"][2])
        self.assertGreater(barrier[2]+.79, library.config["target_world_m"][2])


if __name__ == "__main__":
    unittest.main()
