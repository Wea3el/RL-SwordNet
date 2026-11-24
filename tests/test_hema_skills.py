"""Separate longsword pose/cut library: geometry, seams and timing, no dual rollout."""
from __future__ import annotations

import unittest

import numpy as np

from sim.hema_skills import (CUT_ORDER, DEFAULT_CONFIG, POSE_ORDER, build_hema_library,
                             library_summary, quintic_inverse, waypoint_path)
from sim.shared_skills import quintic


class WaypointTimingTests(unittest.TestCase):
    def test_quintic_inverse_round_trips(self):
        for value in (0., .1, .37, .5, .9, 1.):
            self.assertAlmostEqual(float(quintic(quintic_inverse(value))), value, places=12)
        with self.assertRaises(ValueError):
            quintic_inverse(1.5)

    def test_waypoint_path_is_continuous_and_does_not_stop_at_interior_waypoints(self):
        pieces = waypoint_path([1., 2.], 41)
        self.assertEqual(pieces[0]["alpha"][0], 0.)
        self.assertEqual(pieces[0]["alpha"][-1], 1.)
        self.assertEqual(pieces[1]["alpha"][0], 0.)
        self.assertEqual(pieces[1]["alpha"][-1], 1.)
        self.assertAlmostEqual(pieces[0]["phase"][-1], pieces[1]["phase"][0])
        self.assertAlmostEqual(pieces[0]["phase_span"]+pieces[1]["phase_span"], 1.)
        # Global arc-length rate at the shared waypoint is the quintic slope there, not zero.
        seam = pieces[1]["phase"][0]
        self.assertGreater(30*seam**2*(1-seam)**2, .5)
        with self.assertRaises(ValueError):
            waypoint_path([1., 0.], 5)


class HemaLibraryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library = build_hema_library()

    def test_ten_poses_and_nine_cuts_with_stable_action_ids(self):
        library = self.library
        self.assertEqual(tuple(library.holds), POSE_ORDER)
        self.assertEqual(len(library.cuts), 9)
        self.assertTrue(all(cut in library.cuts for cut in CUT_ORDER))
        ids = library.action_ids
        self.assertEqual([ids[p] for p in POSE_ORDER], list(range(10)))
        self.assertEqual(sorted(ids.values()), list(range(19)))
        self.assertEqual(library.reports["library"]["pose_count"], 10)
        self.assertTrue(library.reports["library"]["separate_from_shared_and_ase_libraries"])

    def test_stationary_poses_are_zero_velocity_holds_that_pass_geometry(self):
        for name, path in self.library.holds.items():
            q = self.library.poses[name]
            np.testing.assert_array_equal(path.qpos, np.vstack([q, q]))
            np.testing.assert_array_equal(path.sample(.5*path.duration_s), q)
            report = self.library.reports["poses"][name]
            self.assertTrue(report["candidate_passes_pose_geometry"], name)
            self.assertTrue(report["velocity_zero_hold"])
            self.assertLess(report["position_error_m"], 2e-4)
            self.assertGreater(report["tip_above_floor_m"], .05, name)

    def test_sides_heights_and_blade_directions_match_the_pose_list(self):
        reached = self.library.pose_targets
        for name in POSE_ORDER:
            y = reached[name]["grip_local_m"][1]
            if name.endswith("_left"):
                self.assertGreater(y, .2, name)
            elif name.endswith("_right"):
                self.assertLess(y, -.2, name)
            else:
                self.assertAlmostEqual(y, 0., places=3)
        z = lambda n: reached[n]["grip_local_m"][2]
        self.assertGreater(z("kron"), z("roof_left"))
        self.assertGreater(z("roof_left"), z("ox_left"))
        self.assertGreater(z("ox_left"), z("long_point"))
        self.assertGreater(z("long_point"), z("plow_left"))
        self.assertGreater(z("plow_left"), z("fool_left"))
        axis = lambda n: np.asarray(reached[n]["axis_local"])
        for name in ("roof_left", "roof_right", "kron"):
            self.assertGreater(axis(name)[2], .8, name)          # pointed up
        for name in ("fool_left", "fool_right"):
            self.assertLess(axis(name)[2], -.5, name)            # point down
        for name in ("plow_left", "plow_right", "ox_left", "ox_right"):
            self.assertGreater(axis(name)[0], .9, name)          # point forward
        np.testing.assert_allclose(axis("long_point"), [1., 0., 0.], atol=1e-3)
        np.testing.assert_allclose(reached["plow_left"]["grip_local_m"][0], reached["plow_right"]["grip_local_m"][0], atol=1e-3)

    def test_cuts_run_exactly_from_start_pose_through_long_point_to_end_pose(self):
        cfg, library = self.library.config, self.library
        for name, path in library.cuts.items():
            report = library.reports["cuts"][name]
            chain = report["chain"]
            np.testing.assert_array_equal(path.qpos[0], library.poses[chain[0]])
            np.testing.assert_array_equal(path.qpos[-1], library.poses[chain[-1]])
            np.testing.assert_array_equal(path.sample(path.duration_s+1.), library.poses[chain[-1]])
            self.assertTrue(report["candidate_passes_cut_geometry"], name)
            self.assertTrue(report["exact_start_pose"] and report["exact_end_pose"])
            self.assertEqual(path.segments[-1]["name"], "hold_"+chain[-1])
            if "long_point" in chain[1:-1]:
                passage = report["long_point_passage"]["long_point"]
                self.assertTrue(passage["exact_pose_row"])
                np.testing.assert_array_equal(path.qpos[passage["row"]], library.poses["long_point"])
                self.assertGreater(passage["tip_speed_m_s"], cfg["minimum_cut_tip_speed_m_s"], name)
            self.assertTrue(report["tip_speed_requirement_met"], name)
            self.assertGreater(report["peak_tip_speed_m_s"], cfg["minimum_cut_tip_speed_m_s"])
        self.assertTrue(library.reports["library"]["all_cuts_pass_geometry"])
        self.assertTrue(library.reports["library"]["all_cuts_meet_tip_speed"])

    def test_cut_directions_follow_the_specification(self):
        chains = {name: r["chain"] for name, r in self.library.reports["cuts"].items()}
        self.assertEqual(chains["downward_cut_left_to_right"], ["roof_left", "long_point", "fool_right"])
        self.assertEqual(chains["downward_cut_right_to_left"], ["roof_right", "long_point", "fool_left"])
        self.assertEqual(chains["upward_cut_right_to_left"], ["fool_right", "long_point", "roof_left"])
        self.assertEqual(chains["upward_cut_left_to_right"], ["fool_left", "long_point", "roof_right"])
        self.assertEqual(chains["straight_down"], ["kron", "long_point"])
        self.assertEqual(chains["downward_cut_left_to_right_from_ox_left"][0], "ox_left")
        self.assertEqual(chains["upward_cut_left_to_right_from_plow_left"][0], "plow_left")
        tip = np.asarray(self.library.reports["cuts"]["downward_cut_left_to_right"]["tip_path_local_m"])
        self.assertGreater(tip[0, 2], tip[-1, 2])           # downward
        self.assertGreater(tip[0, 1], tip[-1, 1])           # left to right

    def test_every_pose_pair_has_an_exact_seam_transition_that_passes_geometry(self):
        library = self.library
        self.assertEqual(len(library.transitions), 90)
        self.assertEqual(library.reports["library"]["failed_transitions"], [])
        for (a, b), path in library.transitions.items():
            np.testing.assert_array_equal(path.qpos[0], library.poses[a])
            np.testing.assert_array_equal(path.qpos[-1], library.poses[b])
        # A cut's exact end pose is the exact start of any following transition or cut.
        for name, cut in library.cuts.items():
            end = library.reports["cuts"][name]["chain"][-1]
            for (a, b), path in library.transitions.items():
                if a == end:
                    np.testing.assert_array_equal(cut.sample(cut.duration_s), path.sample(0.))

    def test_summary_is_finite_json_without_raw_contact_dumps(self):
        summary = library_summary(self.library)
        self.assertEqual(set(summary["poses_q_rad"]), set(POSE_ORDER))
        self.assertNotIn("all_raw_contacts", summary["reports"]["cuts"]["straight_down"])
        self.assertEqual(summary["reports"]["library"]["status"], DEFAULT_CONFIG["status"])


class HemaConfigTests(unittest.TestCase):
    def test_invalid_configurations_are_rejected_explicitly(self):
        bad_pose = {"poses": {"only": {"grip_local_m": [.3, 0., 1.], "axis_local": [2., 0., 0.], "spec": ""}}, "cuts": {}}
        with self.assertRaisesRegex(ValueError, "unit blade axis"):
            build_hema_library(bad_pose)
        with self.assertRaisesRegex(ValueError, "unknown poses"):
            build_hema_library({"cuts": {"x": {"start": "roof_left", "via": [], "end": "nowhere", "spec": ""}}})
        with self.assertRaisesRegex(ValueError, "joint or task"):
            build_hema_library({"cut_path_space": "cartesian"})
        with self.assertRaisesRegex(ValueError, "positive finite"):
            build_hema_library({"cut_segment_s": 0.})

    def test_task_space_mode_reports_ik_and_settle_without_hiding_failures(self):
        cuts = {"straight_down": DEFAULT_CONFIG["cuts"]["straight_down"]}
        library = build_hema_library({"cut_path_space": "task", "cuts": cuts})
        report = library.reports["cuts"]["straight_down"]
        self.assertEqual(report["cut_path_space"], "task")
        self.assertEqual(len(report["task_ik"]), 1)
        self.assertTrue(report["exact_start_pose"] and report["exact_end_pose"])
        self.assertIn("settle_max_joint_delta_rad", report)
        self.assertEqual(report["candidate_passes_cut_geometry"],
                         report["candidate_passes_static_screen"] and report["task_ik"][0]["all_frames_passed"])


if __name__ == "__main__":
    unittest.main()
