"""Pure reference-selection/clock tests; no OpenGL render needed."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sim.a1_triangle_video import display_times, geometry_from_report, load_curve, select_chapters
from sim.motion_spline import C2JointSpline


def action(region=None, *, passed=True, duration=2.0, start="center", end="left"):
    return {"kind": "attack" if region else "guard_transition_not_verified_parry",
            "target_region": region, "sampled_reference_pass": passed,
            "start_guard": start, "end_guard": end, "knot_time_s": [0, duration]}


class TriangleVideoTests(unittest.TestCase):
    def test_explicit_subset_preserves_order_and_rejects_failed(self):
        report={"actions":{"up":action("upper"),"down":action("upper"),"bad":action("upper",passed=False)}}
        selection=select_chapters(report,["down","up"])
        self.assertEqual(selection["selected_names"],["down","up"])
        self.assertEqual(selection["missing_attack_regions"],["left","right"])
        for names in (["bad"],["missing"],["up","up"],[],"up"):
            with self.assertRaises(ValueError):
                select_chapters(report,names)

    def test_full_clock_and_exact_endpoint_no_acceleration(self):
        for duration in (2.0, 1.123, 9.51):
            times = display_times(duration, 12)
            self.assertEqual(times[0], 0)
            self.assertEqual(times[-1], duration)
            self.assertTrue(np.all(np.diff(times) > 0))
            self.assertLessEqual(float(np.max(np.diff(times))), 1 / 12 + 1e-12)
            self.assertGreaterEqual((len(times) - 1) / 12, duration)
        for duration, fps in [(0, 12), (-1, 12), (float("nan"), 12), (float("inf"), 12), (None, 12), ("1", 12), (1, True), (1, 0), (1, 61)]:
            with self.subTest(duration=duration, fps=fps), self.assertRaises(ValueError):
                display_times(duration, fps)

    def test_selection_unknown_names_all_three_and_only_two_guard_edges(self):
        report = {"actions": {
            "new_left_slash": action("left", duration=3),
            "new_shorter_left": action("left", duration=2),
            "new_right_slash": action("right"),
            "new_upper_cut": action("upper"),
            "failed_upper": action("upper", passed=False, duration=0.1),
            "idle_same_guard": action(start="center", end="center", duration=0.1),
            "transition_one": action(start="left", end="right", duration=0.5),
            "duplicate_edge": action(start="left", end="right", duration=0.6),
            "transition_two": action(start="right", end="upper", duration=0.7),
            "transition_three": action(start="upper", end="center", duration=1),
        }}
        selection = select_chapters(report)
        self.assertEqual(selection["selected_names"], ["new_shorter_left", "new_right_slash", "new_upper_cut", "transition_one", "transition_two"])
        self.assertEqual(selection["missing_attack_regions"], [])
        self.assertTrue(selection["all_three_regions_have_sampled_reference"])
        self.assertIn("failed_upper", selection["excluded_failed_actions"])
        self.assertIn("new_left_slash", selection["passed_but_not_selected"])
        self.assertNotIn("idle_same_guard", selection["selected_names"])

    def test_missing_regions_never_substitute_failed_actions(self):
        selection = select_chapters({"actions": {"left_ok": action("left"), "upper_bad": action("upper", passed=False)}})
        self.assertEqual(selection["selected_names"], ["left_ok"])
        self.assertEqual(selection["missing_attack_regions"], ["right", "upper"])
        self.assertFalse(selection["all_three_regions_have_sampled_reference"])
        self.assertEqual(select_chapters({"actions": {}})["selected_names"], [])
        with self.assertRaises(ValueError):
            select_chapters({})

    def test_geometry_comes_from_report_not_live_config(self):
        geometry = {"base_distance": 1.8, "regions": {
            "left": {"center": [0.3, 0.2, 0.9], "radius": 0.08},
            "right": {"center": [0.3, -0.2, 0.9], "radius": 0.08},
            "upper": {"center": [0.3, 0, 1.2], "radius": 0.08},
        }}
        self.assertEqual(geometry_from_report({"scoring": geometry}).base_distance, 1.8)
        self.assertEqual(geometry_from_report({"geometry": geometry}).region("upper").center[2], 1.2)
        for report in ({}, {"scoring": {}}, {"geometry": None}):
            with self.assertRaises(ValueError):
                geometry_from_report(report)

    def test_reference_archive_matches_report_and_complete_saved_path(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            q = np.stack([np.zeros(7), np.full(7, 0.2)])
            curve = C2JointSpline([0, 2], q)
            times = np.linspace(0, 2, 21)
            spec = action("left")
            spec["knot_qpos"] = q.tolist()
            np.savez_compressed(source / "new_motion.npz", knot_time_s=[0, 2], knot_qpos=q,
                                time_s=times, qpos=curve.sample(times))
            loaded, path = load_curve(source, "new_motion", spec)
            self.assertEqual(loaded.duration_s, 2)
            self.assertEqual(path, source / "new_motion.npz")
            for name in ("../outside", "nested/file", "nested\\file", ""):
                with self.assertRaises(ValueError):
                    load_curve(source, name, spec)
            with self.assertRaises(ValueError):
                load_curve(source, "new_motion", {**spec, "sampled_reference_pass": False})
            with self.assertRaises(ValueError):
                load_curve(source, "new_motion", {**spec, "knot_time_s": [0, 3]})
            corrupted = curve.sample(times)
            corrupted[4, 2] += 0.01
            np.savez_compressed(source / "new_motion.npz", knot_time_s=[0, 2], knot_qpos=q,
                                time_s=times, qpos=corrupted)
            with self.assertRaisesRegex(ValueError, "saved checked samples"):
                load_curve(source, "new_motion", spec)
            np.savez_compressed(source / "new_motion.npz", knot_time_s=[0, 2], knot_qpos=q,
                                time_s=times[:-1], qpos=curve.sample(times[:-1]))
            with self.assertRaisesRegex(ValueError, "entire reference clock"):
                load_curve(source, "new_motion", spec)


if __name__ == "__main__":
    unittest.main()
