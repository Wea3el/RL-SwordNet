import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from sim.sword_orientation_audit import (
    ATTACKS, DEFENCES, DEFAULT_REQUIREMENTS, _actual_frame_indices, _json,
    audit_axes, audit_native_trace, audit_report,
    axis_metrics, requirements_config, world_to_actor_axes,
)


def synthetic_actions(axes=None):
    if axes is None:
        axes = [[.8, .6, 0.], [1., 0., 0.], [.8, -.6, 0.]]
    return {name: {"axes": axes, "family": name, "source_screen_pass": True,
                   "start_guard": "center", "end_guard": "center"}
            for name in ATTACKS + DEFENCES}


def tiny_mirrored_scene():
    """Seven nested hinges per arm without the project's large mesh scene."""
    arms = []
    for slot, attributes in (("a1", 'pos="0 0 0"'),
                             ("a2", 'pos="1.75 0 0" quat="0 0 0 1"')):
        parts = [f'<body name="{slot}_fixed_base" {attributes}>']
        for i in range(7):
            parts.append(f'<body><joint name="{slot}_J{i}" axis="0 0 1"/>'
                         '<geom type="sphere" size=".01" mass=".1"/>')
        parts += [f'<site name="{slot}_shinai_grip" pos="0 0 0"/>'
                  f'<site name="{slot}_shinai_tip" pos="1 0 0"/>', '</body>' * 8]
        arms.append(''.join(parts))
    return '<mujoco><compiler angle="radian"/><worldbody>' + ''.join(arms) + '</worldbody></mujoco>'


class SwordOrientationAuditTests(unittest.TestCase):
    def test_actor_frame_is_independent_of_the_mirrored_slot(self):
        a = np.array([[.8, .6, 0.], [.8, -.6, 0.]])
        rotation = np.diag([-1., -1., 1.])
        np.testing.assert_allclose(world_to_actor_axes(a, np.eye(3)),
                                   world_to_actor_axes(a @ rotation.T, rotation))
        # A world-Y comparison would falsely call physically equivalent slots opposite.
        self.assertGreater(a[0, 1], 0.)
        self.assertLess((a @ rotation.T)[0, 1], 0.)

    def test_camera_not_an_input_and_axis_not_a_velocity(self):
        result = axis_metrics([[.8, .6, 0.], [.8, -.6, 0.]])
        self.assertEqual(result["lateral_y_range"], [-.6, .6])
        self.assertAlmostEqual(result["maximum_pairwise_axis_angle_deg"], 73.7397952917)
        self.assertNotIn("successful_defence", result)
        self.assertNotIn("motion_direction", result)

    def test_wrapped_heading_extent_is_not_a_false_full_rotation(self):
        radians = np.deg2rad([179., -179.])
        result = axis_metrics(np.c_[np.cos(radians), np.sin(radians), [0., 0.]])
        self.assertAlmostEqual(result["heading_circular_extent_deg"], 2.)
        self.assertAlmostEqual(result["maximum_pairwise_axis_angle_deg"], 2.)

    def test_old_same_hemisphere_bank_fails_despite_nine_names(self):
        result = audit_axes(synthetic_actions([[.8, .6, 0.], [.6, .8, 0.]]),
                            {"center": [.5795, .6344, .5116]})
        self.assertEqual(result["status"], "fail")
        self.assertFalse(result["orientation_geometry_candidate_pass"])
        self.assertIn("neutral:center", result["failed_checks"])
        for group in ("all", "attack", "defence"):
            self.assertFalse(result["coverage"][group]["negative_side_covered"])

    def test_balanced_complete_synthetic_bank_passes_geometry_only(self):
        result = audit_axes(synthetic_actions(), {"center": [1., 0., 0.]})
        self.assertEqual(result["status"], "pass")
        self.assertTrue(result["orientation_geometry_candidate_pass"])
        self.assertNotIn("training_admitted", result)

    def test_attacks_cannot_cover_missing_defence_side(self):
        actions = synthetic_actions()
        for name in DEFENCES:
            actions[name]["axes"] = [[.8, .6, 0.]]
        result = audit_axes(actions, {"center": [1., 0., 0.]})
        self.assertTrue(result["coverage"]["all"]["pass"])
        self.assertTrue(result["coverage"]["attack"]["pass"])
        self.assertFalse(result["coverage"]["defence"]["pass"])
        self.assertFalse(result["orientation_geometry_candidate_pass"])

    def test_absent_and_rejected_families_are_incomplete(self):
        actions = synthetic_actions()
        del actions[DEFENCES[0]]
        actions[ATTACKS[0]]["source_screen_pass"] = False
        result = audit_axes(actions, {"center": [1., 0., 0.]})
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["missing_families"], [DEFENCES[0]])
        self.assertEqual(result["unscreened_or_failed_families"], [ATTACKS[0]])

    def test_rejected_or_unregistered_route_cannot_repair_coverage(self):
        actions = synthetic_actions([[.8, .6, 0.]])
        actions[ATTACKS[0]]["axes"] = [[.8, -.6, 0.]]
        actions[ATTACKS[0]]["source_screen_pass"] = False
        actions["unused"] = {"axes": [[.8, -.6, 0.]], "family": "unused",
                             "source_screen_pass": True, "start_guard": "x", "end_guard": "y"}
        result = audit_axes(actions, {"center": [1., 0., 0.]})
        self.assertFalse(result["coverage"]["all"]["negative_side_covered"])
        self.assertNotIn("unused", result["coverage"]["all"]["eligible_action_names"])

    def test_named_side_does_not_implicitly_force_shaft_orientation(self):
        guards = {"center": [1., 0., 0.], "left_high": [1., 0., 0.]}
        self.assertTrue(audit_axes(synthetic_actions(), guards)["orientation_geometry_candidate_pass"])
        declared = {"guard_axes": {"left_high": {"y_min": .25}}}
        result = audit_axes(synthetic_actions(), guards, declared)
        self.assertIn("guard:left_high", result["failed_checks"])
        self.assertFalse(result["orientation_geometry_candidate_pass"])

    def test_action_endpoints_anchors_intervals_are_explicit(self):
        rule = {"action_axes": {ATTACKS[0]: {
            "start": {"y_min": .25}, "end": {"y_max": -.25},
            "anchors": [{"phase": .5, "bounds": {"x_min": .99}}],
            "intervals": [{"phase": [0., .5], "bounds": {"y_min": 0.}}],
        }}}
        result = audit_axes(synthetic_actions(), {"center": [1., 0., 0.]}, rule)
        self.assertTrue(result["orientation_geometry_candidate_pass"])
        broken = copy.deepcopy(rule)
        broken["action_axes"][ATTACKS[0]]["end"] = {"y_min": .25}
        self.assertFalse(audit_axes(synthetic_actions(), {"center": [1., 0., 0.]}, broken)[
            "orientation_geometry_candidate_pass"])

    def test_route_expectations_do_not_create_hidden_bridges(self):
        rule = {"family_routes": {ATTACKS[0]: [["left_high", "right_low"]]}}
        result = audit_axes(synthetic_actions(), {"center": [1., 0., 0.]}, rule)
        self.assertIn("route:" + ATTACKS[0], result["failed_checks"])

    def test_missing_declared_guard_or_route_is_incomplete(self):
        result = audit_axes(synthetic_actions(), {"center": [1., 0., 0.]},
                            {"guard_axes": {"left_high": {"y_min": .25}},
                             "action_axes": {"not_present": {"start": {"x_min": .5}}}})
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(set(result["missing_declared_evidence"]), {"guard:left_high", "action:not_present"})

    def test_invalid_axes_frames_and_flags_are_rejected(self):
        for axes in ([], [1., 0., 0.], [[0., 0., 0.]], [[np.nan, 0., 0.]]):
            with self.subTest(axes=axes), self.assertRaises(ValueError):
                axis_metrics(axes)
        for rotation in (np.zeros((3, 3)), np.diag([-1., 1., 1.]), np.eye(2)):
            with self.assertRaises(ValueError):
                world_to_actor_axes([[1., 0., 0.]], rotation)
        for flag in (1, "true", None):
            actions = synthetic_actions()
            actions[ATTACKS[0]]["source_screen_pass"] = flag
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                audit_axes(actions, {"center": [1., 0., 0.]})

    def test_invalid_declarations_rejected_and_defaults_not_mutated(self):
        before = copy.deepcopy(DEFAULT_REQUIREMENTS)
        requirements_config({"guard_axes": {"left": {"y_min": .3}}})
        self.assertEqual(DEFAULT_REQUIREMENTS, before)
        for config in ({"surprise": True}, {"lateral_threshold": False},
                       {"neutral_axis": {"x_min": float("nan")}},
                       {"guard_axes": {"left": {"y_min": .3, "y_max": -.2}}},
                       {"action_axes": {"x": {"anchors": [{"phase": 2., "bounds": {"x_min": .5}}]}}},
                       {"action_axes": {"x": {"intervals": [{"phase": [.8, .2], "bounds": {"x_min": .5}}]}}}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                requirements_config(config)

    def test_json_and_report_format_errors_do_not_become_empty_pass(self):
        for text in ('{"actions":{},"actions":{}}', '{"x":NaN}', '{"x":Infinity}'):
            with self.assertRaises(ValueError):
                _json(text)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            for report in ([], {}, {"guards": {}, "actions": []}):
                path.write_text(json.dumps(report))
                with self.subTest(report=report), self.assertRaises(ValueError):
                    audit_report(path)
            for samples in (True, 2, 2002, 5.5):
                with self.subTest(samples=samples), self.assertRaises(ValueError):
                    audit_report(path, samples_per_action=samples)

    def test_lightweight_real_fk_report_checks_both_slots_and_hashes(self):
        zero = np.zeros(7)
        left, right = zero.copy(), zero.copy()
        left[0], right[0] = .5, -.5
        report = {"guards": {"center": {"qpos_rad": zero.tolist()}}, "actions": {}}
        for name in ATTACKS + DEFENCES:
            report["actions"][name] = {
                "start_guard": "center", "end_guard": "center", "candidate_pass": True,
                "knot_time_s": [0., 1., 2., 3., 4.],
                "knot_qpos": np.array([zero, left, zero, right, zero]).tolist(),
            }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "report.json"
            source.write_text(json.dumps(report))
            (Path(directory) / "scene.xml").write_text(tiny_mirrored_scene())
            result = audit_report(source, samples_per_action=9)
        self.assertTrue(result["source_unchanged"])
        self.assertTrue(result["mirrored_slot_invariance"]["pass"])
        self.assertTrue(result["orientation_geometry_candidate_pass"])
        self.assertFalse(result["native_dynamics_tested"])
        self.assertFalse(result["admitted_to_training"])
        self.assertAlmostEqual(result["actions"][ATTACKS[0]]["lateral_y_range"][0], -np.sin(.5))

    def test_native_uses_actual_frames_not_desired_or_interpolated_poses(self):
        times = np.array([0., .001, .3, .71, .9, .999, 1.])
        actual = np.zeros((len(times), 14))
        actual[:, 0] = [.1, .2, .3, .4, .5, .6, .7]
        actual[:, 7] = [-.1, -.2, -.3, -.4, -.5, -.6, -.7]
        desired = np.zeros_like(actual)
        desired[:, 0] = -1.5  # Opposite side from actual a1, never used.
        desired[:, 7] = 1.5
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "scene.xml").write_text(tiny_mirrored_scene())
            np.savez_compressed(path / "trace.npz", time_s=times, qpos=actual,
                                target_qpos=desired, desired_reference_qpos=desired)
            (path / "summary.json").write_text(json.dumps({
                "status": "failed", "native_execution_pass": False, "source_unchanged": True}))
            result = audit_native_trace(path, samples=3)
        indices = result["sample_indices"]
        self.assertEqual(indices, [0, 2, 6])
        np.testing.assert_array_equal(result["time_s"], times[indices])
        np.testing.assert_array_equal(result["sampled_actual_qpos"], actual[indices])
        self.assertFalse(result["commands_or_reference_used"])
        self.assertFalse(result["interpolation_used"])
        self.assertEqual(result["new_physics_steps_run"], 0)
        self.assertTrue(result["actors"]["a1"]["all_strictly_positive_y"])
        self.assertTrue(result["actors"]["a2"]["all_strictly_negative_y"])
        np.testing.assert_allclose(np.array(result["actors"]["a2"]["axis_actor_local"])[:, 1],
                                   np.sin(actual[indices, 7]), atol=1e-14)
        self.assertTrue(result["source_unchanged"])
        self.assertEqual(result["source_trial_status"], "failed")
        self.assertFalse(result["source_trial_native_execution_pass"])
        self.assertFalse(result["is_training_gate"])
        self.assertNotIn("orientation_geometry_candidate_pass", result)

    def test_native_selection_is_bounded_and_keeps_both_endpoints(self):
        times = np.linspace(0., 20., 20001)
        selected = _actual_frame_indices(times, 301)
        self.assertLessEqual(len(selected), 301)
        self.assertEqual(selected[0], 0)
        self.assertEqual(selected[-1], len(times) - 1)
        self.assertTrue(np.all(np.diff(selected) > 0))
        for count in (True, 1, 302, 3.5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                _actual_frame_indices(times, count)
        for values in ([0., 0.], [1., 0.], [-1., 0.], [0., float("nan")], [0.]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                _actual_frame_indices(values, 3)

    def test_native_malformed_actual_state_is_not_replaced_by_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "scene.xml").write_text(tiny_mirrored_scene())
            for arrays in (
                {"time_s": [0., 1.], "target_qpos": np.zeros((2, 14))},
                {"time_s": [0., 1.], "qpos": np.zeros((2, 7))},
                {"time_s": [0., 1.], "qpos": np.full((2, 14), np.nan)},
                {"time_s": [0., 0.], "qpos": np.zeros((2, 14))},
            ):
                np.savez_compressed(path / "trace.npz", **arrays)
                with self.assertRaises(ValueError):
                    audit_native_trace(path)


if __name__ == "__main__":
    unittest.main()
