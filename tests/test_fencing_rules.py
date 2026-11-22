"""Physical blade rules: positive, negative, continuity and failure controls.

Synthetic sweeps test the mathematical scorer. One native adapter test checks
the real assembly's proxy dimensions/sites; neither certifies an A1 skill.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from sim.fencing_rules import (BLADE_START_SITE, RULES_VERSION, BladeSection,
                               FirstTouchScorer, blade_sections_from_scene,
                               point_cylinder_distance, _first_strictly_negative,
                               _UnresolvedSweep)


def section(x, *, radius=.1, z0=-1., z1=1., y=0., name="blade"):
    return BladeSection(np.array([x, y, z0]), np.array([x, y, z1]), radius, name)


def frame(first, second=None):
    return {"a1": (first,), "a2": (section(5., name="other") if second is None else second,)}


class FiniteCylinderGeometryTests(unittest.TestCase):
    def test_solid_side_cap_and_corner_distances(self):
        blade = section(0.)
        for point, expected in (([0, 0, 0], 0.), ([.3, 0, 0], .2),
                                ([0, 0, 1.4], .4), ([.4, 0, 1.4], .5)):
            with self.subTest(point=point):
                distance, closest = point_cylinder_distance(point, blade)
                self.assertAlmostEqual(distance, expected, places=12)
                self.assertAlmostEqual(np.linalg.norm(np.asarray(point)-closest), expected, places=12)

    def test_no_capsule_extension_behind_guard(self):
        blade = section(0., radius=.1, z0=0., z1=1.)
        distance, _ = point_cylinder_distance([0., 0., -.08], blade)
        self.assertAlmostEqual(distance, .08)
        # A capsule would contain this center for free; the real blade does not.
        scorer = FirstTouchScorer({"a1": [0, 0, -.08], "a2": [0, 0, 0]}, .05)
        self.assertEqual(scorer.update(frame(blade), frame(blade), 0., .002), [])
        self.assertIsNone(scorer.failure)
        self.assertTrue(scorer.armed["a1"])

    def test_snapshots_copy_arrays(self):
        start, end = np.array([0., 0., 0.]), np.array([0., 0., 1.])
        blade = BladeSection(start, end, .1)
        start[:] = 9
        np.testing.assert_array_equal(blade.start, [0, 0, 0])
        self.assertFalse(blade.start.flags.writeable)


class PhysicalFirstTouchTests(unittest.TestCase):
    def setUp(self):
        self.scorer = FirstTouchScorer({"a1": [0., 0., 0.], "a2": [0., 0., 0.]}, .2)

    def test_full_blade_midpoint_scores_with_tip_far_outside(self):
        before, after = frame(section(1.)), frame(section(0.))
        entries = self.scorer.update(before, after, 0., .002)
        self.assertIsNone(self.scorer.failure)
        self.assertEqual(len(entries), 1)
        event = entries[0]
        self.assertEqual(event["agent"], "a1")
        self.assertAlmostEqual(event["time_s"], .0014, places=9)
        self.assertEqual(event["blade_radius_m"], .1)
        self.assertLess(event["clearance_m"], 0.)
        self.assertEqual(self.scorer.round_result["winner"], "a1")
        self.assertEqual(self.scorer.round_result["points"], {"a1": 1, "a2": 0})

    def test_outside_to_outside_fast_tunneling_scores_and_rearms(self):
        entries = self.scorer.update(frame(section(-1.)), frame(section(1.)), 10., 12.)
        self.assertIsNone(self.scorer.failure)
        self.assertEqual(len(entries), 1)
        self.assertAlmostEqual(entries[0]["time_s"], 10.7, places=8)
        self.assertTrue(self.scorer.armed["a1"])
        self.assertEqual(len(self.scorer.events), 1)

    def test_initial_overlap_cannot_score_without_full_exit_and_reentry(self):
        self.assertEqual(self.scorer.update(frame(section(0.)), frame(section(.1)), 0., 1.), [])
        self.assertEqual(self.scorer.update(frame(section(.1)), frame(section(1.)), 1., 2.), [])
        entries = self.scorer.update(frame(section(1.)), frame(section(0.)), 2., 3.)
        self.assertIsNone(self.scorer.failure)
        self.assertEqual(len(entries), 1)
        self.assertEqual(len(self.scorer.initial_overlap_events), 1)
        self.assertGreater(entries[0]["time_s"], 2.)

    def test_zero_duration_initial_snapshot_arms_only_outside_blades(self):
        initial = frame(section(0.))
        self.assertEqual(self.scorer.update(initial, initial, 0., 0.), [])
        self.assertIsNone(self.scorer.failure)
        self.assertFalse(self.scorer.armed["a1"])
        self.assertTrue(self.scorer.armed["a2"])
        self.assertEqual(self.scorer.update(initial, frame(section(1.)), 0., .002), [])
        self.assertEqual(len(self.scorer.update(frame(section(1.)), frame(section(0.)), .002, .004)), 1)

    def test_zero_duration_initial_snapshot_cannot_hide_a_teleport(self):
        self.scorer.update(frame(section(1.)), frame(section(0.)), 0., 0.)
        self.assertEqual(self.scorer.failure["reason"], "initial_snapshot_cannot_move_at_zero_time")

    def test_same_step_opponent_entries_are_double_touch_despite_different_times(self):
        entries = self.scorer.update(frame(section(-1.), section(2., name="other")),
                                      frame(section(1.), section(-1., name="other")), 0., .002)
        self.assertIsNone(self.scorer.failure)
        self.assertEqual(len(entries), 2)
        self.assertTrue(all(event["double_touch"] for event in entries))
        self.assertNotEqual(entries[0]["time_s"], entries[1]["time_s"])
        self.assertEqual(self.scorer.round_result["status"], "double_touch")
        self.assertEqual(self.scorer.round_result["points"], {"a1": 0, "a2": 0})

    def test_round_frozen_but_late_target_entries_retained(self):
        self.scorer.update(frame(section(-1.)), frame(section(1.)), 0., 1.)
        entries = self.scorer.update(frame(section(1.), section(5., name="other")),
                                      frame(section(1.), section(0., name="other")), 1., 2.)
        self.assertIsNone(self.scorer.failure)
        self.assertEqual(len(entries), 1)
        self.assertFalse(entries[0]["decides_round"])
        self.assertEqual(len(self.scorer.events), 1)
        self.assertEqual(len(self.scorer.all_entries), 2)
        self.assertEqual(self.scorer.round_result["winner"], "a1")

    def test_clear_miss_and_static_weapon_contact_award_nothing(self):
        # No contact/phase argument exists. A physical weapon collision cannot
        # by itself manufacture a point or block immunity in this interface.
        near = section(.5)
        self.assertEqual(self.scorer.update(frame(near), frame(near), 0., .002), [])
        self.assertEqual(self.scorer.finish(.002)["status"], "timeout_draw")
        self.assertEqual(self.scorer.round_result["rules_version"], RULES_VERSION)

    def test_real_radius_used_not_old_tip_radius(self):
        thin_before = frame(section(.5, radius=.01))
        thin_after = frame(section(.22, radius=.01))
        self.assertEqual(self.scorer.update(thin_before, thin_after, 0., .002), [])
        wide = FirstTouchScorer({"a1": [0, 0, 0], "a2": [0, 0, 0]}, .2)
        self.assertEqual(len(wide.update(frame(section(.5, radius=.05)),
                                         frame(section(.22, radius=.05)), 0., .002)), 1)

    def test_three_sections_are_unioned_not_three_points(self):
        def sliced(x):
            return {"a1": tuple(section(x, z0=a, z1=b, name=str(i))
                                 for i, (a, b) in enumerate(((-1, -.3), (-.3, .3), (.3, 1)))),
                    "a2": (section(5., name="other"),)}
        entries = self.scorer.update(sliced(1.), sliced(0.), 0., .002)
        self.assertIsNone(self.scorer.failure)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["blade_section"], "1")


class FailureAccountingTests(unittest.TestCase):
    def setUp(self):
        self.scorer = FirstTouchScorer({"a1": [0, 0, 0], "a2": [0, 0, 0]}, .2)

    def test_nonfinite_state_is_invalid_not_timeout_draw(self):
        invalid = BladeSection([np.nan, 0, -1], [np.nan, 0, 1], .1)
        self.assertEqual(self.scorer.update(frame(section(1.)), frame(invalid), 0., .002), [])
        self.assertEqual(self.scorer.failure["reason"], "nonfinite_blade_state")
        self.assertEqual(self.scorer.finish(.002)["status"], "invalid")
        self.assertEqual(self.scorer.events, [])

    def test_interpolation_axis_collapse_is_not_treated_as_hit(self):
        start = BladeSection([0, 0, -1], [0, 0, 1], .1)
        flipped = BladeSection([0, 0, 1], [0, 0, -1], .1)
        self.scorer.update(frame(start), frame(flipped), 0., .002)
        self.assertEqual(self.scorer.failure["reason"], "interpolated_blade_axis_degenerate")
        self.assertEqual(self.scorer.round_result["status"], "invalid")

    def test_teleport_between_steps_invalidates_audit(self):
        self.scorer.update(frame(section(1.)), frame(section(.9)), 0., .002)
        self.scorer.update(frame(section(-1.)), frame(section(.9)), .002, .004)
        self.assertEqual(self.scorer.failure["reason"], "blade_state_discontinuity_between_steps")

    def test_gap_in_timestamps_is_invalid(self):
        self.scorer.update(frame(section(1.)), frame(section(.9)), 0., .002)
        self.scorer.update(frame(section(.9)), frame(section(.8)), .003, .004)
        self.assertEqual(self.scorer.failure["reason"], "noncontiguous_physics_step_times")

    def test_failure_after_first_touch_overrides_result_but_keeps_evidence(self):
        self.scorer.update(frame(section(-1.)), frame(section(1.)), 0., .002)
        self.scorer.invalidate("mujoco_warning", .003, {"warning": "bad qacc"})
        result = self.scorer.finish(.003)
        self.assertEqual(result["status"], "invalid")
        self.assertIsNone(result["winner"])
        self.assertEqual(result["points"], {"a1": 0, "a2": 0})
        self.assertEqual(result["provisional_result"]["winner"], "a1")
        self.assertEqual(len(self.scorer.events), 1)

    def test_unresolved_search_is_explicit_invalid_not_draw_or_point(self):
        with patch("sim.fencing_rules._MAX_EVALUATIONS", 0):
            self.scorer.update(frame(section(-1.)), frame(section(1.)), 0., .002)
        self.assertEqual(self.scorer.failure["reason"], "swept_distance_evaluation_budget_exhausted")
        self.assertEqual(self.scorer.finish(.002)["status"], "invalid")
        self.assertEqual(self.scorer.events, [])

    def test_uncertified_near_miss_without_inside_witness_is_not_a_hit_or_miss(self):
        # Actual positive clearance at every sample, but a deliberately loose
        # valid motion bound leaves a substep interval uncertified. The new
        # bracket continuation must still reject this, not silently miss it.
        with self.assertRaisesRegex(_UnresolvedSweep, "unresolved_boundary_or_substep_entry"):
            _first_strictly_negative(lambda fraction: 1e-12, 0., 1e-9, 1., evaluation_budget=[0])


class NativeBoundaryFixtureTests(unittest.TestCase):
    def test_native_one_ms_rotating_blade_crossing_keeps_uncertain_bracket_until_witness(self):
        # Exact native inputs captured at 13.122 -> 13.123 s in the unchanged
        # staged A1 attack / A2 Guard at dt=.001, decision=.02. Original
        # evidence: outputs/whole_blade_v2_independent_boundary_fixture/
        # native_boundary_fixture.json. Values are embedded so this regression
        # does not depend on ignored outputs, XML roundoff or a long replay.
        # Both an independent full native run and restoring the native 13.12s
        # q/v/ctrl then taking three native steps produced these same inputs.
        a1_before = [
            ([.64012921041043, .19412881464367562, 1.0549350138837033],
             [.8829480279144858, .1292894414504364, 1.0581197637034567]),
            ([.8829480279144859, .12928944145043642, 1.0581197637034567],
             [1.1257668454185417, .06445006825719718, 1.0613045135232102]),
            ([1.1257668454185417, .06445006825719721, 1.0613045135232104],
             [1.3685856629225974, -.00038930493604202865, 1.064489263342964]),
        ]
        a1_after = [
            ([.640207118154726, .1941224312766605, 1.054932269108436],
             [.8830260259504141, .12928326202991103, 1.0581142857449444]),
            ([.8830260259504141, .12928326202991103, 1.0581142857449444],
             [1.1258449337461023, .06444409278316154, 1.0612963023814528]),
            ([1.1258449337461023, .06444409278316157, 1.061296302381453],
             [1.3686638415417904, -.00039507646358791226, 1.0644783190179614]),
        ]
        a2_static = [
            ([1.2500002676941673, .24999983766142253, .7859595270529423],
             [1.2500002865098265, .24999988472125745, 1.0373064289267708]),
            ([1.2500002865098263, .24999988472125745, 1.0373064289267706],
             [1.2500003053254856, .24999993178109237, 1.2886533308005994]),
            ([1.2500003053254853, .24999993178109237, 1.2886533308005994],
             [1.2500003241411446, .24999997884092728, 1.5400002326744278]),
        ]
        radii = [.0222362, .0157789775, .011486075553695175]

        def native_sections(owner, rows):
            return tuple(BladeSection(start, end, radius, f"{owner}_shinai_blade_collision_{index}")
                         for index, ((start, end), radius) in enumerate(zip(rows, radii)))

        before = {"a1": native_sections("a1", a1_before), "a2": native_sections("a2", a2_static)}
        after = {"a1": native_sections("a1", a1_after), "a2": native_sections("a2", a2_static)}
        scorer = FirstTouchScorer({"a1": [1.43, 0., 1.05], "a2": [.32, 0., 1.05]}, .06)
        entries = scorer.update(before, after, 13.121999999998167, 13.122999999998166)
        self.assertIsNone(scorer.failure)
        self.assertEqual(len(entries), 1)
        event = entries[0]
        self.assertEqual(event["agent"], "a1")
        self.assertEqual(event["blade_section"], "a1_shinai_blade_collision_2")
        self.assertLess(event["clearance_m"], -1e-10)
        self.assertGreater(event["fraction"], .25)
        self.assertLess(event["fraction"], .27)
        lo, hi = event["entry_fraction_bracket"]
        # This native input failed the old implementation in a tiny earlier
        # interval with no witness. Continuing search finds a physical inside
        # point; the recorded bracket includes the earlier uncertain region.
        self.assertGreater(hi-lo, 1e-9)
        self.assertEqual(hi, event["fraction"])
        first, last = before["a1"][2], after["a1"][2]
        witness = BladeSection(first.start+hi*(last.start-first.start),
                               first.end+hi*(last.end-first.end), first.radius_m)
        measured, _ = point_cylinder_distance([1.43, 0., 1.05], witness)
        self.assertLess(measured, .06-1e-10)
        self.assertEqual(scorer.round_result["winner"], "a1")


class NativeAdapterTests(unittest.TestCase):
    def test_real_flat_cylinders_span_guard_far_to_tip_and_native_radii(self):
        import mujoco
        from sim.duel_scene import build_duel_scene

        scene = build_duel_scene(1.75)
        snapshots = blade_sections_from_scene(scene)
        self.assertEqual(BLADE_START_SITE, "shinai_guard_far")
        for owner, mapping in scene.agents.items():
            sections = snapshots[owner]
            self.assertEqual(len(sections), 3)
            np.testing.assert_allclose(sections[0].start,
                                       scene.data.site_xpos[mapping["sites"]["shinai_guard_far"]], atol=1e-12)
            np.testing.assert_allclose(sections[-1].end,
                                       scene.data.site_xpos[mapping["sites"]["shinai_tip"]], atol=1e-12)
            for first, last in zip(sections[:-1], sections[1:]):
                np.testing.assert_allclose(first.end, last.start, atol=1e-12)
            for section_ in sections:
                gid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM, section_.name)
                self.assertEqual(section_.radius_m, scene.model.geom_size[gid, 0])


if __name__ == "__main__":
    unittest.main()
