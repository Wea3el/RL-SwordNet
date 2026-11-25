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
                               _UnresolvedSweep, _point_cylinder_distances,
                               _motion_bound_arrays)


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


def scalar_distances(point, starts, ends, radii):
    pairs = [point_cylinder_distance(point, BladeSection(a, b, r))
             for a, b, r in zip(starts, ends, radii)]
    return np.asarray([p[0] for p in pairs]), np.asarray([p[1] for p in pairs])


class BatchGeometryEquivalenceTests(unittest.TestCase):
    def test_random_cylinders_match_retained_scalar_formula(self):
        rng = np.random.default_rng(20261125)
        starts = rng.normal(size=(250, 3))
        ends = starts+rng.normal(size=(250, 3))
        radii = rng.uniform(.001, .2, 250)
        for point in rng.normal(size=(12, 3)):
            expected_d, expected_p = scalar_distances(point, starts, ends, radii)
            actual_d, actual_p = _point_cylinder_distances(point, starts, ends, radii)
            np.testing.assert_allclose(actual_d, expected_d, rtol=2e-14, atol=2e-14)
            np.testing.assert_allclose(actual_p, expected_p, rtol=2e-14, atol=2e-14)

    def test_axis_cap_corner_zero_radial_and_equal_witness_rows(self):
        starts = np.tile([0., 0., 0.], (4, 1))
        ends = np.tile([0., 0., 1.], (4, 1))
        radii = np.full(4, .1)
        for point in ([0., 0., .5], [0., 0., 1.4], [.4, 0., 1.4], [.1, 0., .5]):
            d, p = _point_cylinder_distances(point, starts, ends, radii)
            expected_d, expected_p = scalar_distances(point, starts, ends, radii)
            np.testing.assert_allclose(d, expected_d, atol=1e-15)
            np.testing.assert_allclose(p, expected_p, atol=1e-15)
            self.assertEqual(int(np.argmin(d)), 0)
        bad_ends = ends.copy(); bad_ends[2] = starts[2]
        with self.assertRaises(ValueError):
            _point_cylinder_distances([0, 0, 0], starts, bad_ends, radii)
        with np.errstate(over='ignore', invalid='ignore'), self.assertRaises(ValueError):
            _point_cylinder_distances([0, 0, 0], np.zeros((1, 3)),
                                     np.full((1, 3), 1e308), np.array([.01]))

    def test_equal_distance_sweep_keeps_first_section_as_witness(self):
        def duplicate_frame(x):
            return {'a1': (section(x, name='first'), section(x, name='second')),
                    'a2': (section(4., name='other'),)}
        scorer = FirstTouchScorer({'a1': [0, 0, 0], 'a2': [0, 0, 0]}, .2)
        hits = scorer.update(duplicate_frame(1.), duplicate_frame(0.), 0., .002)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]['blade_section'], 'first')

    def test_batch_lipschitz_bound_matches_scalar_and_rejects_axis_collapse(self):
        rng = np.random.default_rng(8)
        for _ in range(20):
            a = rng.normal(size=(105, 3)); b = a+rng.normal(size=(105, 3))
            c = a+rng.normal(scale=.01, size=(105, 3)); d = b+rng.normal(scale=.01, size=(105, 3))
            radii = rng.uniform(.005, .03, 105)
            bounds = []
            for sa, ea, sb, eb, radius in zip(a, b, c, d, radii):
                axis = ea-sa; delta = (eb-sb)-axis; dd = float(delta@delta)
                f = float(np.clip(-(axis@delta)/dd, 0., 1.)) if dd else 0.
                minimum = float(np.linalg.norm(axis+f*delta))
                bounds.append(max(np.linalg.norm(sb-sa), np.linalg.norm(eb-ea))
                              + radius*np.linalg.norm(delta)/minimum)
            self.assertAlmostEqual(_motion_bound_arrays(a, b, c, d, radii), max(bounds), places=13)
        with self.assertRaisesRegex(_UnresolvedSweep, 'interpolated_blade_axis_degenerate'):
            _motion_bound_arrays(np.array([[0., 0., 0.]]), np.array([[0., 0., 1.]]),
                                 np.array([[0., 0., 1.]]), np.array([[0., 0., 0.]]), np.array([.01]))

    def test_complete_random_sweeps_keep_scalar_events_and_round_outcome(self):
        rng = np.random.default_rng(2026)
        for trial in range(12):
            count = 3+trial
            y = rng.uniform(-.35, .35, count)
            radii = rng.uniform(.006, .03, count)
            def snapshot(x):
                return {'a1': tuple(BladeSection([x, yi, -.2], [x, yi, .2], r, f'b{i}')
                                    for i, (yi, r) in enumerate(zip(y, radii))),
                        'a2': (section(4., name='other'),)}
            before, after = snapshot(-.7), snapshot(.7)
            fast = FirstTouchScorer({'a1': [0, 0, 0], 'a2': [0, 0, 0]}, .12)
            reference = FirstTouchScorer({'a1': [0, 0, 0], 'a2': [0, 0, 0]}, .12)
            got = fast.update(before, after, 0., .002)
            with patch('sim.fencing_rules._point_cylinder_distances', side_effect=scalar_distances):
                expected = reference.update(before, after, 0., .002)
            self.assertEqual(fast.failure, reference.failure)
            self.assertEqual(fast.round_result['status'], reference.round_result['status'])
            self.assertEqual(fast.round_result['winner'], reference.round_result['winner'])
            self.assertEqual(len(got), len(expected))
            for actual, scalar in zip(got, expected):
                self.assertEqual(actual['blade_section'], scalar['blade_section'])
                self.assertAlmostEqual(actual['time_s'], scalar['time_s'], places=11)
                np.testing.assert_allclose(actual['blade_closest_point_world_m'],
                                           scalar['blade_closest_point_world_m'], atol=1e-10)


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
    @staticmethod
    def synthetic_scene(indices, *, kind="cylinder"):
        import mujoco
        from types import SimpleNamespace

        sections = ''.join(
            f'<geom name="a1_shinai_blade_collision_{i}" type="{kind}" '
            f'size=".01 .1" pos="0 0 {i*.2+.1}" mass=".01"/>' for i in indices)
        legacy = ''.join(
            f'<geom name="a2_shinai_blade_collision_{i}" type="cylinder" '
            f'size=".01 .1" pos="0 0 {i*.2+.1}"/>' for i in range(3))
        model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><body pos="1 0 0"><joint type="slide" axis="1 0 0"/>'
            + sections + '</body><body pos="5 0 0">' + legacy + '</body></worldbody></mujoco>')
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return SimpleNamespace(model=model, data=data,
                               agents={s: {"prefix": s+"_"} for s in ("a1", "a2")})

    def test_historical_three_section_xml_remains_supported(self):
        scene = self.synthetic_scene([0, 1, 2])
        frames = blade_sections_from_scene(scene)
        self.assertEqual([s.name for s in frames['a1']],
                         [f'a1_shinai_blade_collision_{i}' for i in range(3)])
        self.assertEqual(len(frames['a2']), 3)

    def test_every_profile_section_is_read_in_numeric_not_xml_or_lexical_order(self):
        scene = self.synthetic_scene([0, 1, 10, 11, 2, 3, 4, 5, 6, 7, 8, 9])
        before = blade_sections_from_scene(scene)
        self.assertEqual([s.name for s in before['a1']],
                         [f'a1_shinai_blade_collision_{i}' for i in range(12)])
        # Only segment 11 can reach this sphere. Reading a fixed first three
        # segments would silently turn this physical whole-blade hit into a miss.
        import mujoco
        scene.data.qpos[0] = -1.
        mujoco.mj_forward(scene.model, scene.data)
        after = blade_sections_from_scene(scene)
        scorer = FirstTouchScorer({'a1': [0., 0., 2.3], 'a2': [0., 0., 0.]}, .03)
        hits = scorer.update(before, after, 0., .002)
        self.assertIsNone(scorer.failure)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]['blade_section'], 'a1_shinai_blade_collision_11')

    def test_malformed_profile_topology_fails_closed(self):
        for scene in (self.synthetic_scene([0, 2]),
                      self.synthetic_scene([0, 1, 2], kind='capsule')):
            with self.subTest(scene=scene), self.assertRaises(ValueError):
                blade_sections_from_scene(scene)

    def test_real_flat_cylinders_span_guard_far_to_tip_and_native_radii(self):
        import mujoco
        from sim.duel_scene import build_duel_scene

        scene = build_duel_scene(1.75)
        snapshots = blade_sections_from_scene(scene)
        self.assertEqual(BLADE_START_SITE, "shinai_guard_far")
        for owner, mapping in scene.agents.items():
            sections = snapshots[owner]
            self.assertGreaterEqual(len(sections), 3)
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
