"""Fast geometry, command, and causal-comparison tests; no physics rollouts."""
from __future__ import annotations

import copy
import inspect
import json
from types import SimpleNamespace
import unittest

import numpy as np

from sim.manual_duel import (
    DEFAULT_CONFIG, attack_command, evaluate_pair, minimum_duration,
    parry_command, schedule, swept_sphere_fraction,
)


def fake_library():
    phase = np.array([0., .2, 1.])
    attack = np.array([[0., .1], [.6, -.3], [.2, .2]])
    parry = np.array([[.1, -.2], [-.2, .2], [.5, -.1]])
    return SimpleNamespace(attack_phase=phase, attack_q=attack,
                           guard_attack_q=attack[0].copy(), parry_phase=phase,
                           parry_q=parry, guard_defend_q=parry[0].copy())


def valid_pair():
    common = {"status": "completed", "attack_target_command_sha256": "same-command",
              "initial_qpos": [0., 1., .5], "target_hit": True,
              "first_target_hit_time_s": 3., "first_positive_blade_contact_time_s": None,
              "blade_contact_normal_impulse_ns": 0.}
    baseline, defended = copy.deepcopy(common), copy.deepcopy(common)
    defended.update(target_hit=False, first_target_hit_time_s=None,
                    first_positive_blade_contact_time_s=2., blade_contact_normal_impulse_ns=.2)
    return baseline, defended


class SweptSphereTests(unittest.TestCase):
    def test_tunneling_detected_between_outside_endpoints(self):
        at = swept_sphere_fraction([-2., 0., 0.], [2., 0., 0.], [0., 0., 0.], .5)
        self.assertAlmostEqual(at, .375)

    def test_endpoint_and_tangent_contacts(self):
        self.assertAlmostEqual(swept_sphere_fraction([-1., 0., 0.], [-.5, 0., 0.], [0., 0., 0.], .5), 1.)
        self.assertAlmostEqual(swept_sphere_fraction([-2., 1., 0.], [2., 1., 0.], [0., 0., 0.], 1.), .5)
        self.assertEqual(swept_sphere_fraction([.5, 0., 0.], [1., 0., 0.], [0., 0., 0.], .5), 0.)

    def test_zero_length_and_missed_segments(self):
        self.assertEqual(swept_sphere_fraction([0., 0., 0.], [0., 0., 0.], [0., 0., 0.], 1.), 0.)
        self.assertIsNone(swept_sphere_fraction([2., 0., 0.], [2., 0., 0.], [0., 0., 0.], 1.))
        self.assertIsNone(swept_sphere_fraction([-2., 2., 0.], [2., 2., 0.], [0., 0., 0.], 1.))

    def test_invalid_inputs_rejected(self):
        for radius in (0., -1., np.nan, np.inf):
            with self.subTest(radius=radius), self.assertRaises(ValueError):
                swept_sphere_fraction([0., 0., 0.], [1., 0., 0.], [0., 0., 0.], radius)
        with self.assertRaises(ValueError):
            swept_sphere_fraction([0., 0.], [1., 0., 0.], [0., 0., 0.], 1.)


class CommandTests(unittest.TestCase):
    def setUp(self):
        self.library = fake_library()
        self.cfg = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        self.timing = schedule(self.library, self.cfg, 1., 0.)

    def test_minimum_duration_uses_largest_phase_derivative(self):
        # The first interval moves .6 rad in .2 phase: derivative 3 rad/phase.
        minimum = minimum_duration(self.library.attack_phase, self.library.attack_q, .1, .5)
        self.assertAlmostEqual(minimum, 3/.5*1.01)

    def test_attack_program_has_no_defence_switch_and_is_unaffected_by_parry_calls(self):
        self.assertNotIn("enabled", inspect.signature(attack_command).parameters)
        for time in np.linspace(0., self.timing["duration_s"], 30):
            baseline = attack_command(self.library, self.timing, time)
            parry_command(self.library, self.timing, time, False)
            no_defence = attack_command(self.library, self.timing, time)
            parry_command(self.library, self.timing, time, True)
            defended = attack_command(self.library, self.timing, time)
            np.testing.assert_array_equal(baseline, no_defence)
            np.testing.assert_array_equal(baseline, defended)

    def test_schedule_boundaries_and_return_to_guard(self):
        library, timing = self.library, self.timing
        np.testing.assert_array_equal(attack_command(library, timing, 0.), library.guard_attack_q)
        np.testing.assert_array_equal(attack_command(library, timing, timing["attack_start_s"]), library.guard_attack_q)
        np.testing.assert_allclose(attack_command(library, timing, timing["attack_start_s"]+timing["attack_duration_s"]), library.attack_q[-1])
        np.testing.assert_allclose(attack_command(library, timing, timing["attack_end_s"]), library.guard_attack_q, atol=1e-14)
        np.testing.assert_allclose(attack_command(library, timing, timing["duration_s"]+1.), library.guard_attack_q)
        np.testing.assert_array_equal(parry_command(library, timing, timing["parry_start_s"], True), library.guard_defend_q)
        np.testing.assert_allclose(parry_command(library, timing, timing["parry_start_s"]+timing["parry_duration_s"], True), library.parry_q[-1])
        np.testing.assert_allclose(parry_command(library, timing, timing["parry_release_s"]+timing["parry_return_s"], True), library.guard_defend_q, atol=1e-14)
        np.testing.assert_allclose(parry_command(library, timing, timing["duration_s"]+1., True), library.guard_defend_q)
        for time in np.linspace(0., timing["duration_s"], 10):
            np.testing.assert_array_equal(parry_command(library, timing, time, False), library.guard_defend_q)

    def test_command_speed_bound_holds_for_fast_request_and_both_returns(self):
        timing = schedule(self.library, self.cfg, 100., -.2)
        times = np.linspace(0., timing["duration_s"], 2001)
        for fn in (lambda t: attack_command(self.library, timing, t),
                   lambda t: parry_command(self.library, timing, t, True)):
            values = np.array([fn(t) for t in times])
            peak = np.max(np.abs(np.diff(values, axis=0)/np.diff(times)[:, None]))
            self.assertLessEqual(peak, self.cfg["command_speed_ceiling_rad_s"]+1e-9)

    def test_invalid_speed_and_start_time(self):
        for speed, delay in ((0., 0.), (-1., 0.), (np.nan, 0.), (1., np.inf), (1., -10.)):
            with self.subTest(speed=speed, delay=delay), self.assertRaises(ValueError):
                schedule(self.library, self.cfg, speed, delay)


class CausalPairTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))

    def test_accepts_completed_contact_caused_protection_with_identical_attack(self):
        result = evaluate_pair(*valid_pair(), self.cfg)
        self.assertTrue(result["valid_paired_experiment"])
        self.assertTrue(result["same_attack_targets"])
        self.assertTrue(result["same_initial_state"])
        self.assertTrue(result["blade_contact_precedes_baseline_hit"])
        self.assertTrue(result["successful_physical_parry"])

    def test_rejects_baseline_miss_contactless_defence_and_defended_hit(self):
        mutations = [
            ("baseline_miss", 0, {"target_hit": False, "first_target_hit_time_s": None}),
            ("no_actual_contact", 1, {"first_positive_blade_contact_time_s": None, "blade_contact_normal_impulse_ns": 0.}),
            ("defended_hit", 1, {"target_hit": True, "first_target_hit_time_s": 3.1}),
            ("contact_too_late", 1, {"first_positive_blade_contact_time_s": 3.1}),
            ("insufficient_impulse", 1, {"blade_contact_normal_impulse_ns": 0.}),
        ]
        for name, which, fields in mutations:
            with self.subTest(case=name):
                pair = valid_pair()
                pair[which].update(fields)
                self.assertFalse(evaluate_pair(*pair, self.cfg)["successful_physical_parry"])

    def test_rejects_changed_attack_initial_state_or_failed_trial(self):
        for which, fields in [(1, {"attack_target_command_sha256": "different"}),
                              (1, {"initial_qpos": [0., 0., 0.]}),
                              (0, {"status": "failed"}), (1, {"status": "failed"})]:
            with self.subTest(which=which, fields=fields):
                pair = valid_pair()
                pair[which].update(fields)
                result = evaluate_pair(*pair, self.cfg)
                self.assertFalse(result["valid_paired_experiment"])
                self.assertFalse(result["successful_physical_parry"])


if __name__ == "__main__":
    unittest.main()
