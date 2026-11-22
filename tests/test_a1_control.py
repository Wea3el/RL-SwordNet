"""Finite A1 execution/feedback contracts, not physical skill certification."""
from __future__ import annotations

import copy
from types import SimpleNamespace
import unittest

import numpy as np

from sim.a1_control import A1Controller, A1ScriptedPolicy, A1SequencePolicy
from sim.shared_skills import SkillPath


def fixture():
    guard = np.linspace(-.2, .2, 7)
    skills = {}
    for i, name in enumerate(("guard", "stab", "slash_left", "parry_left", "parry_right")):
        q = np.vstack([guard, guard+i*.01, guard])
        skills[name] = SkillPath(name, i, q, np.array([0., .5, 1.]),
                                ["out", "back"], [{"name": "out", "start_s": 0., "end_s": .5},
                                                    {"name": "back", "start_s": .5, "end_s": 1.}])
    return SimpleNamespace(guard_q=guard, skills=skills)


class A1ControllerTests(unittest.TestCase):
    def setUp(self):
        self.library = fixture()
        self.controller = A1Controller(self.library)

    def feedback(self, now, q=None, v=None, controller=None):
        c = self.controller if controller is None else controller
        c.observe_own_state(self.library.guard_q if q is None else q,
                            np.zeros(7) if v is None else v, now)

    def test_feedback_is_required_for_entry_and_copied(self):
        c = self.controller
        self.assertEqual(c.state(0.)["action_mask"], [True, False, False, False, False])
        self.assertFalse(c.request(1, 0.))
        q, v = self.library.guard_q.copy(), np.zeros(7)
        c.observe_own_state(q, v, 0.)
        q[:] = 10; v[:] = 10
        self.assertEqual(c.state(0.)["action_mask"], [True]*5)
        self.assertTrue(c.request(1, 0.))
        np.testing.assert_array_equal(c.target(0.), self.library.guard_q)

    def test_stale_or_non_guard_feedback_blocks_new_skill(self):
        for delta_q, delta_v, age in ((.051, 0., 0.), (0., .101, 0.), (0., 0., .022)):
            c = A1Controller(self.library)
            self.feedback(0., self.library.guard_q+delta_q, np.full(7, delta_v), c)
            self.assertFalse(c.request(1, age))
            self.assertEqual(c.action_id, 0)
            self.assertEqual(c.diagnostics()["rejected_entry_requests"], 1)

    def test_busy_request_cannot_restart_or_cancel_and_zero_is_noop(self):
        c = self.controller
        self.feedback(0.)
        self.assertTrue(c.request(2, 0.))
        for a in (1, 2, 3, 4):
            self.assertFalse(c.request(a, .25))
        self.assertTrue(c.request(0, .25))
        self.assertEqual(c.started_s, 0.)
        self.assertEqual(c.action_id, 2)
        self.assertEqual(c.state(.25)["action_mask"], [True, False, False, False, False])
        np.testing.assert_array_equal(c.target(.25), self.library.skills["slash_left"].sample(.25))
        self.assertEqual(c.ignored_busy_requests, 4)
        self.assertEqual(len(c.events), 1)

    def test_source_end_is_not_actual_completion_and_return_holds_guard(self):
        c = self.controller
        self.feedback(0.)
        c.request(1, 0.)
        c.observe_own_state(self.library.guard_q+.2, np.zeros(7), 1.)
        self.assertTrue(c.state(1.)["busy"])
        self.assertEqual(c.state(1.)["phase"], "return_settling")
        np.testing.assert_array_equal(c.target(1.2), self.library.guard_q)
        self.assertFalse(c.request(3, 1.2))
        self.assertEqual(c.completions, [])

    def test_measured_settle_dwell_is_continuous_and_allows_next_skill(self):
        c = self.controller
        self.feedback(0.)
        c.request(1, 0.)
        self.feedback(1.)
        self.feedback(1.05, v=np.full(7, .101))  # restart dwell
        for t in np.arange(1.10, 1.20, .01):
            self.feedback(float(t))
        self.assertTrue(c.state(1.19)["busy"])
        self.feedback(1.20)
        self.assertFalse(c.state(1.20)["busy"])
        self.assertTrue(c.request(4, 1.20))
        np.testing.assert_array_equal(c.target(1.20), self.library.guard_q)
        self.assertEqual(c.completions[0]["action_id"], 1)
        self.assertAlmostEqual(c.completions[0]["settle_duration_s"], .2)

    def test_future_clock_cannot_fake_completion(self):
        c = self.controller
        self.feedback(0.)
        c.request(1, 0.)
        self.assertTrue(c.state(1.5)["busy"])
        self.assertFalse(c.request(2, 1.5))
        self.assertEqual(c.completions, [])

    def test_failed_exit_times_out_and_blocks_all_new_motion(self):
        c = A1Controller(self.library, settle_timeout_s=.5)
        self.feedback(0., controller=c)
        c.request(1, 0.)
        self.feedback(1.5, self.library.guard_q+.2, controller=c)
        self.assertEqual(c.failure, "a1_guard_settle_timeout")
        self.assertEqual(c.state(1.5)["phase"], "fault")
        self.assertFalse(c.request(2, 1.5))
        np.testing.assert_array_equal(c.target(1.5), self.library.guard_q)

    def test_slot_independent_instances_share_no_clock(self):
        a, b = A1Controller(self.library), A1Controller(self.library)
        self.feedback(0., controller=a); self.feedback(.5, controller=b)
        a.request(3, 0.); b.request(3, .5)
        np.testing.assert_array_equal(a.target(.25), b.target(.75))
        self.assertEqual(a.state(.25)["phase"], b.state(.75)["phase"])
        for t in np.linspace(1., 1.1, 11):
            self.feedback(float(t), controller=a)
        self.assertFalse(a.state(1.1)["busy"])
        self.assertTrue(b.state(.75)["busy"])

    def test_feedback_gap_cannot_count_as_continuous_dwell_or_bypass_timeout(self):
        for last in (1.2, 4.1):
            c = A1Controller(self.library)
            self.feedback(0., controller=c)
            c.request(1, 0.)
            self.feedback(1., controller=c)
            self.feedback(last, controller=c)
            self.assertEqual(c.completions, [])
            self.assertTrue(c.state(last)["busy"])
            if last > 4.:
                self.assertEqual(c.failure, "a1_guard_settle_timeout")
            else:
                self.assertIsNone(c.failure)

    def test_diagnostics_has_no_opponent_state_or_training_claim(self):
        d = self.controller.diagnostics()
        self.assertFalse(d["opponent_inputs"])
        self.assertFalse(d["admitted_to_training"])
        self.assertEqual(d["action_names"][4], "parry_right")
        self.assertTrue(d["robot_slot_independent"])

    def test_invalid_ids_times_arrays_and_nonclosed_paths_rejected(self):
        for a in (-1, 5, True, False, 1., "stab", None, np.nan):
            with self.subTest(action=a), self.assertRaises(ValueError):
                self.controller.request(a, 0.)
        for time in (-1., np.nan, np.inf, True):
            with self.subTest(time=time), self.assertRaises(ValueError):
                self.controller.state(time)
        self.controller.state(1.)
        with self.assertRaises(ValueError):
            self.controller.target(.9)
        with self.assertRaises(ValueError):
            self.controller.observe_own_state(np.zeros(6), np.zeros(7), 1.)
        bad = copy.deepcopy(self.library)
        bad.skills["stab"].qpos[-1, 0] += .001
        with self.assertRaisesRegex(ValueError, "EXACT common Guard"):
            A1Controller(bad)
        for kwargs in ({"settle_timeout_s": .01}, {"settle_dwell_s": 0.},
                       {"position_tolerance_rad": np.nan}, {"maximum_feedback_age_s": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                A1Controller(self.library, **kwargs)


class A1DiagnosticPolicyTests(unittest.TestCase):
    def setUp(self):
        self.library = fixture()

    def test_script_issues_each_request_once_without_hidden_state(self):
        p = A1ScriptedPolicy(self.library, [(1., 4), (.5, 2)])
        self.assertEqual(p.decide({"time_s": .49}), 0)
        self.assertEqual(p.decide({"time_s": .5}), 2)
        self.assertEqual(p.decide({"time_s": .8}), 0)
        self.assertEqual(p.decide({"time_s": 1.}), 4)
        self.assertEqual(p.decide({"time_s": 2.}), 0)
        for event in ((0., 5), (0., True), (-1., 1), (np.nan, 1), (0., 1.)):
            with self.assertRaises(ValueError):
                A1ScriptedPolicy(self.library, [event])

    def test_sequence_waits_for_own_completion_and_mask_not_other_actions(self):
        p = A1SequencePolicy(self.library, [1, 4], start_s=.5, gap_s=.2)
        def obs(time, busy=False, mask=None):
            return {"time_s": time, "own_skill": {"busy": busy, "action_mask": [True]*5 if mask is None else mask}}
        self.assertEqual(p.decide(obs(.4)), 0)
        self.assertEqual(p.decide(obs(.5)), 1)
        self.assertEqual(p.decide(obs(1., True)), 0)
        self.assertEqual(p.decide(obs(2.)), 0)
        self.assertEqual(p.decide(obs(2.2, mask=[True, False, False, False, False])), 0)
        self.assertEqual(p.decide(obs(2.3)), 4)
        self.assertEqual(p.decide(obs(5.)), 0)


if __name__ == "__main__":
    unittest.main()
