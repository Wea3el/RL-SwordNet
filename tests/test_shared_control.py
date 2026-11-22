"""Independent own-clock and outside-entry accounting tests; no MuJoCo run."""
from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from sim.shared_control import EntryScorer, SkillController
from sim.shared_skills import compose_skill


def library_fixture():
    guard = np.array([.1, -.1])
    ready, extended = guard+[.1, .05], guard+[.3, -.05]
    hold = lambda q: np.vstack([q, q])
    prepare, thrust = np.vstack([guard, ready]), np.vstack([ready, extended])
    parry = np.vstack([guard, guard+[-.1, .2]])
    return SimpleNamespace(guard_q=guard, skills={
        "guard": compose_skill("guard", 0, [("guard", hold(guard), .2)]),
        "attack": compose_skill("attack", 1,
                    [("prepare", prepare, .2), ("thrust", thrust, .4),
                     ("extended_hold", hold(extended), .1), ("retract", thrust[::-1], .4),
                     ("unprepare", prepare[::-1], .2), ("cooldown", hold(guard), .3)]),
        "parry": compose_skill("parry", 2,
                   [("sweep", parry, .4), ("parry_hold", hold(parry[-1]), .2),
                    ("recover", parry[::-1], .4), ("cooldown", hold(guard), .3)]),
    })


def state(attack=1, active=True, phase="thrust"):
    return {"attack_id": attack, "strike_active": active, "phase": phase}


class OwnSkillClockTests(unittest.TestCase):
    def setUp(self):
        self.library = library_fixture()

    def test_busy_requests_cannot_repeat_reset_or_cancel_an_action(self):
        controller = SkillController(self.library)
        self.assertTrue(controller.request(1, 0.))
        attack = self.library.skills["attack"]
        for request in (1, 2, 0, 1):
            self.assertFalse(controller.request(request, .3))
            self.assertEqual(controller.started_s, 0.)
            self.assertEqual(controller.action_id, 1)
            self.assertEqual(controller.attack_id, 1)
            np.testing.assert_array_equal(controller.target(.3), attack.sample(.3))
        self.assertEqual(len(controller.events), 1)
        self.assertEqual(controller.ignored_busy_requests, 4)
        self.assertTrue(controller.state(attack.duration_s-.01)["busy"])
        self.assertFalse(controller.request(2, attack.duration_s-.01))
        self.assertTrue(controller.request(2, attack.duration_s))
        self.assertEqual(controller.action_id, 2)

    def test_two_agent_clocks_are_independent_on_same_shared_library(self):
        first, second = SkillController(self.library), SkillController(self.library)
        first.request(1, 0.)
        second.request(1, .5)
        np.testing.assert_allclose(first.target(.3), second.target(.8), atol=1e-15, rtol=0.)
        self.assertEqual(first.state(.3)["phase"], second.state(.8)["phase"])
        # Advancing the other controller to completion cannot advance/reset A1.
        second.state(100.)
        self.assertTrue(first.state(.3)["busy"])
        self.assertEqual(first.started_s, 0.)
        self.assertFalse(second.state(100.)["busy"])

    def test_exact_guard_entry_exit_and_action_masks(self):
        controller = SkillController(self.library)
        np.testing.assert_array_equal(controller.target(0.), self.library.guard_q)
        self.assertEqual(controller.state(0.)["action_mask"], [True]*3)
        for action, name in ((1, "attack"), (2, "parry")):
            controller = SkillController(self.library)
            self.assertTrue(controller.request(action, 0.))
            np.testing.assert_array_equal(controller.target(0.), self.library.guard_q)
            self.assertEqual(controller.state(0.)["action_mask"], [True, False, False])
            end = self.library.skills[name].duration_s
            np.testing.assert_array_equal(controller.target(end), self.library.guard_q)
            self.assertEqual(controller.state(end)["action_mask"], [True]*3)
            self.assertEqual(controller.state(end)["phase"], "guard")

    def test_action_mask_always_has_guard_continue_for_future_masked_policy(self):
        controller = SkillController(self.library)
        self.assertTrue(any(controller.state(0.)["action_mask"]))
        controller.request(1, 0.)
        for now in np.linspace(0., self.library.skills["attack"].duration_s, 30):
            mask = controller.state(now)["action_mask"]
            self.assertTrue(any(mask))
            self.assertTrue(mask[0])
            if controller.state(now)["busy"]:
                self.assertEqual(mask, [True, False, False])

    def test_scoring_active_only_during_declared_attack_phases(self):
        controller = SkillController(self.library)
        controller.request(1, 0.)
        for segment in self.library.skills["attack"].segments:
            middle = .5*(segment["start_s"]+segment["end_s"])
            expected = segment["name"] in {"thrust", "extended_hold", "retract"}
            self.assertEqual(controller.state(middle)["strike_active"], expected)

    def test_invalid_actions_and_times_rejected(self):
        controller = SkillController(self.library)
        for action in (-1, 3, True, False, "attack", None, np.nan):
            with self.subTest(action=action), self.assertRaises(ValueError):
                controller.request(action, 0.)
        for time in (-.1, np.nan, np.inf):
            with self.subTest(time=time), self.assertRaises(ValueError):
                controller.state(time)
        self.assertEqual(controller.events, [])


class OutsideEntryScoringTests(unittest.TestCase):
    def setUp(self):
        self.scorer = EntryScorer({"a1": [0., 0., 0.]}, 1.)

    def update(self, start, end, attack=1, active=True, phase="thrust", t0=0., t1=1.):
        return self.scorer.update({"a1": state(attack, active, phase)},
                                  {"a1": np.asarray(start, dtype=float)},
                                  {"a1": np.asarray(end, dtype=float)}, t0, t1)

    def test_starting_inside_is_not_farmed_until_leaving_and_reentering(self):
        self.assertEqual(self.update([0., 0., 0.], [.5, 0., 0.]), [])
        self.assertEqual(self.update([.5, 0., 0.], [.4, 0., 0.]), [])
        self.assertEqual(self.update([.4, 0., 0.], [2., 0., 0.]), [])
        events = self.update([2., 0., 0.], [0., 0., 0.])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["attack_id"], 1)
        # A newly accepted attack while already inside cannot get a free point.
        self.assertEqual(self.update([0., 0., 0.], [.1, 0., 0.], attack=2), [])
        self.assertEqual(self.update([.1, 0., 0.], [2., 0., 0.], attack=2), [])
        self.assertEqual(len(self.update([2., 0., 0.], [0., 0., 0.], attack=2)), 1)

    def test_outside_to_outside_tunneling_crossing_is_counted(self):
        events = self.update([-2., 0., 0.], [2., 0., 0.], t0=10., t1=12.)
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0]["time_s"], 10.5)
        self.assertFalse(events[0]["double_touch"])

    def test_at_most_one_hit_per_accepted_attack(self):
        self.assertEqual(len(self.update([-2., 0., 0.], [2., 0., 0.])), 1)
        self.assertEqual(self.update([2., 0., 0.], [-2., 0., 0.]), [])
        self.assertEqual(self.update([-2., 0., 0.], [2., 0., 0.]), [])
        self.assertEqual(len(self.update([2., 0., 0.], [-2., 0., 0.], attack=2)), 1)
        self.assertEqual(len(self.scorer.events), 2)

    def test_inactive_entry_is_only_diagnostic_not_a_delayed_point(self):
        self.assertEqual(self.update([-2., 0., 0.], [0., 0., 0.], active=False, phase="prepare"), [])
        self.assertEqual(len(self.scorer.inactive_entries), 1)
        self.assertEqual(self.scorer.events, [])
        self.assertEqual(self.update([0., 0., 0.], [.2, 0., 0.], active=True), [])
        self.assertEqual(self.update([.2, 0., 0.], [2., 0., 0.]), [])
        self.assertEqual(len(self.update([2., 0., 0.], [0., 0., 0.])), 1)

    def test_same_step_both_agents_score_symmetric_double_touch(self):
        scorer = EntryScorer({"a1": [0., 0., 0.], "a2": [0., 0., 0.]}, 1.)
        events = scorer.update({"a1": state(), "a2": state()},
                               {"a1": [-2., 0., 0.], "a2": [3., 0., 0.]},
                               {"a1": [2., 0., 0.], "a2": [-2., 0., 0.]}, 0., .002)
        self.assertEqual(len(events), 2)
        self.assertEqual({e["agent"] for e in events}, {"a1", "a2"})
        self.assertTrue(all(e["double_touch"] for e in events))
        self.assertTrue(all("winner" not in e for e in events))
        self.assertEqual(len(scorer.events), 2)

    def test_one_active_one_inactive_entry_is_not_double_touch(self):
        scorer = EntryScorer({"a1": [0., 0., 0.], "a2": [0., 0., 0.]}, 1.)
        events = scorer.update({"a1": state(), "a2": state(active=False, phase="sweep")},
                               {"a1": [-2., 0., 0.], "a2": [2., 0., 0.]},
                               {"a1": [2., 0., 0.], "a2": [-2., 0., 0.]}, 0., .002)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["agent"], "a1")
        self.assertFalse(events[0]["double_touch"])
        self.assertEqual(len(scorer.inactive_entries), 1)


if __name__ == "__main__":
    unittest.main()
