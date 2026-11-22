"""Diagnostic reaction-policy contracts, not physical defence certification."""
import copy
import unittest

import numpy as np

from sim.a1_reaction_validation import ObservedMotionParry, validate_reaction


def obs(time, tip=(0.,0.,0.), velocity=(0.,0.,0.), *, busy=False, enabled=True):
    return dict(time_s=time, opponent_tip_minus_own_target_local_m=list(tip),
        opponent_tip_velocity_local_m_s=list(velocity),
        own_skill=dict(busy=busy, action_mask=[True]+[enabled]*4))


class ReactionContracts(unittest.TestCase):
    def test_positive_finite_thresholds_required(self):
        for value in (0., -1., np.nan, np.inf, True, '1'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ObservedMotionParry(value)

    def test_absolute_time_or_hidden_opponent_metadata_cannot_trigger(self):
        policy = ObservedMotionParry()
        for time in (0., 10., 10000.):
            current = dict(obs(time), opponent_action=1, opponent_phase='attack', opponent_future_hit_s=0.)
            self.assertEqual(policy.decide(current), 0)
        self.assertEqual(policy.events, [])

    def test_threshold_and_motion_are_both_required(self):
        p = ObservedMotionParry()
        self.assertEqual(p.decide(obs(0.)), 0)
        self.assertEqual(p.decide(obs(.02, (.0009,0.,0.), (.1,0.,0.))), 0)
        self.assertEqual(p.decide(obs(.04, (.0011,0.,0.))), 0)
        self.assertEqual(p.decide(obs(.06, (.0011,0.,0.), (.1,0.,0.))), 4)
        self.assertEqual(p.decide(obs(.08, (.002,0.,0.), (.1,0.,0.))), 0)

    def test_busy_or_unavailable_own_action_blocks_trigger_without_spending_it(self):
        p = ObservedMotionParry(); p.decide(obs(0.))
        self.assertEqual(p.decide(obs(.02, (.002,0.,0.), (.1,0.,0.), busy=True)), 0)
        self.assertEqual(p.decide(obs(.04, (.002,0.,0.), (.1,0.,0.), enabled=False)), 0)
        self.assertEqual(p.decide(obs(.06, (.002,0.,0.), (.1,0.,0.))), 4)

    def test_rearm_requires_observed_quiet_return_and_continuous_dwell(self):
        p = ObservedMotionParry(); p.decide(obs(0.))
        p.decide(obs(.02, (.002,0.,0.), (.1,0.,0.)))
        for time in (.04, .06, .08):
            p.decide(obs(time))
        self.assertEqual(p.decide(obs(.10, (.002,0.,0.), (.1,0.,0.))), 0)
        for time in (.12, .14, .16, .18, .20, .22):
            p.decide(obs(time))
        self.assertEqual(p.decide(obs(.24, (.002,0.,0.), (.1,0.,0.))), 4)
        self.assertEqual(len(p.events), 2)

    def test_observation_gap_does_not_retroactively_certify_quiet_return(self):
        p = ObservedMotionParry(); p.decide(obs(0.))
        p.decide(obs(.02, (.002,0.,0.), (.1,0.,0.)))
        p.decide(obs(.04)); p.decide(obs(.50))
        self.assertEqual(p.decide(obs(.52, (.002,0.,0.), (.1,0.,0.))), 0)

    def test_time_translation_and_unused_metadata_preserve_choices(self):
        p, q = ObservedMotionParry(), ObservedMotionParry()
        inputs = [obs(0.),obs(.02),obs(.04,(.002,0.,0.),(.1,0.,0.)),obs(.06)]
        for current in inputs:
            shifted = copy.deepcopy(current); shifted['time_s'] += 100.
            shifted.update(opponent_action_id=999, opponent_elapsed_s=-1)
            self.assertEqual(p.decide(current), q.decide(shifted))

    def test_input_vectors_and_diagnostics_do_not_alias_policy_history(self):
        p = ObservedMotionParry(); initial = obs(0.); p.decide(initial)
        initial['opponent_tip_minus_own_target_local_m'][0] = 10.
        self.assertEqual(p.decide(obs(.02,(.002,0.,0.),(.1,0.,0.))), 4)
        output = p.diagnostics(); output['events'].clear()
        self.assertEqual(len(p.events), 1)
        self.assertFalse(p.diagnostics()['opponent_action_phase_clock_or_program_inputs'])

    def test_nonfinite_or_rewound_observation_cannot_be_motion_evidence(self):
        for current in (obs(-1.), obs(1.,(np.nan,0.,0.)), obs(1.,velocity=(np.inf,0.,0.))):
            with self.subTest(current=current), self.assertRaises(ValueError):
                ObservedMotionParry().decide(current)
        p = ObservedMotionParry(); p.decide(obs(1.))
        with self.assertRaises(ValueError):
            p.decide(obs(.5))

    def test_invalid_probe_parameters_fail_before_loading_any_bank(self):
        for changes in (dict(attacker='other'), dict(attack_start_s=np.nan),
                dict(repeat_count=True), dict(repeat_count=0), dict(physics_timestep_s=.003)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_reaction('missing', 'fresh', **changes)


if __name__ == '__main__':
    unittest.main()
