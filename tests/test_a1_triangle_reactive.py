import copy
import unittest

import numpy as np

from sim.a1_triangle_reactive import CorridorPolicy, PUBLIC_FIELDS


class CorridorPolicyTests(unittest.TestCase):
    def setUp(self):
        self.corridors=[{'action':'upper','tip':[.2,0.,.1],
                         'grip':[.6,.2,-.5],'velocity':[-.2,0.,.1]}]

    def obs(self,tip=None,velocity=None,grip=None):
        return dict(zip(PUBLIC_FIELDS,[np.zeros(7),np.zeros(7),
            np.array([1.,0.,0.] if tip is None else tip),
            np.array([1.,0.,-.7] if grip is None else grip),
            np.array([0.,0.,0.] if velocity is None else velocity)]))

    def cue(self):
        return self.obs([.2,0.,.1],[-.2,0.,.1],[.6,.2,-.5])

    def test_requires_confirmation_and_available_entry(self):
        p=CorridorPolicy(self.corridors)
        self.assertIsNone(p.choose(self.obs(),('upper',)))
        self.assertIsNone(p.choose(self.cue(),()))
        self.assertIsNone(p.choose(self.cue(),('upper',)))
        self.assertEqual(p.choose(self.cue(),('upper',)),'upper')
        self.assertIsNone(p.choose(self.cue(),('upper',)))

    def test_no_stationary_or_outgoing_false_trigger(self):
        for velocity in ([0.,0.,0.],[.2,0.,-.1]):
            p=CorridorPolicy(self.corridors,confirmation_samples=1)
            self.assertIsNone(p.choose(self.obs([.2,0.,.1],velocity,[.6,.2,-.5]),('upper',)))

    def test_no_time_skill_label_or_future_read(self):
        a=CorridorPolicy(self.corridors,confirmation_samples=1)
        b=CorridorPolicy(self.corridors,confirmation_samples=1)
        observation=self.cue()
        extra=copy.deepcopy(observation)
        extra.update(time_s=-123,opponent_action='wrong',future_qpos='malicious')
        self.assertEqual(a.choose(observation,('upper',)),b.choose(extra,('upper',)))

    def test_delay_requires_persistent_cue_not_stale_future_action(self):
        p=CorridorPolicy(self.corridors,confirmation_samples=1,reaction_delay_steps=2)
        self.assertIsNone(p.choose(self.cue(),('upper',)))
        self.assertIsNone(p.choose(self.obs(),('upper',)))
        self.assertIsNone(p.choose(self.cue(),('upper',)))
        self.assertIsNone(p.choose(self.cue(),('upper',)))
        self.assertEqual(p.choose(self.cue(),('upper',)),'upper')

    def test_rearm_needs_observed_rest_and_own_readiness(self):
        p=CorridorPolicy(self.corridors,confirmation_samples=1)
        p.choose(self.obs(),('upper',))
        p.choose(self.cue(),('upper',))
        for _ in range(10): p.choose(self.obs(),())
        self.assertTrue(p.fired)
        for _ in range(5): p.choose(self.obs(),('upper',))
        self.assertFalse(p.fired)
        self.assertEqual(p.choose(self.cue(),('upper',)),'upper')

    def test_different_corridors_select_from_public_geometry(self):
        other={'action':'left','tip':[.2,.3,0.], 'grip':[.6,.2,-.5],'velocity':[-.2,0.,0.]}
        p=CorridorPolicy(self.corridors+[other],confirmation_samples=1)
        self.assertEqual(p.choose(self.obs(other['tip'],other['velocity'],other['grip']),('upper','left')),'left')

    def test_bad_public_state_rejected(self):
        p=CorridorPolicy(self.corridors)
        obs=self.obs();obs[PUBLIC_FIELDS[2]][0]=np.nan
        with self.assertRaises(ValueError): p.choose(obs,('upper',))


if __name__=='__main__': unittest.main()
