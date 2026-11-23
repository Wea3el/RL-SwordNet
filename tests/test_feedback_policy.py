import unittest
import numpy as np

from sim.feedback_policy import GraphAction, SkillGraphExecutor, ThreatParry, TimedRequests, RearmingThreatParry
from sim.motion_spline import C2JointSpline

Q = np.array([1., 1., 1., 1., 1., .5, 0.])


class FeedbackPolicyTests(unittest.TestCase):
    def executor(self):
        end = Q.copy(); end[4] += .2
        action = GraphAction('move', 'a', 'b', C2JointSpline([0., 1.], [Q, end]))
        return SkillGraphExecutor({'a': Q, 'b': end}, {'move': action}, 'a')

    def test_requires_exact_endpoint_and_bounded_path(self):
        bad = Q.copy(); bad[2] = -1.
        with self.assertRaises(ValueError):
            SkillGraphExecutor({'a': Q, 'b': bad}, {'move': GraphAction('move','a','b',C2JointSpline([0,1],[Q,bad]))},'a')
        end = Q.copy(); end[4] += .2
        with self.assertRaises(ValueError):
            SkillGraphExecutor({'a':Q,'b':end}, {'move':GraphAction('move','a','b',C2JointSpline([0,1],[Q,Q]))},'a')

    def test_stable_entry_end_and_nonteleporting_command(self):
        ex = self.executor()
        with self.assertRaises(ValueError): ex.request('move',0.)
        ex.observe(0.,Q,np.zeros(7)); ex.observe(.1,Q,np.zeros(7))
        self.assertEqual(ex.available(), ('move',))
        ex.request('move',.1)
        np.testing.assert_array_equal(ex.target(.1),Q)
        ex.observe(1.1,ex.nodes['b'],np.full(7,.11))
        self.assertIsNotNone(ex.active)
        ex.observe(1.2,ex.nodes['b'],np.zeros(7))
        ex.observe(1.3,ex.nodes['b'],np.zeros(7))
        self.assertIsNone(ex.active)
        self.assertEqual(ex.node,'b')
        np.testing.assert_array_equal(ex.target(2.),ex.nodes['b'])

    def test_displacement_and_approach_both_required_and_no_clock_read(self):
        p = ThreatParry()
        def obs(tip, vx):
            return {'opponent_tip_minus_own_target_local_m':np.array(tip),
                    'opponent_tip_velocity_local_m_s':np.array([vx,0,0])}
        self.assertIsNone(p.choose(obs([1,0,0],0),('parry',)))
        self.assertIsNone(p.choose(obs([1,.002,0],.01),('parry',)))
        self.assertIsNone(p.choose(obs([.998,0,0],-.01),('parry',)))
        self.assertEqual(p.choose(obs([.997,0,0],-.01),('parry',)),'parry')
        self.assertIsNone(p.choose(obs([.996,0,0],-.01),('parry',)))

    def test_busy_cannot_force_action(self):
        p = ThreatParry(consecutive_samples=1)
        p.choose({'opponent_tip_minus_own_target_local_m':[1,0,0],
                  'opponent_tip_velocity_local_m_s':[0,0,0]},())
        self.assertIsNone(p.choose({'opponent_tip_minus_own_target_local_m':[.9,0,0],
                  'opponent_tip_velocity_local_m_s':[-1,0,0]},()))
        self.assertFalse(p.fired)

    def test_any_motion_mode_is_explicit_not_an_attack_classifier(self):
        p=ThreatParry(require_approach=False,consecutive_samples=1)
        p.choose({'opponent_tip_minus_own_target_local_m':[1,0,0],
                  'opponent_tip_velocity_local_m_s':[0,0,0]},('parry',))
        self.assertEqual(p.choose({'opponent_tip_minus_own_target_local_m':[1.01,0,0],
                  'opponent_tip_velocity_local_m_s':[.01,0,0]},('parry',)),'parry')

    def test_timed_requests_are_explicit_test_script(self):
        p = TimedRequests([(1,'move')])
        self.assertIsNone(p.choose({},('move',)))
        self.assertIsNone(p.choose({},()))
        self.assertEqual(p.choose({},('move',)),'move')
        self.assertIsNone(p.choose({},('move',)))

    def test_rearm_requires_observed_recovery_and_own_readiness(self):
        p=RearmingThreatParry(consecutive_samples=1,recovery_samples=2)
        rest={'opponent_tip_minus_own_target_local_m':[1,0,0],
              'opponent_tip_velocity_local_m_s':[0,0,0]}
        incoming={'opponent_tip_minus_own_target_local_m':[.9,0,0],
                  'opponent_tip_velocity_local_m_s':[-.1,0,0]}
        self.assertIsNone(p.choose(rest,('parry',)))
        self.assertEqual(p.choose(incoming,('parry',)),'parry')
        p.choose(rest,());p.choose(rest,())
        self.assertTrue(p.fired)
        p.choose(rest,('parry',))
        self.assertFalse(p.fired)
        self.assertEqual(p.choose(incoming,('parry',)),'parry')


if __name__ == '__main__': unittest.main()
