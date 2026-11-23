import tempfile
from pathlib import Path
import unittest
import numpy as np

from sim.feedback_policy import GraphAction, SkillGraphExecutor, WaitPolicy, TimedRequests
from sim.feedback_trial import physical_observation, run_feedback_trial
from sim.limited_duel import build_limited_duel
from sim.motion_spline import C2JointSpline
from sim.assembly import ROOT

Q = np.array([5.650832768709585,.6530728953767453,5.2537069343171225,
              1.4432567828190404,2.046114103620745,.2667993877991494,-1.1659888081140661])


class FeedbackTrialTests(unittest.TestCase):
    def test_physical_observation_has_no_opponent_intent_or_clock(self):
        obs = physical_observation(build_limited_duel(initial_qpos=Q),'a1')
        self.assertEqual(set(obs), {'own_q_rad','own_qvel_rad_s',
            'opponent_tip_minus_own_target_local_m','opponent_grip_minus_own_target_local_m',
            'opponent_tip_velocity_local_m_s'})
        self.assertTrue(all(np.isfinite(x).all() for x in obs.values()))

    def test_short_native_hold_is_complete_and_preserves_actual_states(self):
        ex = {s:SkillGraphExecutor({'ready':Q},{},'ready') for s in ('a1','a2')}
        policies = {s:WaitPolicy() for s in ex}
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'run'
            result=run_feedback_trial(ex,policies,path,duration_s=.22)
            self.assertTrue(result['native_execution_pass'])
            self.assertFalse(result['admitted_to_training'])
            self.assertTrue(result['evidence_complete'])
            self.assertIn('initial_policy_state',result)
            np.testing.assert_array_equal(result['graph']['a1']['nodes']['ready'],Q)
            self.assertEqual(result['all_target_entries'],[])
            self.assertEqual(result['qpos_resets_after_initialization'],0)
            self.assertEqual(result['official_match_result']['status'],'timeout_draw')
            self.assertEqual(result['completion']['termination_category'],'complete')
            with np.load(path/'trace.npz') as trace:
                self.assertEqual(len(trace['time_s']),221)
            with self.assertRaises(FileExistsError):
                run_feedback_trial(ex,policies,path,duration_s=.22)
            with self.assertRaises(ValueError):
                run_feedback_trial(ex,policies,Path(temp)/'reuse',duration_s=.22)

    def test_short_horizon_is_not_mislabeled_physical_failure(self):
        action=GraphAction('long_hold','ready','ready',C2JointSpline([0.,1.],[Q,Q]))
        ex={s:SkillGraphExecutor({'ready':Q},{'long_hold':action},'ready') for s in ('a1','a2')}
        policies={'a1':TimedRequests([(6,'long_hold')]),'a2':WaitPolicy()}
        with tempfile.TemporaryDirectory() as temp:
            r=run_feedback_trial(ex,policies,Path(temp)/'short',duration_s=.22)
            self.assertEqual(r['status'],'truncated')
            self.assertIsNone(r['failure'])
            self.assertFalse(r['native_execution_pass'])
            self.assertIsNone(r['official_match_result'])
            self.assertEqual(r['completion']['termination_category'],'horizon_active_reference')
            self.assertAlmostEqual(r['completion']['unfinished']['a1']['reference_end_s'],1.12)

    def test_future_script_request_prevents_false_completed_episode(self):
        action=GraphAction('later','ready','ready',C2JointSpline([0.,.1],[Q,Q]))
        ex={s:SkillGraphExecutor({'ready':Q},{'later':action},'ready') for s in ('a1','a2')}
        with tempfile.TemporaryDirectory() as temp:
            r=run_feedback_trial(ex,{'a1':TimedRequests([(30,'later')]),'a2':WaitPolicy()},
                Path(temp)/'pending',duration_s=.22)
            self.assertEqual(r['completion']['termination_category'],'horizon_unserved_script_requests')
            self.assertFalse(r['native_execution_pass'])
            self.assertIsNone(r['official_match_result'])

    def test_explicit_half_ms_profile_keeps_rules_and_full_window(self):
        ex={s:SkillGraphExecutor({'ready':Q},{},'ready') for s in ('a1','a2')}
        with tempfile.TemporaryDirectory() as temp:
            r=run_feedback_trial(ex,{s:WaitPolicy() for s in ex},Path(temp)/'half',duration_s=.12,
                runtime_profile=ROOT/'config/a1_runtime_rk4_half_ms_candidate.json')
            self.assertTrue(r['native_execution_pass'])
            self.assertEqual(r['physics_steps'],240)
            self.assertEqual(r['runtime_config']['physics_timestep_s'],.0005)
            self.assertEqual(r['runtime_config']['penetration_limit_m'],.005)
            self.assertEqual(r['runtime_config']['control_period_s'],.01)


if __name__ == '__main__': unittest.main()
