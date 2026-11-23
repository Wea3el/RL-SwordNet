import json
import unittest

from sim.trial_completion import (
    ExecutorCompletionSnapshot as Executor,
    TrialCompletionSnapshot as Snapshot,
    assess_trial_completion,
)


class TrialCompletionTests(unittest.TestCase):
    def assess(self, *, now=8., horizon=8., stop='horizon', a1=None, a2=None,
               physical=(), evidence=(), execution=()):
        return assess_trial_completion(Snapshot(
            completed_time_s=now, requested_duration_s=horizon, stop_reason=stop,
            executors={'a1': a1 or Executor(settled=True),
                       'a2': a2 or Executor(settled=True)},
            physical_failures=physical, evidence_failures=evidence,
            execution_failures=execution))

    def test_full_measured_completion_can_allow_match_timeout_draw(self):
        result = self.assess()
        self.assertEqual(result['termination_category'], 'complete')
        self.assertTrue(result['complete'])
        self.assertFalse(result['is_truncation'])
        self.assertTrue(result['allows_timeout_draw_if_no_scored_result'])
        self.assertEqual(result['unfinished'], {})
        json.dumps(result, allow_nan=False)

    def test_short_horizon_keeps_active_motion_and_reference_deadline(self):
        result = self.assess(a1=Executor('strike_recover', 10., False))
        self.assertEqual(result['termination_category'], 'horizon_active_reference')
        self.assertTrue(result['is_truncation'])
        self.assertFalse(result['allows_timeout_draw_if_no_scored_result'])
        self.assertEqual(result['latest_active_reference_end_s'], 10.)
        self.assertEqual(result['unfinished']['a1']['reference_remaining_s'], 2.)
        self.assertFalse(result['reference_end_guarantees_actual_settling'])

    def test_reference_finished_does_not_mean_actual_recovery_settled(self):
        result = self.assess(a1=Executor('strike_recover', 7., False))
        self.assertEqual(result['termination_category'], 'horizon_recovery_unsettled')
        self.assertEqual(result['unfinished']['a1']['reference_remaining_s'], 0.)
        self.assertFalse(result['complete'])

    def test_no_active_action_can_still_have_unsettled_measured_tail(self):
        result = self.assess(a1=Executor(settled=False))
        self.assertEqual(result['termination_category'], 'horizon_recovery_unsettled')
        self.assertIsNone(result['latest_active_reference_end_s'])

    def test_future_unserved_script_is_not_a_complete_trial(self):
        result = self.assess(a1=Executor(settled=True,
            pending_script_requests=('attack', 'attack')))
        self.assertEqual(result['termination_category'], 'horizon_unserved_script_requests')
        self.assertEqual(result['unfinished']['a1']['pending_script_requests'], ['attack', 'attack'])

    def test_simultaneous_outstanding_work_all_preserved(self):
        result = self.assess(a1=Executor('attack', 10., False, ('attack',)),
                             a2=Executor('parry', 7., False))
        self.assertEqual(result['termination_category'], 'horizon_active_reference')
        self.assertEqual(result['unfinished']['a2']['phase'], 'recovery_unsettled')
        self.assertEqual(result['unfinished']['a1']['pending_script_requests'], ['attack'])

    def test_wall_budget_is_not_physical_failure_even_if_idle(self):
        result = self.assess(now=3., stop='wall_budget')
        self.assertEqual(result['termination_category'], 'wall_budget_exhausted')
        self.assertTrue(result['is_truncation'])
        self.assertFalse(result['complete'])
        self.assertFalse(result['allows_timeout_draw_if_no_scored_result'])

    def test_physical_fault_precedes_horizon_and_ongoing_actions(self):
        result = self.assess(now=3.203, horizon=12., stop='interrupted',
            a1=Executor('attack', 10., False), physical=('excessive_proxy_penetration',))
        self.assertEqual(result['termination_category'], 'physical_failure')
        self.assertFalse(result['is_truncation'])
        self.assertEqual(result['unfinished']['a1']['reference_end_s'], 10.)

    def test_physical_fault_precedes_wall_budget_and_retains_evidence_faults(self):
        result = self.assess(stop='wall_budget', physical=('joint_limit_violation',),
                             evidence=('evidence_write_failure',))
        self.assertEqual(result['termination_category'], 'physical_failure')
        self.assertEqual(result['evidence_failures'], ['evidence_write_failure'])

    def test_evidence_and_execution_errors_do_not_become_timeout_draw(self):
        for field, expected in [('evidence', 'evidence_failure'),
                                ('execution', 'execution_error')]:
            with self.subTest(field=field):
                result = self.assess(**{field: ('broken',)})
                self.assertEqual(result['termination_category'], expected)
                self.assertFalse(result['complete'])
                self.assertFalse(result['allows_timeout_draw_if_no_scored_result'])

    def test_external_interruption_remains_incomplete_even_if_settled(self):
        result = self.assess(now=3., stop='interrupted')
        self.assertEqual(result['termination_category'], 'interrupted')
        self.assertTrue(result['is_truncation'])

    def test_invalid_snapshot_rejected_instead_of_inventing_completion(self):
        cases = [dict(now=7.), dict(now=float('nan')), dict(horizon=0.),
                 dict(stop='unknown'), dict(a1=Executor('attack', None, False)),
                 dict(a1=Executor(None, 10., False)),
                 dict(a1=Executor('attack', 10., True)),
                 dict(a1=Executor(settled=True, pending_script_requests='attack'))]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.assess(**kwargs)


if __name__ == '__main__':
    unittest.main()
