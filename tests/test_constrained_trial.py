import tempfile
from pathlib import Path
import unittest

import numpy as np

from sim.constrained_trial import (ScheduledReference, hold_reference,
    run_constrained_trial, validate_references)
from sim.joint_constraints import load_joint_constraints
from sim.motion_spline import C2JointSpline


READY = np.array([5.650832768709585, .6530728953767453, 5.2537069343171225,
                  1.4432567828190404, 2.046114103620745, .2667993877991494,
                  -1.1659888081140661])


class ConstrainedTrialTests(unittest.TestCase):
    def test_explicit_finite_schedule_and_observation_tail(self):
        curve = C2JointSpline([0., 1.], [READY, READY])
        with self.assertRaises(ValueError): ScheduledReference(curve, -1.)
        with self.assertRaises(ValueError): ScheduledReference(curve, True)
        with self.assertRaises(ValueError): ScheduledReference(curve, np.nan)
        p = ScheduledReference(curve)
        np.testing.assert_array_equal(p.target(0.), READY)
        with self.assertRaises(ValueError):
            validate_references({'a1': p}, load_joint_constraints(), 2.)
        with self.assertRaises(ValueError):
            validate_references({'a1': p, 'a2': p}, load_joint_constraints(), 1.8)

    def test_references_cannot_cross_confirmed_limits(self):
        bad = READY.copy(); bad[5] = 0.
        p = ScheduledReference(C2JointSpline([0., 1.], [READY, bad]))
        with self.assertRaises(ValueError):
            validate_references({'a1':p, 'a2':p}, load_joint_constraints(), 2.)

    def test_native_short_hold_retains_per_step_data_and_timeout(self):
        p = hold_reference(READY, start_s=.01, duration_s=.01)
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp)/'trial'
            result = run_constrained_trial({'a1':p, 'a2':p}, destination, duration_s=.12)
            self.assertEqual(result['status'], 'completed')
            self.assertTrue(result['native_execution_pass'])
            self.assertTrue(result['source_unchanged'])
            self.assertEqual(result['official_match_result']['status'], 'timeout_draw')
            self.assertEqual(result['all_target_entries'], [])
            self.assertEqual(result['qpos_resets_after_initialization'], 0)
            self.assertFalse(result['admitted_to_training'])
            with np.load(destination/'trace.npz') as trace:
                self.assertEqual(len(trace['time_s']), 121)
                self.assertIn('desired_reference_qpos', trace.files)
                np.testing.assert_allclose(trace['target_qpos'], np.tile(READY, (121, 2)))
            with self.assertRaises(FileExistsError):
                run_constrained_trial({'a1':p, 'a2':p}, destination, duration_s=.12)


if __name__ == '__main__': unittest.main()
