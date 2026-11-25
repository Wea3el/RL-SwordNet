"""Saved-evidence gate regressions, not simulated attack/defence examples.

Reference/trace I/O uses tiny fixtures; geometric event audits are mocked so
these tests isolate matching, recovery and failure propagation only.
"""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from sim.manual_coordinated_trial import review
from sim.manual_guard_reference import DEFAULT_REFERENCE, choose_pose, load_reference
from sim.joint_constraints import load_joint_constraints
from sim.motion_spline import C2JointSpline


class ManualRecoveryContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        constraints = load_joint_constraints()
        manual, display, _ = load_reference(DEFAULT_REFERENCE, constraints)
        self.ready = choose_pose(display, constraints, manual['facing_variant']['base_yaw_delta_rad'])
        self.ready[0] -= .1  # Explicit new ready, not the screenshot's exact reset.
        self.attack = C2JointSpline([0., 4., 11.], [self.ready, self.ready, self.ready])
        self.hold = C2JointSpline([0., 13.25], [self.ready, self.ready])
        middle = self.ready.copy()
        middle[0] += .1
        self.motion = C2JointSpline([0., 2., 7., 12.], [self.ready, middle, middle, self.ready])
        self.summaries, self.plans = {}, {}
        self.audits = {
            'baseline': {'all_entries': [{'agent': 'a1', 'time_s': 4., 'region': 'upper'}],
                         'intended_region_first': True, 'initial_overlap': {}, 'sweep_failures': {}, 'attacker_slot': 'a1'},
            'defended': {'all_entries': [], 'initial_overlap': {}, 'sweep_failures': {}, 'attacker_slot': 'a1'},
        }
        for name, curve, start in (('baseline', self.hold, .75), ('defended', self.motion, 1.5)):
            folder = self.root/name/'native'
            folder.mkdir(parents=True)
            self.plans[name] = {'source_attack_sha256': 'a'*64, 'manual_reference_sha256': 'b'*64,
                'initial_guard_cad_rad': self.ready.tolist(), 'attacker_slot': 'a1', 'defender_slot': 'a2',
                'attacker_action': 'cut_down_center', 'target_region': 'upper',
                'requested_duration_s': 16., 'scoring': {'fixture_only': True},
                'recipe': {'candidate_id': name, 'start_s': start}}
            self.summaries[name] = {'runtime_config': {'physics_timestep_s': .1,
                'minimum_blade_force_n': .1, 'minimum_blade_impulse_ns': .02},
                'joint_constraints': {'fixture': True}, 'source_sha256': {'fixture': 'same'},
                'requested_duration_s': 16., 'model_source_sha256': {'fixture': 'same'},
                'program_start_s': {'a1': .75, 'a2': start},
                'metadata': {'attacker_slot': 'a1'}, 'native_execution_pass': True,
                'source_unchanged': True, 'failure': None,
                'entry_ready': {'a1': True, 'a2': True},
                'final_settled': {'a1': True, 'a2': True}, 'qpos_resets_after_initialization': 0}
            self.write(name)
            self.write_references(name, curve)
            times = np.arange(161)*.1
            np.savez(folder/'trace.npz', time_s=times,
                     qpos=np.tile(np.r_[self.ready, self.ready], (len(times), 1)),
                     qvel=np.zeros((len(times), 14)))
            (folder/'contacts.json').write_text(json.dumps([{
                'interval_start_s': 2., 'interval_end_s': 2.1, 'contacts': [{
                    'category': 'cross_blade_blade', 'geom1': 'a1_shinai_blade_collision_0',
                    'geom2': 'a2_shinai_blade_collision_0', 'normal_force_n': 1.}]}]))

    def write(self, name):
        (self.root/name/'plan.json').write_text(json.dumps(self.plans[name]))
        (self.root/name/'native/summary.json').write_text(json.dumps(self.summaries[name]))

    def write_references(self, name, defender_curve, attack=None):
        curves = {'a1': self.attack if attack is None else attack, 'a2': defender_curve}
        arrays = {slot+'_'+key: getattr(curve, key) for slot, curve in curves.items()
                  for key in ('knot_time_s', 'knot_qpos', 'coefficients')}
        np.savez(self.root/name/'native/references.npz', **arrays)

    def run_review(self):
        def audit(folder, *args):
            return copy.deepcopy(self.audits[Path(folder).parent.name])
        with patch('sim.a1_triangle_parry.audit_recorded_triangle', side_effect=audit), \
                patch('mujoco.mj_step', side_effect=AssertionError('No native physics')):
            return review(self.root)

    def test_changed_defender_timing_can_share_an_identical_ready_and_attack_baseline(self):
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in self.root.rglob('*') if p.is_file()}
        result = self.run_review()
        self.assertEqual(result['status'], 'qualified')
        self.assertTrue(all(result['matching_checks'].values()))
        self.assertFalse(result['admitted_to_training'])
        for path, digest in before.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)

    def test_old_ready_baseline_is_rejected_before_physical_comparison(self):
        self.plans['baseline']['initial_guard_cad_rad'][0] += .1
        self.write('baseline')
        with self.assertRaisesRegex(ValueError, 'baseline threat, reset'):
            self.run_review()

    def test_static_guard_that_already_prevents_the_baseline_hit_is_not_credited(self):
        self.audits['baseline']['all_entries'] = []
        result = self.run_review()
        self.assertFalse(result['successful_physical_defence'])
        self.assertFalse(result['baseline_valid_threat'])

    def test_late_arm_foul_invalidates_earlier_successful_contact(self):
        self.summaries['defended'].update(native_execution_pass=False,
            failure=['foul_ends_physical_diagnostic'])
        self.write('defended')
        result = self.run_review()
        self.assertEqual(result['status'], 'invalid')
        self.assertFalse(result['successful_physical_defence'])

    def test_contact_then_recovery_hit_in_any_region_is_still_failure(self):
        self.audits['defended']['all_entries'] = [{'agent': 'a1', 'time_s': 14., 'region': 'right'}]
        result = self.run_review()
        self.assertEqual(result['status'], 'failed')
        self.assertFalse(result['successful_physical_defence'])

    def test_truncated_trace_or_unsettled_exit_is_not_a_success(self):
        folder = self.root/'defended/native'
        np.savez(folder/'trace.npz', time_s=[0., .1],
                 qpos=np.tile(np.r_[self.ready, self.ready], (2, 1)), qvel=np.zeros((2, 14)))
        self.summaries['defended']['native_execution_pass'] = False
        self.summaries['defended']['final_settled']['a2'] = False
        self.write('defended')
        result = self.run_review()
        self.assertFalse(result['successful_physical_defence'])
        self.assertFalse(result['matching_checks']['defended_complete_native_trace'])
        self.assertFalse(result['matching_checks']['both_final_states_settled'])

    def test_late_baseline_hit_after_attack_end_is_not_a_valid_pair(self):
        self.audits['baseline']['all_entries'][0]['time_s'] = 12.5
        result = self.run_review()
        self.assertEqual(result['status'], 'invalid')
        self.assertFalse(result['successful_physical_defence'])
        self.assertFalse(result['matching_checks']['baseline_hit_within_attack_window'])

    def test_changed_attack_clock_or_nonconstant_baseline_cannot_qualify(self):
        self.summaries['defended']['program_start_s']['a1'] = 1.
        self.write('defended')
        q = self.ready.copy()
        q[0] += .1
        self.write_references('baseline', C2JointSpline([0., 13.25], [self.ready, q]))
        result = self.run_review()
        self.assertFalse(result['successful_physical_defence'])
        self.assertFalse(result['matching_checks']['attack_start_s'])
        self.assertFalse(result['matching_checks']['baseline_defender_entire_reference_is_original_guard'])


if __name__ == '__main__':
    unittest.main()
