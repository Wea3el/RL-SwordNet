"""Pure offline candidate tests, not a deployed A2 controller certificate."""
import unittest

import numpy as np

from tools.audit_a2_command_contract import bounded_command_step, demonstration, DEMO_BOUNDS_DEG, STATUS
from sim.joint_constraints import load_joint_constraints, model_joint_ranges_rad


class A2CommandContractTests(unittest.TestCase):
    def setUp(self):
        self.bounds = np.deg2rad(np.asarray(DEMO_BOUNDS_DEG))
        self.q = self.bounds.mean(axis=1)
        self.parameters = dict(control_period_s=.01,max_reference_speed_rad_s=2.,
                               tracking_leash_rad=.10,joint_margin_rad=np.deg2rad(1.))

    def step(self, q=None, reference=None, action=None, **parameters):
        return bounded_command_step(self.q if q is None else q,
            self.q if reference is None else reference,
            np.zeros(7) if action is None else action,self.bounds,**dict(self.parameters,**parameters))

    def test_one_second_stalled_target_does_not_accumulate_two_radians(self):
        result = demonstration()
        self.assertAlmostEqual(result['naive_target_delta_after_1s_rad'],2.,places=12)
        self.assertAlmostEqual(result['bounded_target_delta_after_1s_rad'],.1,places=12)
        self.assertEqual(len(result['history']),100)
        self.assertEqual(result['history'][-1]['decision_index'],49)

    def test_demonstration_uses_corrected_shared_cad_bounds(self):
        bounds=model_joint_ranges_rad(load_joint_constraints())
        np.testing.assert_allclose(self.bounds,bounds,atol=1e-15)
        np.testing.assert_array_equal(demonstration()['joint_bounds_cad_rad'],bounds)
        self.assertLess(self.bounds[3,0],np.deg2rad(-20.))
        self.assertLess(self.bounds[3,1],np.deg2rad(120.))

    def test_reverse_command_retreats_at_rate_limit(self):
        reference = self.q+.1
        r = self.step(reference=reference,action=-np.ones(7))
        np.testing.assert_allclose(r['accepted_q_ref_rad'],self.q+.08,atol=1e-14)

    def test_zero_action_holds_reference_exactly(self):
        reference = self.q+.3
        r = self.step(reference=reference)
        np.testing.assert_array_equal(r['accepted_q_ref_rad'],reference)

    def test_existing_excess_leash_does_not_snap_back(self):
        reference = self.q+.3
        r = self.step(reference=reference,action=np.ones(7))
        np.testing.assert_array_equal(r['accepted_q_ref_rad'],reference)
        self.assertTrue(r['governed_axes'].all())

    def test_existing_excess_leash_allows_only_inward_progress(self):
        reference = self.q-.3
        r = self.step(reference=reference,action=np.ones(7))
        np.testing.assert_allclose(r['accepted_delta_rad'],.02,atol=1e-14)
        np.testing.assert_allclose(r['accepted_q_ref_rad'],reference+.02,atol=1e-14)

    def test_two_control_ticks_per_decision_give_point_zero_four_unblocked(self):
        a=np.ones(7)
        first=self.step(action=a)
        second=self.step(reference=first['accepted_q_ref_rad'],action=a)
        np.testing.assert_allclose(second['accepted_q_ref_rad']-self.q,.04,atol=1e-14)

    def test_reference_rate_and_interior_for_many_mixed_states(self):
        rng=np.random.default_rng(121)
        low=self.bounds[:,0]+self.parameters['joint_margin_rad']
        high=self.bounds[:,1]-self.parameters['joint_margin_rad']
        for _ in range(300):
            q=rng.uniform(self.bounds[:,0],self.bounds[:,1])
            reference=rng.uniform(low,high)
            r=self.step(q=q,reference=reference,action=rng.uniform(-1,1,7))
            accepted=r['accepted_q_ref_rad']
            self.assertTrue(np.all(accepted>=low))
            self.assertTrue(np.all(accepted<=high))
            self.assertLessEqual(abs(accepted-reference).max(),.02+1e-14)
            self.assertTrue(np.all(r['admissible_low_rad']<=reference))
            self.assertTrue(np.all(r['admissible_high_rad']>=reference))

    def test_upper_joint_interior_stops_outward_command(self):
        high=self.bounds[:,1]-self.parameters['joint_margin_rad']
        r=self.step(q=high,reference=high,action=np.ones(7))
        np.testing.assert_array_equal(r['accepted_q_ref_rad'],high)

    def test_lower_joint_interior_stops_outward_command(self):
        low=self.bounds[:,0]+self.parameters['joint_margin_rad']
        r=self.step(q=low,reference=low,action=-np.ones(7))
        np.testing.assert_array_equal(r['accepted_q_ref_rad'],low)

    def test_joint_coordinate_reflection_is_symmetric(self):
        total=self.bounds.sum(axis=1)
        q=self.q+np.array([.1,-.2,.15,-.1,.2,-.15,.1])
        reference=q+np.array([.15,-.15,.05,-.05,.2,-.2,0])
        action=np.array([1,-1,.5,-.5,-1,1,.2])
        a=self.step(q=q,reference=reference,action=action)
        b=self.step(q=total-q,reference=total-reference,action=-action)
        np.testing.assert_allclose(b['accepted_q_ref_rad'],total-a['accepted_q_ref_rad'],atol=1e-14)

    def test_stalled_and_free_slots_keep_independent_state(self):
        a=np.zeros(7);a[0]=1
        stalled=self.q.copy();free=self.q.copy()
        for _ in range(100):
            stalled=self.step(reference=stalled,action=a)['accepted_q_ref_rad'].copy()
            free=self.step(q=free,reference=free,action=a)['accepted_q_ref_rad'].copy()
        self.assertAlmostEqual(stalled[0]-self.q[0],.1,places=12)
        self.assertAlmostEqual(free[0]-self.q[0],2.,places=12)
        self.assertFalse(np.shares_memory(stalled,free))

    def test_outside_physical_q_is_rejected_not_repaired(self):
        for joint in range(7):
            for side in (0,1):
                q=self.q.copy()
                q[joint]=self.bounds[joint,side]+(-1e-8 if side==0 else 1e-8)
                with self.subTest(joint=joint,side=side),self.assertRaisesRegex(ValueError,'outside physical'):
                    self.step(q=q)

    def test_exact_physical_edge_is_accepted_without_state_modification(self):
        q=self.bounds[:,0].copy()
        before=q.copy()
        self.step(q=q)
        np.testing.assert_array_equal(q,before)

    def test_reference_outside_configured_interior_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'configured interior'):
            self.step(reference=self.bounds[:,0])

    def test_rejects_malformed_nonfinite_or_unbounded_actions(self):
        for action in (np.zeros(6),np.ones(7)*1.01,np.ones(7)*-1.01,
                       np.ones(7)*np.nan,np.ones(7)*np.inf,[True]*7,['0']*7):
            with self.subTest(action=action),self.assertRaises(ValueError):
                self.step(action=action)

    def test_rejects_bad_parameters(self):
        for change in ({'control_period_s':0.},{'control_period_s':True},
                       {'control_period_s':np.nan},{'max_reference_speed_rad_s':0.},
                       {'tracking_leash_rad':-1.},{'joint_margin_rad':-1.},
                       {'joint_margin_rad':100.},{'max_reference_speed_rad_s':[1,2]},
                       {'tracking_leash_rad':True}):
            with self.subTest(change=change),self.assertRaises(ValueError):self.step(**change)

    def test_raw_action_and_accepted_command_are_not_conflated(self):
        r=self.step(reference=self.q+.1,action=np.ones(7))
        np.testing.assert_array_equal(r['raw_action'],np.ones(7))
        np.testing.assert_allclose(r['requested_delta_rad'],.02)
        np.testing.assert_array_equal(r['accepted_delta_rad'],np.zeros(7))
        self.assertTrue(r['governed_axes'].all())

    def test_inputs_and_outputs_do_not_alias_or_mutate(self):
        q=self.q.copy();reference=self.q.copy();action=np.ones(7)
        r=self.step(q=q,reference=reference,action=action)
        np.testing.assert_array_equal(q,self.q)
        np.testing.assert_array_equal(reference,self.q)
        np.testing.assert_array_equal(action,np.ones(7))
        for name in ('raw_action','accepted_q_ref_rad','accepted_delta_rad'):
            self.assertFalse(r[name].flags.writeable)
            for original in (q,reference,action):self.assertFalse(np.shares_memory(r[name],original))

    def test_pure_repeated_call_is_deterministic(self):
        a=self.step(action=np.ones(7));b=self.step(action=np.ones(7))
        np.testing.assert_array_equal(a['accepted_q_ref_rad'],b['accepted_q_ref_rad'])

    def test_reports_candidate_not_deployed(self):
        self.assertEqual(STATUS,'candidate_not_deployed')
        self.assertEqual(self.step()['status'],'candidate_not_deployed')


if __name__=='__main__':unittest.main()
