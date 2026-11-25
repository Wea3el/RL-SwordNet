import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from sim.a1_compact_bank import ACTION_NAMES, CompactBankController, close_at_ready
from sim.assembly import ROOT
from sim.motion_spline import C2JointSpline


class CompactBankTests(unittest.TestCase):
    def test_bridge_keeps_old_interior_and_closes_at_ready(self):
        ready=np.ones(7);q=ready+.1
        source=C2JointSpline([0.,1.,2.],[q,q+.03,q])
        result,meta=close_at_ready(source,ready)
        np.testing.assert_array_equal(result.sample(0),ready)
        np.testing.assert_array_equal(result.sample(result.duration_s),ready)
        start,end=meta['source_interval_s']
        ts=np.linspace(0,2,41)
        np.testing.assert_allclose(result.sample(ts+start),source.sample(ts),rtol=0,atol=1e-13)
        self.assertAlmostEqual(end-start,2.)
        self.assertLess(result.continuity_report()['maximum_velocity_jump_rad_s'],1e-10)
        self.assertTrue(meta['new_native_validation_required'])

    def test_no_bridge_for_exact_closed_source(self):
        q=np.ones(7);source=C2JointSpline([0.,1.,2.],[q,q+.1,q])
        out,meta=close_at_ready(source,q)
        self.assertIs(out,source)
        self.assertFalse(meta['added_prefix']);self.assertFalse(meta['added_suffix'])

    def test_bad_bridge_parameters_rejected(self):
        source=C2JointSpline([0.,1.],[np.ones(7),np.ones(7)])
        for kwargs in ({'rest_s':0},{'reference_speed':-1},{'reference_speed':True},
                       {'rest_s':float('nan')}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                close_at_ready(source,np.ones(7),**kwargs)

    def fixture(self,path):
        cfg=json.loads((ROOT/'config/a1_shared_ready_glancing_seed.json').read_text())
        # Synthetic interface fixture, not a requalified historical motion.
        # The old seed's J3 now exceeds the confirmed straight-zero limit.
        cfg['nodes']={'ready':[3., .5, 3., 1., 2., .5, 0.]}
        cfg['action_ids']={str(i):n for i,n in enumerate(ACTION_NAMES)}
        cfg['actions']={n:{'id':i,'start_node':'ready','end_node':'ready','waypoint_nodes':['ready','ready'],
                           'knot_time_s':[0.,1.]} for i,n in enumerate(ACTION_NAMES[1:],1)}
        path.write_text(json.dumps(cfg));return cfg

    def test_stable_ids_busy_mask_wait_and_state_isolation(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'bank.json';self.fixture(path)
            a,b=CompactBankController(path),CompactBankController(path)
            self.assertEqual(a.mask().tolist(),[True]+[False]*6)
            self.assertFalse(a.actor_decision_allowed)
            for c in (a,b):
                c.executor.observe(0,c.executor.initial_q,np.zeros(7))
                c.executor.observe(.1,c.executor.initial_q,np.zeros(7))
            self.assertTrue(a.mask().all())
            self.assertTrue(a.actor_decision_allowed)
            a.request_id(1,.1)
            self.assertEqual(a.executor.active,'attack_left')
            self.assertIsNone(b.executor.active)
            a.request_id(0,.2)
            self.assertEqual(a.executor.active,'attack_left')
            self.assertFalse(a.actor_decision_allowed)
            self.assertEqual(a.own_status(.6)['own_active_action_id'],1)
            self.assertAlmostEqual(a.own_status(.6)['own_reference_progress'],.5)
            self.assertEqual(a.mask().tolist(),[True]+[False]*6)
            with self.assertRaises(ValueError):a.request_id(4,.2)
            for bad in (True,7,-1,1.):
                with self.subTest(bad=bad),self.assertRaises(ValueError):a.request_id(bad,.2)

    def test_unknown_action_map_and_training_admission_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'bank.json';cfg=self.fixture(path)
            cfg['action_ids']['1']='stab';path.write_text(json.dumps(cfg))
            with self.assertRaises(ValueError):CompactBankController(path)
            cfg=self.fixture(path);cfg['admitted_to_training']=True
            path.write_text(json.dumps(cfg))
            with self.assertRaises(ValueError):CompactBankController(path)

    def test_changed_bank_invalidates_controller_identity(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'bank.json';cfg=self.fixture(path)
            c=CompactBankController(path);c.assert_identity()
            cfg['notes']=['changed'];path.write_text(json.dumps(cfg))
            with self.assertRaises(ValueError):c.assert_identity()

    def test_compact_ids_and_closed_ready_routes_rechecked_on_load(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'bank.json'
            for bad_id in (True,2,None):
                cfg=self.fixture(path);cfg['actions']['attack_left']['id']=bad_id
                path.write_text(json.dumps(cfg))
                with self.subTest(bad_id=bad_id),self.assertRaises(ValueError):
                    CompactBankController(path)
            cfg=self.fixture(path)
            cfg['nodes']['other']=cfg['nodes']['ready']
            cfg['actions']['attack_left'].update(end_node='other',waypoint_nodes=['ready','other'])
            path.write_text(json.dumps(cfg))
            with self.assertRaises(ValueError):CompactBankController(path)


if __name__=='__main__':unittest.main()
