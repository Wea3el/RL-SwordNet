import unittest
import json
import numpy as np

from sim.a1_reactive_counter import SettleThenRiposte, counter_executor
from sim.a1_triangle_reactive import PUBLIC_FIELDS


class ReactiveCounterTests(unittest.TestCase):
    def policy(self):
        return SettleThenRiposte({'tip':[.2,0,.1],'grip':[.6,0,-.5],'velocity':[-.2,0,.1]})

    def observation(self):
        return dict(zip(PUBLIC_FIELDS,[np.zeros(7),np.zeros(7),[.2,0,.1],[.6,0,-.5],[-.2,0,.1]]))

    def test_riposte_cannot_bypass_physical_entry_availability(self):
        p=self.policy();o=self.observation()
        self.assertIsNone(p.choose(o,('enter_parry',)))
        self.assertEqual(p.choose(o,('enter_parry',)),'enter_parry')
        self.assertIsNone(p.choose(o,()))
        self.assertEqual(p.choose(o,('riposte',)),'riposte')
        self.assertIsNone(p.choose(o,('riposte',)))

    def test_no_counter_before_observed_threat(self):
        p=self.policy()
        self.assertIsNone(p.choose(self.observation(),('riposte',)))
        self.assertFalse(p.countered)

    def test_initial_policy_state_is_serializable(self):
        json.dumps(self.policy().__dict__)

    def test_clock_fields_cannot_release_counter(self):
        p=self.policy();o=self.observation();p.entered=True
        o.update(time_s=999,opponent_future='attack_done')
        self.assertIsNone(p.choose(o,()))
        self.assertFalse(p.countered)

    def test_phase_time_origin_is_exact_despite_roundoff(self):
        ready=np.array([1.,1.,1.,1.,1.,.6,0.])
        guard=ready+.02
        skill={'counter_departure_relative_s':1.2,'counter':{
            'time_s':[0.,.3,1.1999999999999997,2.5],
            'qpos':[ready,guard,guard,ready]}}
        ex=counter_executor(skill,ready)
        self.assertEqual(ex.actions['riposte'].curve.knot_time_s[0],0.)


if __name__=='__main__':unittest.main()
