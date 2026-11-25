import copy
import unittest

from sim.a1_reactive_audit import episode_checks
from sim.a1_triangle_reactive import PUBLIC_FIELDS


class ReactiveAuditTests(unittest.TestCase):
    def fixture(self):
        physical={k:[0.]*(7 if k.startswith('own_') else 3) for k in PUBLIC_FIELDS}
        obs=[]
        for when in (2.,12.):
            obs.append({'time_s':when,'physical':{'a1':physical,'a2':physical},
                'available':{'a1':[],'a2':['parry']},'requests':{'a1':None,'a2':'parry'}})
        b={'metadata':{'attacker_slot':'a1','repeat_count':2},'completed_time_s':20.,
           'executor_events':{'a1':[{'event':'started','time_s':1.},{'event':'started','time_s':11.}]}}
        d={'executor_events':{'a2':[
                {'event':'started','time_s':2.},{'event':'completed','time_s':5.},
                {'event':'started','time_s':12.},{'event':'completed','time_s':15.}]},
           'policy_events':{'a2':[
                {'event':'current_geometry_trigger','action':'parry','sample':1,'public_physical':physical},
                {'event':'observed_rearm'},
                {'event':'current_geometry_trigger','action':'parry','sample':2,'public_physical':physical}]}}
        c={'status':'qualified','baseline_triangle_audit':{'target_region':'upper','all_entries':[
            {'agent':'a1','region':'upper','time_s':3.}, {'agent':'a1','region':'upper','time_s':13.}]}}
        return c,b,d,obs

    def test_two_actual_threats_and_two_public_decisions_pass(self):
        self.assertEqual(episode_checks(*self.fixture(),4.)['status'],'passed')

    def test_second_script_without_second_threat_does_not_pass(self):
        c,b,d,o=self.fixture();c['baseline_triangle_audit']['all_entries'].pop()
        r=episode_checks(c,b,d,o,4.)
        self.assertEqual(r['status'],'failed')
        self.assertFalse(r['checks']['every_attack_has_a_real_target_threat'])

    def test_one_block_cannot_stand_in_for_repeated_defence(self):
        c,b,d,o=self.fixture();d['executor_events']['a2']=d['executor_events']['a2'][:2]
        self.assertEqual(episode_checks(c,b,d,o,4.)['status'],'failed')

    def test_future_fields_fail_common_contract(self):
        c,b,d,o=copy.deepcopy(self.fixture());o[0]['physical']['a2']['opponent_future_action']=2
        self.assertEqual(episode_checks(c,b,d,o,4.)['status'],'failed')

    def test_event_does_not_replace_actual_request_evidence(self):
        c,b,d,o=self.fixture();o[1]['requests']['a2']=None
        self.assertEqual(episode_checks(c,b,d,o,4.)['status'],'failed')

    def test_unavailable_action_is_not_valid_decision(self):
        c,b,d,o=self.fixture();o[0]['available']['a2']=[]
        self.assertEqual(episode_checks(c,b,d,o,4.)['status'],'failed')


if __name__=='__main__': unittest.main()
