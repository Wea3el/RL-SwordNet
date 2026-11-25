"""Evidence binding tests; fixtures do not run MuJoCo dynamics."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from sim import a1_reactive_library as lib
from sim.motion_spline import C2JointSpline


class ReactiveLibraryEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        (self.root/'config').mkdir()
        self.scoring={'test_scoring_version':1}
        self.write(self.root/'config/scoring_triangle_v1_candidate.json',self.scoring)
        # Fabricated evidence-binding fixture only; not a robot demonstration.
        # Keep it within current J3 limits without altering archived evidence.
        self.ready=np.array([3.2859,.4101,3.5544,1.,2.0118,.4118,.7799])
        middle=self.ready.copy();middle[0]+=.02
        self.curve=C2JointSpline([0.,.3,1.2,2.],[self.ready,middle,middle,self.ready])
        # Interleave physical slots to ensure extraction uses model addresses,
        # never an assumed contiguous [a1(7),a2(7)] memory layout.
        bodies=[]
        for i in range(7):
            for slot in ('a2','a1'):
                bodies.append(f'<body><joint name="{slot}_J{i}" type="hinge"/>'
                              '<geom type="sphere" size=".01" mass="1"/></body>')
        self.xml='<mujoco><worldbody>'+''.join(bodies)+'</worldbody></mujoco>'
        self.model=mujoco.MjModel.from_xml_string(self.xml)
        self.folders={n:self.root/n for n in ('baseline','defended','counter')}
        for name,folder in self.folders.items():
            self.make_native(folder,hold_defender=name=='baseline')
        self.hit={'agent':'a2','region':'left','time_s':2.2}
        self.gate=self.make_gate(counter=False)
        self.counter_gate=self.make_gate(counter=True)
        self.write(self.root/'defence_gate.json',self.gate)
        self.write(self.root/'counter_gate.json',self.counter_gate)
        self.spec={'defences':[{'region':'upper','baseline':'baseline',
            'defended':'defended','comparison':'defence_gate.json','calibrate_s':1.,
            'counter':{'native':'counter','comparison':'counter_gate.json',
                       'departure_relative_s':1.2}}]}
        self.write(self.root/'spec.json',self.spec)
        fields={lib.PUBLIC_FIELDS[0]:self.ready.tolist(),lib.PUBLIC_FIELDS[1]:[0.]*7,
                lib.PUBLIC_FIELDS[2]:[1.,0.,0.],lib.PUBLIC_FIELDS[3]:[.5,0.,0.],
                lib.PUBLIC_FIELDS[4]:[-.1,0.,0.]}
        self.patcher=patch.multiple(lib,ROOT=self.root,
                                  corridor_observation=lambda *a,**kw:{'physical':fields})
        self.patcher.start();self.addCleanup(self.patcher.stop)

    @staticmethod
    def write(path,value):
        path.write_text(json.dumps(value))

    @staticmethod
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def make_native(self,folder,hold_defender=False):
        folder.mkdir();(folder/'scene.xml').write_text(self.xml)
        starts={'a1':.5,'a2':.7}
        summary={'status':'completed','native_execution_pass':True,'failure':None,
            'source_unchanged':True,'metadata':{'attacker_slot':'a1'},
            'program_start_s':starts,'requested_duration_s':4.,
            'runtime_config':{'physics_timestep_s':.1}}
        self.write(folder/'summary.json',summary)
        times=np.arange(41)*.1;desired=np.zeros((len(times),self.model.nq));refs={}
        for slot in ('a1','a2'):
            curve=(C2JointSpline([0.,2.],[self.ready,self.ready])
                   if slot=='a2' and hold_defender else self.curve)
            for field in ('knot_time_s','knot_qpos','coefficients'):
                refs[slot+'_'+field]=getattr(curve,field)
            ids=[self.model.joint(slot+'_J'+str(i)).id for i in range(7)]
            desired[:,self.model.jnt_qposadr[ids]]=curve.sample(times-starts[slot])
        np.savez(folder/'references.npz',**refs)
        np.savez(folder/'trace.npz',time_s=times,desired_reference_qpos=desired,qpos=desired)
        self.write(folder/'contacts.json',[{'fixture_contact':1}])

    def audit(self,folder,entries):
        return {'attacker_slot':'a1','scoring_snapshot':copy.deepcopy(self.scoring),
                'initial_overlap':{},'sweep_failures':{},'all_entries':entries,
                'native_trace_source_sha256':{n:self.digest(folder/n)
                    for n in ('summary.json','trace.npz','scene.xml')}}

    def make_gate(self,counter):
        folder=self.folders['counter' if counter else 'defended']
        hits=[copy.deepcopy(self.hit)] if counter else []
        result={'status':'defence_with_counter_candidate' if counter else 'qualified',
                'attacker_slot':'a1','matching_checks':{'identical':True},
                'baseline_triangle_audit':self.audit(self.folders['baseline'],[]),
                'defended_triangle_audit':self.audit(folder,hits),
                'attacker_entries_entire_window':[], 'defender_entries_entire_window':hits,
                'native_contact_source_sha256':self.digest(folder/'contacts.json')}
        if counter:
            result.update(counterattack_feasibility_pass=True,
                          first_pre_hit_weapon_contact_s=1.1,counter_release_s=1.9)
        return result

    def build(self):
        return lib.build(self.root/'spec.json',self.root/'built.json',require_three_regions=False)

    def mutate_npz(self,path,fn):
        with np.load(path) as source: values={k:source[k].copy() for k in source.files}
        fn(values);np.savez(path,**values)

    def test_valid_bound_counter_remains_candidate_only(self):
        result=self.build()
        self.assertFalse(result['admitted_to_training'])
        self.assertTrue(result['candidate_only'])
        self.assertEqual(result['skills'][0]['counter_departure_relative_s'],1.2)
        self.assertIn(str(self.folders['counter']/'trace.npz'),result['provenance_sha256'])

    def test_changed_counter_trace_is_rejected(self):
        self.mutate_npz(self.folders['counter']/'trace.npz',
                        lambda v:v['desired_reference_qpos'].__setitem__((5,0),9.))
        with self.assertRaisesRegex(ValueError,'Native evidence changed'):self.build()

    def test_changed_counter_contacts_are_rejected(self):
        self.write(self.folders['counter']/'contacts.json',[])
        with self.assertRaisesRegex(ValueError,'contact evidence'):self.build()

    def test_noncomplete_counter_is_rejected(self):
        p=self.folders['counter']/'summary.json';s=json.loads(p.read_text())
        s['native_execution_pass']=False;self.write(p,s)
        with self.assertRaisesRegex(ValueError,'Incomplete or invalid'):self.build()

    def test_wrong_counter_role_is_rejected(self):
        p=self.folders['counter']/'summary.json';s=json.loads(p.read_text())
        s['metadata']['attacker_slot']='a2';self.write(p,s)
        with self.assertRaisesRegex(ValueError,'attacker identity'):self.build()

    def test_scoring_snapshot_difference_is_rejected(self):
        self.counter_gate['baseline_triangle_audit']['scoring_snapshot']={'test_scoring_version':2}
        self.write(self.root/'counter_gate.json',self.counter_gate)
        with self.assertRaisesRegex(ValueError,'scoring snapshots'):self.build()

    def test_hit_before_departure_is_rejected(self):
        hits=[dict(self.hit,time_s=1.8)]
        self.counter_gate['defender_entries_entire_window']=hits
        self.counter_gate['defended_triangle_audit']['all_entries']=hits
        self.write(self.root/'counter_gate.json',self.counter_gate)
        with self.assertRaisesRegex(ValueError,'after departure'):self.build()

    def test_counter_gate_with_failed_matching_is_rejected(self):
        self.counter_gate['matching_checks']['identical']=False
        self.write(self.root/'counter_gate.json',self.counter_gate)
        with self.assertRaisesRegex(ValueError,'Mismatched source'):self.build()

    def test_modified_coefficients_are_rejected(self):
        self.mutate_npz(self.folders['counter']/'references.npz',
                        lambda v:v['a2_coefficients'].__setitem__((0,0,0),9.))
        with self.assertRaisesRegex(ValueError,'coefficients'):self.build()

    def test_rebuilt_but_changed_reference_is_rejected_by_native_trace(self):
        def change(values):
            q=values['a2_knot_qpos'];q[1:3,0]+=.001
            curve=C2JointSpline(values['a2_knot_time_s'],q)
            values['a2_coefficients']=curve.coefficients
        self.mutate_npz(self.folders['counter']/'references.npz',change)
        with self.assertRaisesRegex(ValueError,'hash-locked native trace'):self.build()

    def test_ordinary_defence_reference_is_also_bound(self):
        def change(values):
            q=values['a2_knot_qpos'];q[1:3,0]+=.001
            values['a2_coefficients']=C2JointSpline(values['a2_knot_time_s'],q).coefficients
        self.mutate_npz(self.folders['defended']/'references.npz',change)
        with self.assertRaisesRegex(ValueError,'hash-locked native trace'):self.build()

    def test_missing_required_trace_hash_is_rejected(self):
        del self.counter_gate['defended_triangle_audit']['native_trace_source_sha256']['trace.npz']
        self.write(self.root/'counter_gate.json',self.counter_gate)
        with self.assertRaisesRegex(ValueError,'all required native'):self.build()


if __name__=='__main__':unittest.main()
