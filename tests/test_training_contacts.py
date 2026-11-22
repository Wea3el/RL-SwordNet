"""New training settlement must not rewrite old validation or invent rewards."""
import copy
import json
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np

from sim.foul_attribution import ContactEvidence, attribute_contact
from sim.skill_validation import V2_CONFIG
from sim.training_contacts import (TrainingContactMonitor, TrainingEnvironmentError,
    previous_material_point_velocity, settle_training_tick, _same_material_manifold,
    DEPLOYED_RULE_VERSION)


def verdict(tick=1, *, prior=False, a=.2, b=-.2):
    return attribute_contact(ContactEvidence(tick,"one","illegal_cross_robot_nonblade_contact",
        "a1","a2",(a,0.,0.),(b,0.,0.),(1.,0.,0.),True,prior))


def tiny_scene():
    model = mujoco.MjModel.from_xml_string('''<mujoco><option timestep=".001" gravity="0 0 0"/>
    <worldbody><body name="a1_body" pos="-.105 0 0"><joint type="slide" axis="1 0 0"/>
    <geom name="a1_arm" type="sphere" size=".1" mass="1"/></body>
    <body name="a2_body" pos=".105 0 0"><joint type="slide" axis="1 0 0"/>
    <geom name="a2_arm" type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    data.qvel[:] = [.2,-.2]
    mujoco.mj_forward(model,data)
    return SimpleNamespace(model=model,data=data)


class TrainingSettlementTests(unittest.TestCase):
    def test_same_tick_foul_precedes_hit(self):
        result = settle_training_tick(1, contacts=[verdict(a=.2,b=0.)], hit_agents=["a1"])
        self.assertEqual(result["status"],"foul_win")
        self.assertEqual(result["match_winner"],"a2")
        self.assertEqual(result["rl_reward"],{"a1":-1.,"a2":1.})
        self.assertTrue(result["foul_precedes_same_tick_hit"])
        self.assertEqual(result["rule_version"],DEPLOYED_RULE_VERSION)
        self.assertTrue(result["live_runner_changed"])
        self.assertFalse(result["candidate_only"])
        self.assertEqual(result["source_offline_result_metadata"]["rule_version"],
                         "motion_foul_offline_candidate_v1")

    def test_mutual_and_ambiguous_fouls_are_not_zero_sum_wins(self):
        for prior in (False,True,None):
            result=settle_training_tick(1, contacts=[verdict(prior=prior)])
            self.assertTrue(result["terminal"])
            self.assertIsNone(result["match_winner"])
            self.assertEqual(result["rl_reward"],{"a1":-1.,"a2":-1.})
            self.assertFalse(result["zero_sum_reward"])

    def test_errors_never_become_reward_or_draw(self):
        with self.assertRaises(TrainingEnvironmentError) as context:
            settle_training_tick(1, numerical_faults=["bad_state"], hit_agents=["a1"])
        self.assertIsNone(context.exception.details["rl_reward"])
        self.assertFalse(context.exception.details["valid_for_reward"])

    def test_real_native_preimpact_collection_does_not_mutate_live_state(self):
        scene=tiny_scene()
        monitor=TrainingContactMonitor(scene,dict(V2_CONFIG,physics_timestep_s=.001))
        total=[]
        for tick in range(1,101):
            monitor.capture_before_step(scene)
            mujoco.mj_step(scene.model,scene.data);mujoco.mj_forward(scene.model,scene.data)
            fields={k:getattr(scene.data,k).copy() for k in
                    ("qpos","qvel","qacc","ctrl","qfrc_applied","xfrc_applied")}
            report=monitor.inspect(scene,tick)
            for k,value in fields.items():np.testing.assert_array_equal(getattr(scene.data,k),value)
            total.extend(report["verdicts"])
            if total:break
        self.assertTrue(total)
        self.assertTrue(all(v["status"]=="double_foul" for v in total))
        first=monitor.summary()["first_contact_episodes"][0]
        self.assertAlmostEqual(first["evidence"][0]["velocity_a_world_m_s"][0],.2)
        self.assertAlmostEqual(first["evidence"][0]["velocity_b_world_m_s"][0],-.2)
        self.assertEqual(total[0]["physics_tick"],tick)

    def test_monitoring_and_no_monitoring_have_bit_identical_physics(self):
        tracked,control=tiny_scene(),tiny_scene()
        monitor=TrainingContactMonitor(tracked,dict(V2_CONFIG,physics_timestep_s=.001))
        for tick in range(1,31):
            monitor.capture_before_step(tracked)
            for scene in (tracked,control):
                mujoco.mj_step(scene.model,scene.data);mujoco.mj_forward(scene.model,scene.data)
            monitor.inspect(tracked,tick)
            for k in ("qpos","qvel","qacc","qfrc_constraint"):
                np.testing.assert_array_equal(getattr(tracked.data,k),getattr(control.data,k))

    def test_missing_repeated_or_skipped_frame_is_an_evidence_error(self):
        cfg=dict(V2_CONFIG,physics_timestep_s=.001)
        scene=tiny_scene();monitor=TrainingContactMonitor(scene,cfg)
        with self.assertRaises(TrainingEnvironmentError):monitor.inspect(scene,1)
        monitor.reset(scene);monitor.capture_before_step(scene)
        with self.assertRaises(TrainingEnvironmentError):monitor.capture_before_step(scene)
        mujoco.mj_step(scene.model,scene.data);mujoco.mj_forward(scene.model,scene.data)
        with self.assertRaises(TrainingEnvironmentError):monitor.inspect(scene,2)

    def test_initial_contact_cannot_fabricate_preimpact(self):
        scene=tiny_scene();scene.data.qpos[:]=[.01,-.01]
        mujoco.mj_forward(scene.model,scene.data)
        monitor=TrainingContactMonitor(scene,dict(V2_CONFIG,physics_timestep_s=.001))
        monitor.capture_before_step(scene)
        mujoco.mj_step(scene.model,scene.data);mujoco.mj_forward(scene.model,scene.data)
        with self.assertRaises(TrainingEnvironmentError):monitor.inspect(scene,1)

    def test_material_point_velocity_includes_rotation_and_old_frame(self):
        model=mujoco.MjModel.from_xml_string('''<mujoco><option gravity="0 0 0"/>
        <worldbody><body pos=".1 .2 .3"><joint type="hinge" axis="0 0 1"/>
        <geom type="sphere" size=".1" pos=".3 0 0" mass="1"/></body></worldbody></mujoco>''')
        before=mujoco.MjData(model);before.qpos[0]=.2;before.qvel[0]=1.5
        mujoco.mj_forward(model,before)
        after=copy.copy(before);after.qpos[0]+=.0015;mujoco.mj_forward(model,after)
        local=np.array([.3,.07,0.])
        point=after.xpos[1]+after.xmat[1].reshape(3,3)@local
        velocity=previous_material_point_velocity(model,before,after,1,point)
        offset=before.xmat[1].reshape(3,3)@local
        np.testing.assert_allclose(velocity,np.cross([0.,0.,1.5],offset),atol=1e-12)

    def test_material_manifold_matches_reordering_not_replacement_or_added_points(self):
        a,b=np.zeros(6),np.ones(6)
        self.assertTrue(_same_material_manifold([a,b],[b,a]))
        self.assertFalse(_same_material_manifold([a],[a,b]))
        self.assertFalse(_same_material_manifold([a],[b]))
        self.assertTrue(_same_material_manifold([a],[a+1e-12]))

    def test_error_evidence_remains_json_serializable(self):
        evidence=ContactEvidence(1,"id","illegal_cross_robot_nonblade_contact",
            "a1","a2",(np.float64(.2),0.,0.),(0.,0.,0.),(1.,0.,0.),True,False)
        error=TrainingEnvironmentError({"evidence":[evidence],"point":np.zeros(3),
                                        "bad":np.float64("nan")})
        encoded=json.dumps(error.details,allow_nan=False)
        self.assertIn('"contact_id": "id"',encoded)


if __name__=="__main__":unittest.main()
