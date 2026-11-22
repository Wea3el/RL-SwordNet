"""Decision compression, physical discount and invalid-sample contracts."""
import unittest
from dataclasses import replace
import numpy as np

from sim.a1_rollout import DecisionEpisode, physical_time_gae


READY=np.ones(5,dtype=bool)


def start(buffer,slot="a1",action=1,value=0.,time=0.):
    buffer.start(slot,[1.,2.],READY,action,-1.,value,
                 policy_id=buffer.policy_ids[slot],time_s=time)


def advance(buffer,time,*,reward=0.,ready=False,action=0,end=False):
    buffer.advance(time_s=time,rewards={"a1":reward},next_observations={"a1":[3.,4.]},
        ready_by_slot={"a1":ready},effective_actions={"a1":action},accepted={"a1":True},episode_end=end)


class DecisionRolloutTests(unittest.TestCase):
    def test_busy_steps_do_not_become_actor_decisions(self):
        b=DecisionEpisode({"a1":"snapshot"},gamma_per_second=.9);start(b)
        advance(b,.02,action=1)
        for t in (.04,.06,.08):advance(b,t)
        advance(b,.1,reward=1.,end=True)
        rows=b.commit()["a1"]
        self.assertEqual(len(rows),1)
        self.assertAlmostEqual(rows[0].discounted_reward,.9**.1)
        self.assertAlmostEqual(rows[0].duration_s,.1)

    def test_ready_guard_zero_is_a_real_wait_choice(self):
        b=DecisionEpisode({"a1":"snapshot"});start(b,action=0)
        advance(b,.02,ready=True)
        start(b,action=1,time=.02);advance(b,.04,action=1,end=True)
        rows=b.commit()["a1"]
        self.assertEqual([r.action for r in rows],[0,1])

    def test_async_two_slots_close_independently_and_rewards_are_not_negated(self):
        b=DecisionEpisode({"a1":"learner","a2":"learner"})
        for s in b.policy_ids:start(b,s,action=0 if s=="a1" else 1)
        b.advance(time_s=.02,rewards={"a1":0.,"a2":0.},
            next_observations={s:[3.,4.] for s in b.policy_ids},ready_by_slot={"a1":True,"a2":False},
            effective_actions={"a1":0,"a2":1},accepted={s:True for s in b.policy_ids})
        start(b,"a1",action=0,time=.02)
        b.advance(time_s=.04,rewards={s:-1. for s in b.policy_ids},
            next_observations={s:[3.,4.] for s in b.policy_ids},ready_by_slot={s:False for s in b.policy_ids},
            effective_actions={s:0 for s in b.policy_ids},accepted={s:True for s in b.policy_ids},episode_end=True)
        rows=b.commit();self.assertEqual(len(rows["a1"]),2);self.assertEqual(len(rows["a2"]),1)
        self.assertLess(rows["a1"][-1].discounted_reward,0)
        self.assertLess(rows["a2"][-1].discounted_reward,0)

    def test_invalid_round_never_commits_partial_good_samples(self):
        b=DecisionEpisode({"a1":"s"});start(b,action=0);advance(b,.02,ready=True)
        b.abort("unusable_first_contact_manifold")
        with self.assertRaises(RuntimeError):b.commit()
        with self.assertRaises(RuntimeError):start(b,time=.02)

    def test_unfinished_round_is_not_a_timeout_draw(self):
        b=DecisionEpisode({"a1":"s"});start(b);advance(b,.02,action=1)
        with self.assertRaises(RuntimeError):b.commit()

    def test_actual_action_or_snapshot_mismatch_rejected_without_partial_mutation(self):
        b=DecisionEpisode({"a1":"s"});start(b)
        with self.assertRaises(ValueError):advance(b,.02,action=0)
        self.assertEqual(b.time_s,0.)
        advance(b,.02,action=1,end=True);self.assertEqual(len(b.commit()["a1"]),1)
        c=DecisionEpisode({"a1":"s"})
        with self.assertRaises(ValueError):c.start("a1",[0.],READY,1,-1.,0.,policy_id="old",time_s=0.)

    def test_forced_busy_mask_cannot_create_an_actor_record(self):
        b=DecisionEpisode({"a1":"s"})
        with self.assertRaises(ValueError):b.start("a1",[0.],np.array([True,False,False,False,False]),
            0,0.,0.,policy_id="s",time_s=0.)

    def test_physical_time_gae_uses_long_duration_not_one_default_tick(self):
        b=DecisionEpisode({"a1":"s"},gamma_per_second=.9)
        start(b,action=1,value=.2);advance(b,30.,ready=True,action=1)
        start(b,action=0,value=.3,time=30.);advance(b,30.02,reward=1.,end=True)
        rows=b.commit()["a1"];adv,ret=physical_time_gae(rows,lambda_per_second=.8)
        last=.9**.02-.3
        first=.9**30*.3-.2+.9**30*.8**30*last
        np.testing.assert_allclose(adv,[first,last])
        np.testing.assert_allclose(ret,adv+[.2,.3])

    def test_timeout_game_end_bootstraps_zero(self):
        b=DecisionEpisode({"a1":"s"});start(b,value=7.)
        advance(b,1.,action=1,end=True)
        adv,ret=physical_time_gae(b.commit()["a1"])
        np.testing.assert_allclose(adv,[-7.]);np.testing.assert_allclose(ret,[0.])

    def test_physical_reward_sum_is_independent_of_zero_reward_subdivision(self):
        results=[]
        for split in (False,True):
            b=DecisionEpisode({"a1":"s"},gamma_per_second=.8);start(b)
            if split:
                advance(b,.1,action=1);advance(b,.2)
                advance(b,.3,reward=1.,end=True)
            else:advance(b,.3,reward=1.,action=1,end=True)
            results.append(b.commit()["a1"][0].discounted_reward)
        self.assertEqual(results[0],results[1])

    def test_inputs_are_copied_and_bad_time_or_metadata_is_not_a_zero_reward(self):
        b=DecisionEpisode({"a1":"s"});x=np.array([1.,2.]);mask=READY.copy()
        b.start("a1",x,mask,1,-1.,0.,policy_id="s",time_s=0.)
        x[:]=999.;mask[:]=False
        with self.assertRaises(ValueError):advance(b,float("nan"),action=1,end=True)
        advance(b,.02,action=1,end=True);r=b.commit()["a1"][0]
        np.testing.assert_array_equal(r.observation,[1.,2.]);self.assertTrue(r.action_mask.all())
        self.assertFalse(r.observation.flags.writeable)

    def test_overflow_rejected_before_advancing_or_mutating_pending_reward(self):
        b=DecisionEpisode({"a1":"s"},gamma_per_second=1.);start(b)
        advance(b,.02,reward=1e308,action=1)
        with self.assertRaises(ValueError):advance(b,.04,reward=1e308,end=True)
        self.assertEqual(b.time_s,.02)
        advance(b,.04,reward=-1e308,end=True)
        self.assertEqual(b.commit()["a1"][0].discounted_reward,0.)

    def test_gae_rejects_forged_nonfinite_or_invalid_transition_fields(self):
        b=DecisionEpisode({"a1":"s"});start(b);advance(b,.02,action=1,end=True)
        row=b.commit()["a1"][0]
        for changes in ({"start_s":float("nan")},{"end_s":float("nan")},
                        {"value":float("inf")},{"discounted_reward":float("nan")},
                        {"discount":1.1},{"discount":0.},{"episode_end":1}):
            with self.assertRaises(ValueError):physical_time_gae([replace(row,**changes)])
        with self.assertRaises(ValueError):physical_time_gae([object()])

    def test_lambda_one_complete_episode_returns_do_not_depend_on_value_guesses(self):
        outputs=[]
        for values in ((0.,0.),(.7,-.4)):
            b=DecisionEpisode({"a1":"s"},gamma_per_second=.9)
            start(b,action=0,value=values[0]);advance(b,.02,ready=True)
            start(b,action=1,value=values[1],time=.02);advance(b,1.,reward=1.,action=1,end=True)
            outputs.append(physical_time_gae(b.commit()["a1"])[1])
        np.testing.assert_allclose(outputs[0],outputs[1],atol=1e-14)


if __name__=="__main__":unittest.main()
