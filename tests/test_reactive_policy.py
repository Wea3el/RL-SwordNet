"""Rule-policy observation and own-clock regressions, not game effectiveness."""
import copy
import unittest

from sim.shared_control import ReactivePolicy, ReactiveCounterPolicy


def obs(time=1., *, busy=False, action=0, p=(.4,.1,.25), v=(-.1,-.08,-.2)):
    return {"time_s": time, "own_skill": {"busy": busy, "action_id": action},
            "opponent_tip_minus_own_target_local_m": p,
            "opponent_tip_velocity_local_m_s": v}


class ReactivePolicyTests(unittest.TestCase):
    def test_preparation_defence_depends_on_current_geometry_not_global_time(self):
        for time in (1., 15., 100.):
            policy = ReactivePolicy(attack_wait_s=(30.,30.))
            self.assertEqual(policy.decide(obs(time)), 2)
        policy = ReactivePolicy(attack_wait_s=(30.,30.))
        self.assertEqual(policy.decide(obs(v=(0.,0.,0.))), 0)
        self.assertEqual(policy.decide(obs(p=(.4,.1,.5))), 0)

    def test_busy_never_requests_cancel_and_same_seed_same_wait(self):
        policies = [ReactivePolicy(seed=5) for _ in range(2)]
        for time in (.1,.5,1.,2.,3.):
            observation = obs(time, p=(.5,.3,.5), v=(0.,0.,0.))
            self.assertEqual(policies[0].decide(copy.deepcopy(observation)), policies[1].decide(copy.deepcopy(observation)))
        self.assertEqual(policies[0].decide(obs(busy=True,action=1)), 0)

    def test_counter_is_triggered_only_by_own_finished_parry(self):
        policy = ReactiveCounterPolicy(attack_wait_s=(30.,30.))
        self.assertEqual(policy.decide(obs(busy=True,action=2)), 0)
        self.assertEqual(policy.decide(obs(8., p=(.05,.3,0.),v=(0.,0.,0.))), 0)
        self.assertEqual(policy.decide(obs(8., p=(.8,.3,.5),v=(.01,0.,0.))), 1)
        self.assertEqual(policy.decide(obs(8.1,p=(.8,.3,.5),v=(0.,0.,0.))), 0)


if __name__ == "__main__":
    unittest.main()
