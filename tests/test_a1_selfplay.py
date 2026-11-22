"""Pure short fixtures only: no original-model native run or optimizer."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from sim.a1_policy import A1FeatureEncoder, A1Policy
from sim.a1_rollout import DecisionEpisode
from sim.a1_selfplay import (FrozenSnapshotPool, _NATIVE_ARRAYS, batch_identity,
    collect_episode, environment_identity, export_transitions, metadata_hash,
    run_zero_update, weights_hash, zero_update_replay)
from sim.assembly import ROOT
from sim.training_contacts import DEPLOYED_RULE_VERSION, TrainingEnvironmentError


def policy():
    return A1Policy(A1FeatureEncoder((1., 27.6, 28.72, 30., 25.)),
                    hidden_sizes=(8, 8), guard_hazard_per_s=1000.)


def metadata(actor):
    physical = {'fixture': 'pure_no_physics', 'target': [.32, 0., 1.05],
                'implementation_sha256': {'sim/training_contacts.py': 'd'*64}}
    return {'bank_numeric_sha256': 'a'*64, 'runtime_profile_sha256': 'b'*64,
        'deployed_rule_version': DEPLOYED_RULE_VERSION,
        'environment_sha256': metadata_hash(physical), 'full_environment_identity': physical,
        'encoder': actor.encoder.metadata(), 'policy_version': actor.metadata()['policy_version'],
        'policy_metadata': actor.metadata()}


class FakeEnv:
    """Two-tick nonzero skills; deliberately misleading PRE-step choice info."""
    def __init__(self, *, error_at=None, game='timeout'):
        self.error_at, self.game = error_at, game
        self.library = SimpleNamespace(skills={name: SimpleNamespace(name=name, action_id=i, duration_s=d)
            for i, (name, d) in enumerate(zip(('guard', 'stab', 'slash_left', 'parry_left', 'parry_right'),
                                             (1., 27.6, 28.72, 30., 25.)))})
        self.scene = None; self.cfg = {}; self._reset_done = False

    def observations(self):
        result = {}
        for slot in ('a1', 'a2'):
            busy = self.remaining[slot] > 0
            result[slot] = {'time_s': self.steps*.02,
                'own_skill': {'busy': busy, 'action_id': self.actions[slot] if busy else 0,
                    'elapsed_s': .02 if busy else 0., 'action_mask': [True]+[not busy]*4},
                'own_q_rad': np.zeros(7), 'own_qvel_rad_s': np.zeros(7),
                'opponent_tip_minus_own_target_local_m': np.array([.2, 0., .3]),
                'opponent_grip_minus_own_target_local_m': np.array([.4, 0., .3]),
                'opponent_tip_velocity_local_m_s': np.array([.02, 0., 0.]),
                'opponent_skill_id': 'must not become a feature'}
        return result

    def reset(self):
        self.steps = 0; self.remaining = {'a1': 0, 'a2': 0}; self.actions = {'a1': 0, 'a2': 0}
        self._reset_done = True; self.log = []
        return self.observations(), {}

    def step(self, actions):
        if self.error_at == self.steps+1:
            raise TrainingEnvironmentError({'status': 'evidence_error', 'reason': 'fixture'})
        before = self.observations(); effective = {}; accepted = {}
        for slot in ('a1', 'a2'):
            ready = not before[slot]['own_skill']['busy']
            effective[slot] = int(actions[slot]) if ready else 0
            accepted[slot] = int(actions[slot]) == effective[slot]
            if ready and effective[slot]:
                self.remaining[slot] = 2; self.actions[slot] = effective[slot]
        self.log.append(copy.deepcopy(actions)); self.steps += 1
        self.remaining = {slot: max(0, count-1) for slot, count in self.remaining.items()}
        ended = self.steps == 4
        rewards = {slot: -1. if ended and self.game == 'double_foul' else 0. for slot in ('a1', 'a2')}
        return self.observations(), rewards, bool(ended and self.game == 'double_foul'), bool(ended and self.game == 'timeout'), {
            'effective_actions': effective, 'accepted': accepted, 'elapsed_s': .02,
            'choice_enabled': {s: not before[s]['own_skill']['busy'] for s in ('a1', 'a2')}}

    def result(self):
        return {'status': 'terminated' if self.game == 'double_foul' and self.steps == 4 else 'truncated' if self.steps == 4 else 'running',
            'completed_time_s': self.steps*.02, 'round_result': {'status': self.game},
            'reproducibility': {'bank_numeric_and_metadata_sha256': 'a'*64}}

    def export(self, path):
        target = Path(path); target.mkdir(parents=True, exist_ok=False)
        (target/'summary.json').write_text(json.dumps(self.result()))
        return target


class SelfPlayContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.learner = policy(); self.meta = metadata(self.learner)
        self.pool = FrozenSnapshotPool(); self.pool.add(self.learner, self.meta)
        self.opponent = self.pool.sample(torch.Generator().manual_seed(1))

    def generators(self):
        return {slot: torch.Generator().manual_seed(seed) for slot, seed in (('a1', 55), ('a2', 66))}

    def collect(self, env, slot='a1'):
        with patch('sim.a1_selfplay.snapshot_metadata', side_effect=lambda env, actor: metadata(actor)):
            return collect_episode(env, self.learner, self.opponent, slot, self.generators(), 'current')

    def test_snapshot_is_deep_frozen_and_external_mutation_does_not_reach_pool(self):
        before = weights_hash(self.opponent.policy)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in self.opponent.policy.parameters()))
        with torch.no_grad():
            self.learner.actor.bias[1] += 3.
            self.opponent.policy.actor.bias[2] += 4.
        fresh = self.pool.sample(torch.Generator().manual_seed(1))
        self.assertEqual(weights_hash(fresh.policy), before)
        fresh.metadata['encoder']['feature_order'].clear()
        self.assertEqual(len(self.pool.sample(torch.Generator().manual_seed(1)).metadata['encoder']['feature_order']), 30)

    def test_pool_capacity_ids_and_role_balance(self):
        identities = [self.opponent.snapshot_id]
        for _ in range(5):
            with torch.no_grad(): self.learner.actor.bias[1] += .1
            identities.append(self.pool.add(self.learner, metadata(self.learner)))
        self.assertEqual(len(self.pool), 4)
        self.assertNotIn(identities[0], [s['snapshot_id'] for s in self.pool.metadata()])
        roles = [self.pool.for_episode(i, torch.Generator().manual_seed(i))[1] for i in range(4)]
        self.assertEqual(roles, ['a1', 'a2', 'a1', 'a2'])
        for capacity in (0, 5, True):
            with self.assertRaises(ValueError): FrozenSnapshotPool(capacity)

    def test_pool_rejects_mixed_environment_bank_profile_or_encoder(self):
        for key in ('bank_numeric_sha256', 'runtime_profile_sha256', 'full_environment_identity'):
            changed = copy.deepcopy(self.meta)
            if key == 'full_environment_identity':
                changed[key]['target'][0] += .1
                changed['environment_sha256'] = metadata_hash(changed[key])
            else: changed[key] = 'c'*64
            with self.assertRaises(ValueError): self.pool.add(self.learner, changed)
            with self.assertRaises(ValueError): self.pool.sample(torch.Generator(), expected_metadata=changed)

    def test_actual_environment_identity_detects_changed_native_caps(self):
        model = SimpleNamespace(**{name: np.ones((2, 2)) for name in _NATIVE_ARRAYS})
        model.opt = SimpleNamespace(integrator=1, timestep=.001, solver=2, iterations=100,
            tolerance=1e-8, disableflags=0, enableflags=0, gravity=np.array([0., 0., -9.81]))
        implementations = {'sim/a1_env.py': 'd'*64, 'sim/training_contacts.py': 'e'*64,
            'sim/foul_attribution.py': 'f'*64, 'sim/a1_control.py': 'a'*64, 'sim/fencing_rules.py': 'b'*64}
        env = SimpleNamespace(scene=SimpleNamespace(model=model, spec=SimpleNamespace(to_xml=lambda: '<same/>'),
            config={'target': [0., 0., 1.]}, source_hashes={'model': 'same'}), cfg={'physics_timestep_s': .001}, max_duration_s=80.,
            result=lambda: {'reproducibility': {'implementation_sha256': implementations}})
        before = metadata_hash(environment_identity(env))
        model.actuator_forcerange[0, 1] += 1.
        self.assertNotEqual(before, metadata_hash(environment_identity(env)))
        model.actuator_forcerange[0, 1] -= 1.
        implementations['sim/a1_control.py'] = 'c'*64
        self.assertNotEqual(before, metadata_hash(environment_identity(env)))

    def test_only_learner_choices_recorded_and_ready_comes_from_next_observation(self):
        env = FakeEnv(); before = weights_hash(self.learner)
        result = self.collect(env, 'a2')
        rows = result['transitions']
        self.assertEqual(len(rows), 2)  # PRE-step info would falsely create four rows.
        self.assertTrue(all(r.slot == 'a2' and r.policy_id == 'current' for r in rows))
        np.testing.assert_allclose([r.duration_s for r in rows], [.04, .04])
        self.assertEqual(result['collected_opponent_samples'], 0)
        self.assertEqual(result['statistics']['busy_environment_frames'], 2)
        self.assertEqual(result['statistics']['nonzero_skill_choices'], 2)
        self.assertEqual(result['statistics']['learner_ready_with_visible_opponent_motion_frames'], 2)
        self.assertEqual(weights_hash(self.learner), before)
        self.assertLess(result['zero_update_replay']['maximum_ratio_difference_from_one'], 1e-5)
        self.assertTrue(all(p.grad is None for p in self.learner.parameters()))

    def test_double_foul_is_not_zero_sum_and_timeout_zero_bootstraps(self):
        foul = self.collect(FakeEnv(game='double_foul'))
        self.assertLess(foul['transitions'][-1].discounted_reward, 0.)
        timeout = self.collect(FakeEnv(game='timeout'))
        np.testing.assert_allclose(timeout['returns'], 0., atol=1e-7)
        self.assertTrue(timeout['transitions'][-1].episode_end)

    def test_any_error_aborts_whole_episode_and_does_not_retry(self):
        instances = []
        class SpyEpisode(DecisionEpisode):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs); instances.append(self)
        with patch('sim.a1_selfplay.DecisionEpisode', SpyEpisode):
            with self.assertRaises(TrainingEnvironmentError) as caught:
                self.collect(FakeEnv(error_at=2))
        self.assertTrue(caught.exception.collection_invalid['whole_episode_aborted'])
        self.assertEqual(caught.exception.collection_invalid['committed_samples'], 0)
        self.assertTrue(all(not rows for rows in instances[-1]._rows.values()))
        with patch.object(self.learner, 'act', side_effect=ValueError('bad actor')):
            with self.assertRaises(ValueError) as bad:
                self.collect(FakeEnv())
        self.assertTrue(bad.exception.collection_invalid['whole_episode_aborted'])

    def test_metadata_and_weights_hashes_are_canonical_and_batch_identity_is_frozen(self):
        self.assertEqual(metadata_hash({'b': 2, 'a': 1}), metadata_hash({'a': 1, 'b': 2}))
        self.assertEqual(weights_hash(copy.deepcopy(self.learner)), weights_hash(self.learner))
        identity = batch_identity(self.learner, 'current', self.meta)
        self.assertEqual(identity['feature_version'], self.learner.encoder.metadata()['feature_version'])
        self.assertEqual(identity['weights_sha256'], weights_hash(self.learner))

    def test_numeric_transition_export_and_replay_mismatch_rejected(self):
        episode = self.collect(FakeEnv())
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/'rows.npz'; export_transitions(target, episode)
            with np.load(target, allow_pickle=False) as data:
                self.assertEqual(data['observation'].shape, (2, 30))
                self.assertEqual(data['action_mask'].dtype, bool)
        changed = list(episode['transitions']); changed[0] = replace(changed[0], log_prob=changed[0].log_prob+.1)
        with self.assertRaises(ValueError): zero_update_replay(self.learner, changed)

    def test_mutated_snapshot_metadata_id_is_not_accepted(self):
        self.opponent.metadata['unbound_note'] = 'changed after selecting this snapshot'
        with self.assertRaises(ValueError): self.collect(FakeEnv())

    def test_cli_error_saves_evidence_and_stops_without_substitute_episode(self):
        env = FakeEnv(error_at=2)
        with tempfile.TemporaryDirectory() as directory, \
             patch('sim.a1_selfplay.A1DuelEnv', return_value=env), \
             patch('sim.a1_selfplay.snapshot_metadata', side_effect=lambda env, actor: metadata(actor)), \
             patch('sim.a1_selfplay.freeze_experiment'):
            output = Path(directory)/'new'
            with self.assertRaises(TrainingEnvironmentError):
                run_zero_update('fake', ROOT/'config/a1_runtime_rk4_candidate.json', output, episodes=2)
            self.assertTrue((output/'invalid_or_interrupted_env/summary.json').exists())
            failure = json.loads((output/'driver_failure.json').read_text())
            self.assertEqual(failure['error_type'], 'TrainingEnvironmentError')
            self.assertTrue(failure['collection_invalid']['whole_episode_aborted'])
            saved = json.loads((output/'summary.json').read_text())
            self.assertEqual(saved['status'], 'zero_update_aborted_invalid')
            self.assertEqual(saved['episodes'], [])
            self.assertFalse((output/'episode_01').exists())

    def test_bounded_cli_arguments_fail_before_output_or_env(self):
        with tempfile.TemporaryDirectory() as directory, patch('sim.a1_selfplay.A1DuelEnv') as build:
            for count in (0, 9, True):
                with self.assertRaises(ValueError): run_zero_update('none', 'none', Path(directory)/'new', episodes=count)
            build.assert_not_called()


if __name__ == '__main__': unittest.main()
