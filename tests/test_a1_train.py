"""Bounded trainer plumbing with short synthetic episodes, not native fencing."""
import contextlib
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from sim.a1_policy import A1Policy
from sim.a1_ppo import ppo_update_once
from sim.a1_selfplay import FrozenSnapshotPool, metadata_hash
from sim.a1_train import _input_manifest, run_training
from sim.training_contacts import DEPLOYED_RULE_VERSION, TrainingEnvironmentError


def metadata(actor):
    physical = {'fixture': 'trainer_no_native_physics', 'immutable': True}
    return {'bank_numeric_sha256': 'a'*64, 'runtime_profile_sha256': 'b'*64,
        'deployed_rule_version': DEPLOYED_RULE_VERSION,
        'environment_sha256': metadata_hash(physical), 'full_environment_identity': physical,
        'encoder': actor.encoder.metadata(), 'policy_version': actor.metadata()['policy_version'],
        'policy_metadata': actor.metadata()}


class FakeEnv:
    def __init__(self, *args, error_reset=None, **kwargs):
        self.error_reset = error_reset; self.resets = 0; self._reset_done = False
        self.library = SimpleNamespace(skills={name: SimpleNamespace(name=name, action_id=i, duration_s=d)
            for i, (name, d) in enumerate(zip(('guard', 'stab', 'slash_left', 'parry_left', 'parry_right'),
                                             (1., 27.6, 28.72, 30., 25.)))})
        self.scene = None; self.cfg = {}

    def observations(self):
        return {slot: {'time_s': self.steps*.02,
            'own_skill': {'busy': self.busy[slot] > 0, 'action_id': self.actions[slot],
                         'elapsed_s': .02 if self.busy[slot] else 0.,
                         'action_mask': [True]+[not self.busy[slot]]*4},
            'own_q_rad': np.zeros(7), 'own_qvel_rad_s': np.zeros(7),
            'opponent_tip_minus_own_target_local_m': np.array([.2, 0., .3]),
            'opponent_grip_minus_own_target_local_m': np.array([.4, 0., .3]),
            'opponent_tip_velocity_local_m_s': np.array([.02, 0., 0.])}
            for slot in ('a1', 'a2')}

    def reset(self):
        self.resets += 1; self.steps = 0; self.busy = {'a1': 0, 'a2': 0}
        self.actions = {'a1': 0, 'a2': 0}; self._reset_done = True
        return self.observations(), {}

    def step(self, actions):
        if self.resets == self.error_reset:
            raise TrainingEnvironmentError({'status': 'fixture_evidence_error'})
        effective = {}; accepted = {}
        for slot in ('a1', 'a2'):
            effective[slot] = actions[slot] if not self.busy[slot] else 0
            accepted[slot] = effective[slot] == actions[slot]
            if effective[slot]: self.busy[slot] = 2; self.actions[slot] = effective[slot]
        self.steps += 1; self.busy = {s: max(0, n-1) for s, n in self.busy.items()}
        ended = self.steps == 4
        return self.observations(), {'a1': float(ended), 'a2': -float(ended)}, ended, False, {
            'effective_actions': effective, 'accepted': accepted, 'elapsed_s': .02}

    def result(self):
        return {'status': 'terminated' if self.steps == 4 else 'running',
            'completed_time_s': self.steps*.02,
            'round_result': {'status': 'synthetic_fixture_only'}}

    def export(self, path):
        path = Path(path); path.mkdir(parents=True, exist_ok=False)
        (path/'summary.json').write_text(json.dumps(self.result()))


class TrainerContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bank = self.root/'bank'; self.bank.mkdir()
        self.profile = self.root/'runtime.json'; self.profile.write_text('{}')
        self.output = self.root/'run'
        self.stack = contextlib.ExitStack(); self.addCleanup(self.stack.close)
        self.stack.enter_context(patch('sim.a1_train.A1DuelEnv', FakeEnv))
        self.stack.enter_context(patch('sim.a1_train.load_runtime_profile', return_value=object()))
        self.stack.enter_context(patch('sim.a1_train.snapshot_metadata', side_effect=lambda env, actor: metadata(actor)))
        self.stack.enter_context(patch('sim.a1_selfplay.snapshot_metadata', side_effect=lambda env, actor: metadata(actor)))
        self.stack.enter_context(patch('sim.a1_train.freeze_experiment'))
        self.stack.enter_context(patch('sim.a1_train._input_manifest', return_value={'fixture_source': 'a'*64}))

    def run_fixture(self, **kwargs):
        return run_training(self.bank, self.profile, self.output, **kwargs)

    def test_two_updates_persistent_adam_balanced_slots_and_growing_history(self):
        sizes = []; optimizer_ids = []; calls = []
        class RecordingPool(FrozenSnapshotPool):
            def for_episode(self, index, generator, **kwargs):
                sizes.append(len(self)); return super().for_episode(index, generator, **kwargs)
        def update(*args, **kwargs):
            optimizer_ids.append(id(kwargs['optimizer']))
            calls.append(copy.deepcopy(kwargs['batch_identity']))
            return ppo_update_once(*args, **kwargs)
        with patch('sim.a1_train.FrozenSnapshotPool', RecordingPool), \
                patch('sim.a1_train.ppo_update_once', side_effect=update):
            result = self.run_fixture()
        self.assertEqual(result['completed_optimizer_updates'], 2)
        self.assertEqual(sizes, [1, 1, 2, 2])
        self.assertEqual(len(set(optimizer_ids)), 1)
        self.assertNotEqual(calls[0], calls[1])
        self.assertEqual([x['optimizer_step_count_after'] for x in result['updates']], [1, 2])
        self.assertEqual([r['statistics']['learner_slot'] for r in result['episodes']], ['a1', 'a2', 'a1', 'a2'])
        self.assertEqual(len(result['snapshot_pool']), 3)
        self.assertFalse(result['learned_fencing_ability_demonstrated'])
        checkpoint = torch.load(self.output/'final_candidate.pt', weights_only=True)
        self.assertEqual(checkpoint['completed_optimizer_updates'], 2)
        self.assertFalse(checkpoint['resume_supported'])
        self.assertIn('rng_state', checkpoint); self.assertIn('optimizer_state_dict', checkpoint)
        self.assertEqual(len(list((self.output/'snapshots').glob('*.pt'))), 3)
        for episode in self.output.glob('update_*/episode_*'):
            arrays = np.load(episode/'transitions.npz')
            self.assertIn('next_observation', arrays.files)
            self.assertTrue(arrays['episode_end'][-1])
            self.assertTrue((episode/'env'/'summary.json').is_file())

    def test_error_in_second_episode_rejects_batch_without_optimizer_or_retry(self):
        env = FakeEnv(error_reset=3)
        with patch('sim.a1_train.A1DuelEnv', return_value=env), \
                patch('sim.a1_train.ppo_update_once') as update:
            with self.assertRaises(TrainingEnvironmentError): self.run_fixture()
        self.assertEqual(env.resets, 3)
        update.assert_not_called()
        summary = json.loads((self.output/'summary.json').read_text())
        self.assertEqual(summary['completed_optimizer_updates'], 0)
        self.assertEqual(len(summary['episodes']), 1)
        self.assertTrue(summary['failure']['pending_batch_rejected'])
        self.assertTrue((self.output/'invalid_or_interrupted_env'/'summary.json').exists())
        self.assertFalse((self.output/'final_candidate.pt').exists())

    def test_later_invalid_batch_preserves_only_completed_prior_update(self):
        env = FakeEnv(error_reset=5)
        with patch('sim.a1_train.A1DuelEnv', return_value=env):
            with self.assertRaises(TrainingEnvironmentError): self.run_fixture()
        summary = json.loads((self.output/'summary.json').read_text())
        self.assertEqual(summary['completed_optimizer_updates'], 1)
        checkpoint = torch.load(self.output/'aborted_candidate.pt', weights_only=True)
        self.assertEqual(checkpoint['completed_optimizer_updates'], 1)
        self.assertTrue((self.output/'update_01'/'candidate.pt').exists())
        self.assertFalse((self.output/'update_02'/'candidate.pt').exists())

    def test_immutable_input_change_is_not_mixed_into_update(self):
        with patch('sim.a1_train._input_manifest', side_effect=[{'f': 'old'}, {'f': 'new'}]), \
                patch('sim.a1_train.ppo_update_once') as update:
            with self.assertRaisesRegex(ValueError, 'changed during'): self.run_fixture()
        update.assert_not_called()

    def test_manifest_includes_real_model_and_external_skill_sources(self):
        (self.root/'sim').mkdir(); (self.root/'plan1_skills').mkdir()
        (self.root/'sim'/'source.py').write_text('a')
        (self.root/'plan1_skills'/'matching.py').write_text('b')
        (self.root/'robot.xml').write_text('<robot/>')
        (self.root/'scene.xml').write_text('<scene/>')
        with patch('sim.a1_train.ROOT', self.root):
            before = _input_manifest(self.bank, self.profile)
            for relative in ('robot.xml', 'scene.xml', 'sim/source.py', 'plan1_skills/matching.py'):
                self.assertIn(str(self.root/relative), before)
            (self.root/'robot.xml').write_text('<changed/>')
            self.assertNotEqual(before, _input_manifest(self.bank, self.profile))

    def test_keyboard_interrupt_preserves_failure_and_no_update(self):
        with patch('sim.a1_train.collect_episode', side_effect=KeyboardInterrupt), \
                patch('sim.a1_train.ppo_update_once') as update:
            with self.assertRaises(KeyboardInterrupt): self.run_fixture()
        update.assert_not_called()
        failure = json.loads((self.output/'failure.json').read_text())
        self.assertEqual(failure['type'], 'KeyboardInterrupt')
        self.assertEqual(failure['completed_optimizer_updates'], 0)

    def test_stale_collector_identity_is_rejected_before_update(self):
        from sim.a1_selfplay import collect_episode
        def stale(*args, **kwargs):
            result = collect_episode(*args, **kwargs)
            result['batch_identity']['weights_sha256'] = 'f'*64
            return result
        with patch('sim.a1_train.collect_episode', side_effect=stale), \
                patch('sim.a1_train.ppo_update_once') as update:
            with self.assertRaisesRegex(ValueError, 'immutable pending'): self.run_fixture()
        update.assert_not_called()

    def test_invalid_configuration_does_not_create_output(self):
        for bad in ({'updates': 0}, {'updates': 9}, {'updates': True},
                {'episodes_per_update': 1}, {'episodes_per_update': 3},
                {'updates': 8, 'episodes_per_update': 8}, {'seed': -1},
                {'seed': True}, {'seed': 2**32}, {'learning_rate': float('nan')},
                {'learning_rate': True}, {'learning_rate': np.bool_(True)},
                {'learning_rate': .1}, {'learning_rate': 1j}):
            with self.subTest(bad=bad), self.assertRaises(ValueError): self.run_fixture(**bad)
            self.assertFalse(self.output.exists())

    def test_existing_output_or_output_inside_bank_is_rejected(self):
        self.output.mkdir(); (self.output/'keep').write_text('unchanged')
        with self.assertRaises(FileExistsError): self.run_fixture()
        self.assertEqual((self.output/'keep').read_text(), 'unchanged')
        with self.assertRaisesRegex(ValueError, 'inside an immutable'):
            run_training(self.bank, self.profile, self.bank/'run')


if __name__ == '__main__': unittest.main()
