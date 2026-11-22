"""Frozen opponents and decision-only collection; ZERO optimizer updates.

This is A1-family plumbing, not trained self-play performance, A2, or bank
admission. Opponent identities are bookkeeping, never actor input features.
Invalid rounds abort the whole staged transaction and stop this driver.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
import numbers
from pathlib import Path

import numpy as np
import torch

from .a1_env import A1DuelEnv
from .a1_policy import A1FeatureEncoder, A1Policy, FEATURE_VERSION
from .a1_rollout import DecisionEpisode, physical_time_gae, DEFAULT_GAMMA_PER_SECOND
from .a1_runtime import load_runtime_profile
from .assembly import ROOT
from .skill_validation import freeze_experiment, save_json
from .training_contacts import DEPLOYED_RULE_VERSION


def metadata_hash(value):
    """Canonical SHA256 of JSON metadata (also shared by the PPO adapter)."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def weights_hash(policy):
    """Canonical sorted state_dict SHA: names, Torch dtype, shape, raw bytes."""
    digest = hashlib.sha256()
    for name, value in sorted(policy.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        if not torch.isfinite(tensor).all():
            raise ValueError('Nonfinite policy parameters are not a snapshot')
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)],
                                 separators=(',', ':')).encode())
        digest.update(tensor.numpy().tobytes(order='C'))
    return digest.hexdigest()


def batch_identity(policy, policy_id, compatibility_metadata=None):
    if not isinstance(policy_id, str) or not policy_id.strip():
        raise ValueError('Explicit on-policy identity required')
    result = {'policy_id': policy_id, 'weights_sha256': weights_hash(policy),
        'policy_metadata_sha256': metadata_hash(policy.metadata()),
        'feature_version': policy.encoder.metadata()['feature_version']}
    if compatibility_metadata is not None:
        result['compatibility_metadata_sha256'] = metadata_hash(compatibility_metadata)
    return result


_NATIVE_ARRAYS = ('body_mass', 'body_inertia', 'body_pos', 'body_quat',
    'geom_type', 'geom_size', 'geom_pos', 'geom_quat', 'geom_contype', 'geom_conaffinity',
    'geom_friction', 'geom_solref', 'geom_solimp', 'jnt_range', 'actuator_gainprm',
    'actuator_biasprm', 'actuator_ctrlrange', 'actuator_forcerange', 'actuator_gear',
    'dof_damping', 'dof_armature')


def environment_identity(env):
    """Bind actual physical/game settings, not just the numerical clock file."""
    arrays = {}
    for name in _NATIVE_ARRAYS:
        value = np.ascontiguousarray(getattr(env.scene.model, name))
        arrays[name] = {'dtype': value.dtype.str, 'shape': list(value.shape),
                       'sha256': hashlib.sha256(value.tobytes()).hexdigest()}
    option = env.scene.model.opt
    return {'runtime_config': copy.deepcopy(env.cfg),
        'scene_config': copy.deepcopy(env.scene.config),
        'source_hashes': copy.deepcopy(env.scene.source_hashes),
        'implementation_sha256': copy.deepcopy(env.result()['reproducibility']['implementation_sha256']),
        'episode_physical_time_limit_s': env.max_duration_s,
        'actual_native_arrays': arrays,
        'actual_native_options': {'integrator': int(option.integrator), 'timestep': float(option.timestep),
            'solver': int(option.solver), 'iterations': int(option.iterations),
            'tolerance': float(option.tolerance), 'disableflags': int(option.disableflags),
            'enableflags': int(option.enableflags), 'gravity': np.asarray(option.gravity).tolist()},
        'exported_spec_xml_sha256': hashlib.sha256(env.scene.spec.to_xml().encode()).hexdigest()}


def snapshot_metadata(env, policy):
    physical = environment_identity(env)
    return {'bank_numeric_sha256': env.result()['reproducibility']['bank_numeric_and_metadata_sha256'],
        'runtime_profile_sha256': env.profile.source_sha256,
        'deployed_rule_version': DEPLOYED_RULE_VERSION,
        'environment_sha256': metadata_hash(physical), 'full_environment_identity': physical,
        'encoder': policy.encoder.metadata(), 'policy_version': policy.metadata()['policy_version'],
        'policy_metadata': policy.metadata()}


_BINDING_KEYS = ('bank_numeric_sha256', 'runtime_profile_sha256', 'deployed_rule_version',
                 'environment_sha256', 'full_environment_identity', 'encoder', 'policy_version')


def _binding(metadata):
    return {key: metadata[key] for key in _BINDING_KEYS}


def _validate_metadata(policy, metadata):
    if not isinstance(policy, A1Policy) or not isinstance(metadata, dict) \
            or not set(_BINDING_KEYS).issubset(metadata):
        raise ValueError('An A1 policy and complete environment-bound snapshot metadata are required')
    for key in ('bank_numeric_sha256', 'runtime_profile_sha256', 'environment_sha256'):
        value = metadata[key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError(f'Canonical SHA256 required for {key}')
    if metadata['environment_sha256'] != metadata_hash(metadata['full_environment_identity']):
        raise ValueError('Physical/game identity does not match its SHA256')
    if metadata['encoder'] != policy.encoder.metadata() \
            or metadata['policy_version'] != policy.metadata()['policy_version'] \
            or metadata.get('policy_metadata') != policy.metadata() \
            or metadata['deployed_rule_version'] != DEPLOYED_RULE_VERSION:
        raise ValueError('Snapshot policy, encoder, or deployed rule metadata mismatch')
    metadata_hash(metadata)


@dataclass(frozen=True)
class FrozenSnapshot:
    snapshot_id: str
    policy: A1Policy
    metadata: dict
    weights_sha256: str


class FrozenSnapshotPool:
    """At most four independently owned, eval-mode, no-gradient opponents."""
    def __init__(self, max_size=4):
        if type(max_size) is not int or not 1 <= max_size <= 4:
            raise ValueError('Snapshot capacity must be an integer1--4')
        self.max_size, self._snapshots, self._binding_hash = max_size, [], None

    def __len__(self): return len(self._snapshots)

    def add(self, policy, metadata):
        _validate_metadata(policy, metadata)
        binding_hash = metadata_hash(_binding(metadata))
        if self._binding_hash is not None and binding_hash != self._binding_hash:
            raise ValueError('A pool cannot mix physical scenes, games, banks, profiles or encoders')
        frozen = copy.deepcopy(policy).cpu().eval().requires_grad_(False)
        for parameter in frozen.parameters(): parameter.grad = None
        weight_hash = weights_hash(frozen)
        snapshot_id = metadata_hash({'weights_sha256': weight_hash, 'metadata': metadata})
        if any(s.snapshot_id == snapshot_id for s in self._snapshots): return snapshot_id
        self._binding_hash = binding_hash
        self._snapshots.append(FrozenSnapshot(snapshot_id, frozen, copy.deepcopy(metadata), weight_hash))
        self._snapshots = self._snapshots[-self.max_size:]
        return snapshot_id

    def sample(self, generator, *, expected_metadata=None):
        if not self._snapshots or not isinstance(generator, torch.Generator):
            raise ValueError('A nonempty pool and explicit Torch random generator are required')
        if expected_metadata is not None and metadata_hash(_binding(expected_metadata)) != self._binding_hash:
            raise ValueError('Opponent pool is incompatible with this environment')
        index = int(torch.randint(len(self._snapshots), (1,), generator=generator))
        snapshot = self._snapshots[index]
        if weights_hash(snapshot.policy) != snapshot.weights_sha256 or snapshot.snapshot_id != metadata_hash(
                {'weights_sha256': snapshot.weights_sha256, 'metadata': snapshot.metadata}):
            raise ValueError('Stored frozen opponent weights or metadata identity changed')
        return copy.deepcopy(snapshot)

    def for_episode(self, index, generator, *, expected_metadata=None):
        if type(index) is not int or index < 0:
            raise ValueError('Nonnegative episode index required')
        return self.sample(generator, expected_metadata=expected_metadata), ('a1' if index % 2 == 0 else 'a2')

    def metadata(self):
        return [{'snapshot_id': s.snapshot_id, 'weights_sha256': s.weights_sha256,
                 'metadata': copy.deepcopy(s.metadata)} for s in self._snapshots]


def zero_update_replay(policy, transitions, *, tolerance=1e-5):
    rows = tuple(transitions)
    if not rows: raise ValueError('A valid complete episode must contain actual actor decisions')
    with torch.no_grad():
        logp, _, values = policy.evaluate_actions(np.stack([r.observation for r in rows]),
            np.array([r.action for r in rows]), np.stack([r.action_mask for r in rows]))
    new_logp, new_values = logp.cpu().numpy(), values.cpu().numpy()
    old_logp = np.array([r.log_prob for r in rows]); old_values = np.array([r.value for r in rows])
    ratios = np.exp(new_logp-old_logp)
    report = {'decision_rows': len(rows), 'maximum_log_prob_difference': float(np.max(np.abs(new_logp-old_logp))),
        'maximum_value_difference': float(np.max(np.abs(new_values-old_values))),
        'maximum_ratio_difference_from_one': float(np.max(np.abs(ratios-1.))), 'tolerance': tolerance}
    if not all(np.isfinite(v) for v in report.values()) or any(report[key] > tolerance for key in (
            'maximum_log_prob_difference', 'maximum_value_difference', 'maximum_ratio_difference_from_one')):
        raise ValueError(f'Zero-update replay mismatch: {report}')
    return report


def collect_episode(env, learner, opponent, learner_slot, generators, policy_id):
    """Collect ONE complete valid round, only current learner's actor choices.

    ``opponent`` is one frozen snapshot selected BEFORE this call; it cannot
    change mid-round. Any exception aborts the staged buffer and propagates.
    Env and caller retain invalid evidence; this function never retries.
    """
    if learner_slot not in ('a1', 'a2') or not isinstance(opponent, FrozenSnapshot):
        raise ValueError('Explicit learner physical slot and frozen opponent snapshot required')
    if not isinstance(generators, dict) or set(generators) != {'a1', 'a2'} \
            or any(not isinstance(g, torch.Generator) for g in generators.values()):
        raise ValueError('Independent explicit generators for both physical slots required')
    if generators['a1'] is generators['a2']:
        raise ValueError('The two actors must not share one mutable random generator')
    if any(p.device.type != 'cpu' for policy in (learner, opponent.policy) for p in policy.parameters()):
        raise ValueError('Initial collector supports CPU policies only')
    buffer = None
    stats = {'learner_slot': learner_slot, 'opponent_snapshot_id': opponent.snapshot_id,
        'learner_decisions': 0, 'ready_guard_wait_choices': 0, 'nonzero_skill_choices': 0,
        'learner_action_counts': [0]*5, 'busy_environment_frames': 0, 'unavailable_environment_frames': 0,
        'environment_steps': 0, 'learner_ready_with_visible_opponent_motion_frames': 0,
        'learner_ready_with_visible_opponent_motion_elapsed_s': 0.,
        'visible_motion_condition': 'current opponent tip speed norm>.01m/s and current own encoder.choice; not hidden attack phase'}
    try:
        observations, _ = env.reset()
        metadata = snapshot_metadata(env, learner)
        _validate_metadata(opponent.policy, opponent.metadata)
        if metadata_hash(_binding(metadata)) != metadata_hash(_binding(opponent.metadata)):
            raise ValueError('Frozen opponent identity differs from actual current environment')
        if weights_hash(opponent.policy) != opponent.weights_sha256 \
                or opponent.snapshot_id != metadata_hash({'weights_sha256': opponent.weights_sha256,
                                                           'metadata': opponent.metadata}) \
                or any(p.requires_grad for p in opponent.policy.parameters()):
            raise ValueError('Opponent is not the selected unchanged frozen snapshot')
        identity = batch_identity(learner, policy_id, metadata)
        initial_opponent_weights = weights_hash(opponent.policy)
        initial_opponent_metadata = metadata_hash(opponent.metadata)
        initial_policy_metadata = metadata_hash(learner.metadata())
        buffer = DecisionEpisode({learner_slot: policy_id}, start_s=observations[learner_slot]['time_s'])
        opponent_slot = 'a2' if learner_slot == 'a1' else 'a1'
        actors = {learner_slot: learner, opponent_slot: opponent.policy}
        while True:
            # Both acts consume only the SAME completed-state public snapshot.
            decisions = {slot: actors[slot].act(observations[slot], generator=generators[slot])
                         for slot in ('a1', 'a2')}
            choice = decisions[learner_slot]
            public = observations[learner_slot]
            moving_opportunity = bool(choice.actor_choice and
                np.linalg.norm(public['opponent_tip_velocity_local_m_s']) > .01)
            if choice.actor_choice:
                buffer.start(learner_slot, choice.features, choice.action_mask, choice.action,
                    choice.log_prob, choice.value, policy_id=policy_id, time_s=public['time_s'])
                stats['learner_decisions'] += 1
                stats['learner_action_counts'][choice.action] += 1
                stats['ready_guard_wait_choices' if choice.action == 0 else 'nonzero_skill_choices'] += 1
            else:
                stats['busy_environment_frames' if public['own_skill']['busy'] else 'unavailable_environment_frames'] += 1
            next_obs, rewards, terminated, truncated, info = env.step({s: d.action for s, d in decisions.items()})
            next_features, _, next_choice = learner.encoder.encode(next_obs[learner_slot])
            buffer.advance(time_s=next_obs[learner_slot]['time_s'], rewards={learner_slot: rewards[learner_slot]},
                next_observations={learner_slot: next_features}, ready_by_slot={learner_slot: next_choice},
                effective_actions={learner_slot: info['effective_actions'][learner_slot]},
                accepted={learner_slot: info['accepted'][learner_slot]}, episode_end=bool(terminated or truncated))
            stats['environment_steps'] += 1
            stats['learner_ready_with_visible_opponent_motion_frames'] += int(moving_opportunity)
            if moving_opportunity:
                stats['learner_ready_with_visible_opponent_motion_elapsed_s'] += info['elapsed_s']
            observations = next_obs
            if terminated or truncated: break
        if weights_hash(learner) != identity['weights_sha256'] \
                or metadata_hash(learner.metadata()) != initial_policy_metadata \
                or weights_hash(opponent.policy) != initial_opponent_weights \
                or metadata_hash(opponent.metadata) != initial_opponent_metadata:
            raise ValueError('Actor weights or metadata changed during this on-policy episode')
        rows = buffer.commit()[learner_slot]
        replay = zero_update_replay(learner, rows)
        advantages, returns = physical_time_gae(rows)  # Default lambda1, zero final bootstrap.
        return {'transitions': rows, 'advantages': advantages, 'returns': returns,
            'statistics': stats, 'batch_identity': identity, 'snapshot_metadata': metadata,
            'zero_update_replay': replay, 'weights_unchanged': True,
            'env_result': env.result(), 'collected_opponent_samples': 0}
    except Exception as error:
        if buffer is not None: buffer.abort(f'{type(error).__name__}: {error}')
        error.collection_invalid = {'statistics': copy.deepcopy(stats),
            'whole_episode_aborted': True, 'committed_samples': 0,
            'error_type': type(error).__name__, 'error': str(error)}
        raise


def export_transitions(path, episode):
    rows = episode['transitions']
    np.savez_compressed(path, observation=np.stack([r.observation for r in rows]),
        next_observation=np.stack([r.next_observation for r in rows]),
        action_mask=np.stack([r.action_mask for r in rows]), action=np.array([r.action for r in rows]),
        old_log_prob=np.array([r.log_prob for r in rows]), old_value=np.array([r.value for r in rows]),
        discounted_reward=np.array([r.discounted_reward for r in rows]), discount=np.array([r.discount for r in rows]),
        start_s=np.array([r.start_s for r in rows]), end_s=np.array([r.end_s for r in rows]),
        episode_end=np.array([r.episode_end for r in rows]), advantages=episode['advantages'], returns=episode['returns'])


def run_zero_update(bank, runtime_profile, output, *, episodes=2, seed=0):
    if type(episodes) is not int or not 1 <= episodes <= 8 or type(seed) is not int or seed < 0:
        raise ValueError('Bounded1--8 episode count and nonnegative integer seed required')
    torch.set_num_threads(1); torch.manual_seed(seed)
    folder = Path(output); folder.mkdir(parents=True, exist_ok=False)
    env = None
    paths = ('sim/a1_selfplay.py', 'sim/a1_policy.py', 'sim/a1_rollout.py', 'sim/a1_env.py',
        'sim/a1_control.py', 'sim/a1_skills.py', 'sim/a1_runtime.py', 'sim/training_contacts.py', 'sim/foul_attribution.py')
    hashes = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
    summary = {'status': 'zero_update_running', 'seed': seed, 'requested_episodes': episodes,
        'episodes': [], 'optimizer_updates': 0, 'training_performed': False,
        'weights_status': 'untrained', 'cpu_threads': torch.get_num_threads(),
        'gamma_per_second': DEFAULT_GAMMA_PER_SECOND, 'lambda_per_second': 1.,
        'admitted_to_training': False, 'source_sha256': hashes}
    save_json(folder/'summary.json', summary)
    try:
        env = A1DuelEnv(bank, load_runtime_profile(runtime_profile), max_duration_s=80., development_mode=True)
        env.reset()  # No integration: establish exact compatibility identity.
        learner = A1Policy(A1FeatureEncoder.from_library(env.library)).cpu()
        metadata = snapshot_metadata(env, learner)
        pool = FrozenSnapshotPool(); snapshot_id = pool.add(learner, metadata)
        initial_weights = weights_hash(learner)
        torch.save({'status': 'untrained_zero_optimizer_updates', 'state_dict': learner.state_dict(),
            'policy_metadata': learner.metadata(), 'compatibility_metadata': metadata}, folder/'initial_untrained.pt')
        freeze_experiment(folder/'frozen', env.library, env.scene, cfg=env.cfg,
            implementation_paths=paths, metadata={'scope': 'bounded untrained ZERO-update collector smoke',
                'initial_snapshot_id': snapshot_id, 'compatibility_metadata': metadata,
                'learner_slots_alternate': True, 'maximum_episode_physical_duration_s': 80.})
        generators = {slot: torch.Generator(device='cpu').manual_seed(seed+offset)
                      for slot, offset in (('a1', 101), ('a2', 202))}
        pool_generator = torch.Generator(device='cpu').manual_seed(seed+303)
        for index in range(episodes):
            opponent, slot = pool.for_episode(index, pool_generator, expected_metadata=metadata)
            episode_folder = folder/f'episode_{index:02d}'; episode_folder.mkdir()
            print(json.dumps({'event': 'episode_start', 'index': index, 'learner_slot': slot,
                              'opponent_snapshot_id': opponent.snapshot_id}), flush=True)
            episode = collect_episode(env, learner, opponent, slot, generators,
                policy_id=metadata_hash({'weights_sha256': initial_weights, 'metadata': metadata}))
            env.export(episode_folder/'env')
            export_transitions(episode_folder/'transitions.npz', episode)
            record = {key: episode[key] for key in ('statistics', 'batch_identity', 'snapshot_metadata',
                                                   'zero_update_replay', 'weights_unchanged', 'collected_opponent_samples')}
            record.update(index=index, env_status=episode['env_result']['status'],
                completed_time_s=episode['env_result']['completed_time_s'],
                round_result=episode['env_result']['round_result'],
                decision_rows=len(episode['transitions']), gamma_per_second=DEFAULT_GAMMA_PER_SECOND,
                lambda_per_second=1., game_end_zero_bootstrap=True)
            save_json(episode_folder/'collection.json', record)
            summary['episodes'].append(record); save_json(folder/'summary.json', summary)
            print(json.dumps({'event': 'episode_end', 'index': index, 'learner_slot': slot,
                'decision_rows': record['decision_rows'], 'statistics': record['statistics'],
                'round_result': record['round_result']}), flush=True)
        summary.update(status='zero_update_complete_untrained', initial_weights_sha256=initial_weights,
            final_weights_sha256=weights_hash(learner), weights_unchanged=weights_hash(learner)==initial_weights,
            snapshot_pool=pool.metadata(), implementations_unchanged=all(
                hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p, h in hashes.items()))
        if not summary['weights_unchanged'] or not summary['implementations_unchanged']:
            raise ValueError('Zero-update driver or policy changed during collection')
        save_json(folder/'summary.json', summary)
        return summary
    except Exception as error:
        export_error = None
        if env is not None and getattr(env, '_reset_done', False):
            try:
                env.export(folder/'invalid_or_interrupted_env')
            except Exception as failed_export:
                export_error = {'error_type': type(failed_export).__name__, 'error': str(failed_export)}
        failure = {'error_type': type(error).__name__, 'error': str(error),
            'details': getattr(error, 'details', None),
            'collection_invalid': getattr(error, 'collection_invalid', None),
            'evidence_export_error': export_error,
            'no_retry_or_substitute_draw': True, 'optimizer_updates': 0}
        save_json(folder/'driver_failure.json', failure)
        summary.update(status='zero_update_aborted_invalid', driver_failure=failure)
        save_json(folder/'summary.json', summary)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank', required=True)
    parser.add_argument('--runtime-profile', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--episodes', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    result = run_zero_update(args.bank, args.runtime_profile, args.output,
                             episodes=args.episodes, seed=args.seed)
    print(json.dumps({'status': result['status'], 'weights_unchanged': result['weights_unchanged'],
                      'episodes': len(result['episodes'])}), flush=True)


if __name__ == '__main__': main()
