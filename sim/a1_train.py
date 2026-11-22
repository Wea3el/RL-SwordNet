"""Bounded A1-family self-play candidate; not a validated fencing policy.

Each update consumes fresh COMPLETE episodes against independently frozen
historical opponents. Physical slots alternate; they are not the A1/A2 policy
families. This CLI never changes the scene, bank, rewards, or failure rules.
No checkpoint resumption or automatic retry is supported in this version.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import numbers
from pathlib import Path
import random

import mujoco
import numpy as np
import torch

from .a1_env import A1DuelEnv
from .a1_policy import A1FeatureEncoder, A1Policy
from .a1_ppo import ppo_batch_identity, ppo_update_once, UPDATE_VERSION
from .a1_rollout import DEFAULT_GAMMA_PER_SECOND
from .a1_runtime import load_runtime_profile
from .a1_selfplay import (FrozenSnapshotPool, collect_episode, export_transitions,
                          metadata_hash, snapshot_metadata, weights_hash)
from .assembly import ROOT
from .skill_validation import freeze_experiment, save_json


TRAINER_VERSION = 'a1_bounded_historical_selfplay_v1'
EPISODE_SECONDS = 80.


def _validate(updates, episodes_per_update, seed, learning_rate):
    if type(updates) is not int or not 1 <= updates <= 8:
        raise ValueError('updates must be an integer in 1..8')
    if type(episodes_per_update) is not int or episodes_per_update not in (2, 4, 6, 8):
        raise ValueError('episodes_per_update must be an even integer in 2..8')
    if updates*episodes_per_update > 32:
        raise ValueError('At most 32 complete episodes are allowed per invocation')
    if type(seed) is not int or not 0 <= seed <= 2**32-304:
        raise ValueError('seed must be a nonnegative integer <=2**32-304')
    if (isinstance(learning_rate, (bool, np.bool_)) or not isinstance(learning_rate, numbers.Real)
            or not np.isfinite(learning_rate) or not 0 < learning_rate <= .01):
        raise ValueError('learning_rate must be finite, positive and <=.01')


def _input_manifest(bank, profile):
    """Re-read immutable inputs; output files are deliberately excluded."""
    paths = set((ROOT/'sim').glob('*.py'))
    paths.update((ROOT/'plan1_skills').glob('*.py'))
    paths.update(ROOT.glob('*.urdf'))
    paths.update(p for p in (ROOT/'robot.xml', ROOT/'scene.xml') if p.is_file())
    for folder in ('config', 'meshes', 'assets'):
        paths.update(p for p in (ROOT/folder).rglob('*') if p.is_file())
    bank, profile = Path(bank).resolve(), Path(profile).resolve()
    if bank.is_dir():
        paths.update(p for p in bank.rglob('*') if p.is_file())
    else:
        paths.add(bank)
    paths.add(profile)
    return {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}


def _assert_inputs(expected, bank, profile):
    if _input_manifest(bank, profile) != expected:
        raise ValueError('Source, model, bank or profile changed during the run; no mixed batch is allowed')


def _plain(value):
    # Also converts TorchVersion's string subclass into a weights_only-safe str.
    return json.loads(json.dumps(value, allow_nan=False))


def _rng_state(generators, pool_generator):
    numpy = np.random.get_state()
    return {'torch_global': torch.get_rng_state(),
        'actor_generators': {s: g.get_state() for s, g in generators.items()},
        'opponent_pool_generator': pool_generator.get_state(),
        'python_random': random.getstate(),
        'numpy_global': {'algorithm': numpy[0], 'keys': numpy[1].tolist(),
                         'position': numpy[2], 'has_gauss': numpy[3], 'cached_gaussian': numpy[4]}}


def _checkpoint(path, learner, optimizer, *, updates, config, inputs, metadata,
                pool, generators, pool_generator):
    torch.save({'trainer_version': TRAINER_VERSION, 'ppo_version': UPDATE_VERSION,
        'status': 'technical_training_candidate_not_validated_fencing_policy',
        'resume_supported': False, 'completed_optimizer_updates': updates,
        'state_dict': copy.deepcopy(learner.state_dict()),
        'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
        'policy_metadata': _plain(learner.metadata()),
        'compatibility_metadata': _plain(metadata), 'weights_sha256': weights_hash(learner),
        'config': config, 'input_sha256': inputs, 'snapshot_pool': _plain(pool.metadata()),
        'rng_state': _rng_state(generators, pool_generator)}, path)


def run_training(bank, runtime_profile, output, *, updates=2, episodes_per_update=2,
                 seed=0, learning_rate=3e-4):
    """Run a deliberately small, fresh self-play training transaction.

    An invalid episode aborts its entire pending batch, preserves evidence,
    and raises. Earlier completed updates remain explicitly recorded. No
    forced replacement, seed retry, partial-episode update or resume occurs.
    """
    _validate(updates, episodes_per_update, seed, learning_rate)
    bank, runtime_profile = Path(bank).resolve(), Path(runtime_profile).resolve()
    if not bank.exists() or not runtime_profile.is_file():
        raise ValueError('Existing bank and runtime profile paths are required')
    folder = Path(output).resolve()
    if folder == bank or folder.is_relative_to(bank) or folder == runtime_profile:
        raise ValueError('Output must not be inside an immutable input bank/profile')
    folder.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1); torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    generators = {s: torch.Generator(device='cpu').manual_seed(seed+n)
                  for s, n in (('a1', 101), ('a2', 202))}
    pool_generator = torch.Generator(device='cpu').manual_seed(seed+303)
    config = {'trainer_version': TRAINER_VERSION, 'updates': updates,
        'episodes_per_update': episodes_per_update, 'maximum_episodes': updates*episodes_per_update,
        'seed': seed, 'learning_rate': float(learning_rate), 'episode_seconds': EPISODE_SECONDS,
        'cpu_threads': 1, 'snapshot_capacity': 4, 'epochs_per_update': 1,
        'gamma_per_second': DEFAULT_GAMMA_PER_SECOND, 'lambda_per_second': 1.,
        'complete_episode_only': True, 'stochastic_actions': True,
        'resume_supported': False, 'bank': str(bank), 'runtime_profile': str(runtime_profile),
        'versions': {'torch': str(torch.__version__), 'numpy': str(np.__version__),
                     'mujoco': str(mujoco.__version__)}}
    summary = {'status': 'running_candidate', 'config': config, 'updates': [], 'episodes': [],
        'completed_optimizer_updates': 0, 'learned_fencing_ability_demonstrated': False,
        'admitted_to_training': False, 'scope': 'A1 finite-skill family only; not continuous A2 or final comparison'}
    save_json(folder/'summary.json', summary)
    env = learner = optimizer = metadata = pool = inputs = None
    active_batch = active_episode = None
    try:
        inputs = _input_manifest(bank, runtime_profile)
        summary['input_sha256'] = inputs
        save_json(folder/'input_manifest.json', inputs)
        env = A1DuelEnv(bank, load_runtime_profile(runtime_profile),
                        max_duration_s=EPISODE_SECONDS, development_mode=True)
        env.reset()
        learner = A1Policy(A1FeatureEncoder.from_library(env.library)).cpu().eval()
        optimizer = torch.optim.Adam(learner.parameters(), lr=float(learning_rate))
        metadata = snapshot_metadata(env, learner)
        pool = FrozenSnapshotPool(max_size=4)
        snapshots = folder/'snapshots'; snapshots.mkdir()

        def save_snapshot():
            identity = pool.add(learner, metadata)
            path = snapshots/f'{identity}.pt'
            if not path.exists():
                torch.save({'snapshot_id': identity, 'weights_sha256': weights_hash(learner),
                    'state_dict': copy.deepcopy(learner.state_dict()), 'metadata': _plain(metadata),
                    'status': 'frozen_candidate_opponent_not_validated_fencer'}, path)
            return identity

        save_snapshot()
        _checkpoint(folder/'initial.pt', learner, optimizer, updates=0, config=config,
            inputs=inputs, metadata=metadata, pool=pool, generators=generators, pool_generator=pool_generator)
        freeze_experiment(folder/'frozen', env.library, env.scene, cfg=env.cfg,
            implementation_paths=tuple(str(p.relative_to(ROOT)) for p in
                sorted((*((ROOT/'sim').glob('*.py')), *((ROOT/'plan1_skills').glob('*.py'))))),
            metadata={'trainer_version': TRAINER_VERSION, 'compatibility_metadata': metadata,
                      'input_sha256': inputs, 'candidate_only': True})
        _assert_inputs(inputs, bank, runtime_profile)
        for update_index in range(updates):
            active_batch = folder/f'update_{update_index+1:02d}'; active_batch.mkdir()
            _assert_inputs(inputs, bank, runtime_profile)
            if snapshot_metadata(env, learner) != metadata:
                raise ValueError('Live environment or policy metadata changed before collection')
            # Captured before the FIRST episode and never relabelled afterward.
            policy_id = metadata_hash({'update_index': update_index,
                'weights_sha256': weights_hash(learner), 'compatibility_metadata': metadata})
            identity = ppo_batch_identity(learner, policy_id, metadata)
            save_json(active_batch/'batch_identity.json', identity)
            episodes = []
            for within_batch in range(episodes_per_update):
                index = update_index*episodes_per_update+within_batch
                active_episode = active_batch/f'episode_{within_batch:02d}'; active_episode.mkdir()
                _assert_inputs(inputs, bank, runtime_profile)
                opponent, slot = pool.for_episode(index, pool_generator, expected_metadata=metadata)
                print(json.dumps({'event': 'episode_start', 'update': update_index+1,
                    'episode': index, 'learner_slot': slot, 'opponent_snapshot_id': opponent.snapshot_id}), flush=True)
                episode = collect_episode(env, learner, opponent, slot, generators, policy_id)
                env.export(active_episode/'env')
                export_transitions(active_episode/'transitions.npz', episode)
                if episode['batch_identity'] != identity or episode['snapshot_metadata'] != metadata:
                    raise ValueError('Episode does not belong to the immutable pending on-policy batch')
                if snapshot_metadata(env, learner) != metadata:
                    raise ValueError('Live physical/game metadata changed during episode')
                _assert_inputs(inputs, bank, runtime_profile)
                record = {k: episode[k] for k in ('statistics', 'batch_identity', 'snapshot_metadata',
                    'zero_update_replay', 'weights_unchanged', 'collected_opponent_samples')}
                record.update(update=update_index+1, episode=index,
                    env_status=episode['env_result']['status'],
                    round_result=episode['env_result']['round_result'],
                    completed_time_s=episode['env_result']['completed_time_s'],
                    decision_rows=len(episode['transitions']))
                save_json(active_episode/'collection.json', record)
                summary['episodes'].append(record)
                save_json(folder/'summary.json', summary)
                episodes.append(episode['transitions'])
                print(json.dumps({'event': 'episode_end', 'episode': index,
                    'decision_rows': record['decision_rows'], 'round_result': record['round_result']}), flush=True)
            _assert_inputs(inputs, bank, runtime_profile)
            if snapshot_metadata(env, learner) != metadata:
                raise ValueError('Live environment changed before optimizer update')
            report = ppo_update_once(learner, episodes, expected_policy_id=policy_id,
                batch_identity=identity, expected_compatibility_metadata=metadata,
                optimizer=optimizer, learning_rate=float(learning_rate))
            summary['completed_optimizer_updates'] += 1
            report['update_index'] = update_index+1
            summary['updates'].append(report)
            save_json(active_batch/'optimizer_report.json', report)
            _assert_inputs(inputs, bank, runtime_profile)
            report['new_snapshot_id'] = save_snapshot()
            save_json(active_batch/'optimizer_report.json', report)
            save_json(active_batch/'snapshot_pool.json', pool.metadata())
            _checkpoint(active_batch/'candidate.pt', learner, optimizer,
                updates=summary['completed_optimizer_updates'], config=config, inputs=inputs,
                metadata=metadata, pool=pool, generators=generators, pool_generator=pool_generator)
            save_json(folder/'summary.json', summary)
            print(json.dumps({'event': 'update_end', 'update': update_index+1,
                'optimizer_steps': report['optimizer_step_count_after'],
                'weights_changed': report['weights_changed']}), flush=True)
        _checkpoint(folder/'final_candidate.pt', learner, optimizer,
            updates=summary['completed_optimizer_updates'], config=config, inputs=inputs,
            metadata=metadata, pool=pool, generators=generators, pool_generator=pool_generator)
        summary.update(status='bounded_candidate_training_completed', snapshot_pool=pool.metadata(),
            final_weights_sha256=weights_hash(learner), immutable_inputs_verified=True,
            final_checkpoint='final_candidate.pt')
        save_json(folder/'summary.json', summary)
        return summary
    except BaseException as error:
        export_error = None
        if env is not None and getattr(env, '_reset_done', False):
            try: env.export(folder/'invalid_or_interrupted_env')
            except Exception as exc: export_error = {'type': type(exc).__name__, 'message': str(exc)}
        failure = {'type': type(error).__name__, 'message': str(error),
            'details': getattr(error, 'details', None),
            'collection_invalid': getattr(error, 'collection_invalid', None),
            'active_batch': str(active_batch) if active_batch else None,
            'active_episode': str(active_episode) if active_episode else None,
            'evidence_export_error': export_error, 'pending_batch_rejected': True,
            'no_retry_or_substitute_draw': True,
            'completed_optimizer_updates': summary['completed_optimizer_updates']}
        summary.update(status='aborted_invalid_candidate', failure=failure)
        save_json(folder/'failure.json', failure); save_json(folder/'summary.json', summary)
        if all(x is not None for x in (learner, optimizer, inputs, metadata, pool)):
            try:
                _checkpoint(folder/'aborted_candidate.pt', learner, optimizer,
                    updates=summary['completed_optimizer_updates'], config=config, inputs=inputs,
                    metadata=metadata, pool=pool, generators=generators, pool_generator=pool_generator)
            except Exception as checkpoint_error:
                failure['checkpoint_export_error'] = {'type': type(checkpoint_error).__name__,
                                                       'message': str(checkpoint_error)}
                save_json(folder/'failure.json', failure); save_json(folder/'summary.json', summary)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank', required=True)
    parser.add_argument('--runtime-profile', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--updates', type=int, default=2)
    parser.add_argument('--episodes-per-update', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    args = parser.parse_args()
    result = run_training(args.bank, args.runtime_profile, args.output, updates=args.updates,
        episodes_per_update=args.episodes_per_update, seed=args.seed, learning_rate=args.learning_rate)
    print(json.dumps({'status': result['status'],
                      'completed_optimizer_updates': result['completed_optimizer_updates']}), flush=True)


if __name__ == '__main__': main()
