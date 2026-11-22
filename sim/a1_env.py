"""Minimal simultaneous two-slot A1 development environment, not a trainer.

Both physical slots currently use the SAME finite A1 executor. This is not
the continuous A2 action adapter, Gym/PPO, self-play, or skill-bank admission.
The explicit runtime profile and training contact rule are opt-in here only;
historical runners, scoring evidence and numerical defaults are untouched.
"""
from __future__ import annotations

import copy
import hashlib
import json
import numbers
from pathlib import Path

import mujoco
import numpy as np

from .a1_control import A1Controller
from .a1_runtime import RuntimeProfile, apply_runtime_profile, assert_runtime_profile
from .a1_skills import load_a1_library
from .duel_scene import build_duel_scene, duel_contact_records
from .fencing_rules import FirstTouchScorer, blade_sections_from_scene
from .manual_duel import position_control
from .skill_validation import observation, save_json
from .training_contacts import TrainingContactMonitor, settle_training_tick, TrainingEnvironmentError


SLOTS = ('a1', 'a2')


class A1DuelEnv:
    """Synchronous20ms joint-action API over native RK4/1ms physics.

    ``reset()`` returns ``(observations, info)``. ``step({a1:id,a2:id})``
    returns ``(observations, rewards, terminated, truncated, info)``.
    Busy/unavailable slots execute0 (continue), never queue another skill.
    ``info.choice_enabled`` distinguishes these ticks from actual choices;
    consumers must use actual ``elapsed_s`` for time-based discounting.

    A candidate bank is accepted ONLY with explicit development_mode=True.
    No public formal-admission path or continuous A2 adapter exists yet.
    """
    def __init__(self, library, runtime_profile, *, max_duration_s=80.,
                 development_mode=False):
        if development_mode is not True:
            raise ValueError('Candidate banks require explicit development_mode=True; formal admission is not implemented')
        if not isinstance(runtime_profile, RuntimeProfile):
            raise ValueError('An explicitly loaded RuntimeProfile is required')
        cfg = runtime_profile.runtime_config()
        if (cfg['explicit_integrator'] != 'RK4' or cfg['physics_timestep_s'] != .001
                or cfg['control_period_s'] != .01 or cfg['decision_period_s'] != .02):
            raise ValueError('This initial A1 environment requires explicit RK4/1ms,10ms control,20ms decision')
        if (isinstance(max_duration_s, (bool, str)) or not np.isscalar(max_duration_s)
                or not np.isfinite(max_duration_s) or max_duration_s <= 0):
            raise ValueError('A finite positive physical episode duration is required')
        steps = float(max_duration_s)/cfg['physics_timestep_s']
        if not np.isclose(steps, round(steps), rtol=0., atol=1e-8):
            raise ValueError('Episode duration must be an integer number of native physics steps')
        self.library = load_a1_library(library) if isinstance(library, (str, Path)) else library
        if not self.library.reports['library']['all_skills_pass_geometry']:
            raise ValueError('Candidate geometry preflight must pass before development execution')
        # Own the command arrays; a caller changing its source library cannot
        # silently change an episode that was already constructed.
        self.library = copy.copy(self.library)
        self.library.guard_q = np.asarray(self.library.guard_q).copy()
        self.library.skills = copy.deepcopy(self.library.skills)
        self.profile, self.cfg = runtime_profile, cfg
        self.max_steps, self.max_duration_s = int(round(steps)), float(max_duration_s)
        self.control_steps = round(cfg['control_period_s']/cfg['physics_timestep_s'])
        self.decision_steps = round(cfg['decision_period_s']/cfg['physics_timestep_s'])
        self.scene = build_duel_scene(cfg['base_distance_m'])
        apply_runtime_profile(self.scene, self.profile)
        self._reset_done, self._done, self._error = False, False, None

    def _feedback(self):
        now, data = float(self.scene.data.time), self.scene.data
        for slot in SLOTS:
            mapping = self.scene.agents[slot]
            self.controllers[slot].observe_own_state(data.qpos[mapping['qpos_ids']].copy(),
                data.qvel[mapping['dof_ids']].copy(), now)

    def _observations(self):
        # Snapshot BOTH slots before any subsequent request. Opponent skill,
        # phase, action IDs and future clocks never enter this shared schema.
        return {slot: copy.deepcopy(observation(self.scene, slot, self.controllers,
                    float(self.scene.data.time))) for slot in SLOTS}

    def _update_control(self):
        now = float(self.scene.data.time)
        self.commands = {slot: self.controllers[slot].target(now) for slot in SLOTS}
        for slot in SLOTS:
            self.command_hashes[slot].update(np.asarray(self.commands[slot], dtype='<f8').tobytes())
        position_control(self.scene, self.gravity, self.commands)
        mujoco.mj_forward(self.scene.model, self.scene.data)

    def _faults(self, contacts=()):
        model, data = self.scene.model, self.scene.data
        faults = []
        if not np.isfinite(data.time):
            faults.append('nonfinite_native_time')
        if not all(np.isfinite(x).all() for x in (data.qpos, data.qvel, data.qacc,
                                                 data.ctrl, data.actuator_force)):
            faults.append('nonfinite_native_state')
        if any(w.number for w in data.warning):
            faults.append('mujoco_warning')
        if np.any(np.abs(data.actuator_force) > self.caps+1e-6):
            faults.append('native_actuator_cap_exceeded')
        if np.any(model.body_gravcomp) or np.any(data.qfrc_applied) or np.any(data.xfrc_applied):
            faults.append('unsupported_external_force_or_gravity_compensation')
        penetration = max((max(0., -c['distance_m']) for c in contacts), default=0.)
        self.maximum_penetration_m = max(self.maximum_penetration_m, penetration)
        if penetration > self.cfg['penetration_limit_m']:
            faults.append('excessive_proxy_penetration')
        limit_ids = np.flatnonzero(data.efc_type[:data.nefc] == int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT))
        limit_force = float(np.max(np.abs(data.efc_force[limit_ids]), initial=0.))
        self.peak_limit_force_nm = max(self.peak_limit_force_nm, limit_force)
        if limit_force > self.cfg['hypothetical_limit_force_ceiling_nm']:
            faults.append('unidentified_joint_limit_assistance')
        if np.min(np.minimum(data.qpos-model.jnt_range[:, 0], model.jnt_range[:, 1]-data.qpos)) < -.005:
            faults.append('joint_left_hypothetical_range')
        for slot in SLOTS:
            if self.controllers[slot].failure:
                faults.append(f'controller:{slot}:{self.controllers[slot].failure}')
        if self.scorer.failure:
            faults.append(f'scorer:{self.scorer.failure}')
        return faults

    def _freeze_evidence(self):
        arrays = {'guard_q': self.library.guard_q.copy()}
        skills = {}
        for name, path in self.library.skills.items():
            skills[name] = {'action_id': path.action_id, 'name': path.name,
                'sampler_class': f'{type(path).__module__}.{type(path).__qualname__}',
                'segments': copy.deepcopy(path.segments),
                'segment_labels': copy.deepcopy(path.segment_labels)}
            for field in ('qpos', 'time_s', 'knot_qpos', 'knot_time_s', 'coefficients'):
                if hasattr(path, field):
                    arrays[f'{name}__{field}'] = np.asarray(getattr(path, field)).copy()
        metadata = {'config': copy.deepcopy(self.library.config), 'skills': skills,
            'reports': copy.deepcopy(self.library.reports),
            'provenance': copy.deepcopy(getattr(self.library, 'provenance', {}))}
        digest = hashlib.sha256(json.dumps(metadata, sort_keys=True, allow_nan=False).encode())
        for name, values in sorted(arrays.items()):
            digest.update(json.dumps([name, values.dtype.str, values.shape]).encode())
            digest.update(np.ascontiguousarray(values).tobytes())
        self._bank_arrays, self._bank_metadata = arrays, metadata
        root = Path(__file__).resolve().parents[1]
        files = ('sim/a1_env.py', 'sim/training_contacts.py', 'sim/foul_attribution.py',
            'sim/a1_runtime.py', 'sim/a1_control.py', 'sim/a1_skills.py', 'sim/shared_skills.py',
            'sim/skill_validation.py', 'sim/manual_duel.py', 'sim/fencing_rules.py', 'sim/duel_scene.py')
        self._evidence = {'bank_numeric_and_metadata_sha256': digest.hexdigest(),
            'implementation_sha256': {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files},
            'profile_source_sha256': self.profile.source_sha256,
            'software_versions': {'mujoco': mujoco.__version__, 'numpy': np.__version__},
            'numeric_snapshot_includes_exact_c2_knots_and_coefficients': True}

    def _settle(self, *, verdicts=(), hits=(), faults=()):
        try:
            return settle_training_tick(self.physics_tick, contacts=verdicts,
                hit_agents=hits, numerical_faults=faults)
        except TrainingEnvironmentError as exc:
            self._error = {'type': type(exc).__name__, 'message': str(exc),
                           'details': copy.deepcopy(getattr(exc, 'details', None))}
            self._done = True
            raise

    def _record_trace(self):
        data = self.scene.data
        target = np.empty(self.scene.model.nq)
        for slot in SLOTS:
            target[self.scene.agents[slot]['qpos_ids']] = self.commands[slot]
        row = dict(time_s=float(data.time), qpos=data.qpos.copy(), qvel=data.qvel.copy(),
                   target_qpos=target, ctrl=data.ctrl.copy(), actuator_force_nm=data.actuator_force.copy())
        for key, value in row.items():
            self._trace[key].append(value)

    def reset(self):
        """Initialize Guard only; no randomized pose or mid-episode resets."""
        model, data = self.scene.model, self.scene.data
        mujoco.mj_resetData(model, data)
        apply_runtime_profile(self.scene, self.profile)
        for mapping in self.scene.agents.values():
            data.qpos[mapping['qpos_ids']] = self.library.guard_q
        mujoco.mj_forward(model, data)
        self.controllers = {slot: A1Controller(self.library) for slot in SLOTS}
        if self.controllers['a1'] is self.controllers['a2']:
            raise ValueError('Independent controllers required')
        self.gravity = mujoco.MjData(model)
        self.commands = {slot: self.library.guard_q.copy() for slot in SLOTS}
        self.command_hashes = {slot: hashlib.sha256() for slot in SLOTS}
        self.caps = np.minimum(-model.actuator_forcerange[:, 0], model.actuator_forcerange[:, 1]).copy()
        centers = {slot: self.scene.config['virtual_targets'][SLOTS[1-i]]['center_world_m']
                   for i, slot in enumerate(SLOTS)}
        radii = {slot: self.scene.config['virtual_targets'][SLOTS[1-i]]['radius_m']
                 for i, slot in enumerate(SLOTS)}
        if radii['a1'] != radii['a2']:
            raise ValueError('Equal target radii required')
        self.scorer = FirstTouchScorer(centers, radii)
        self.physics_tick, self.decisions, self.native_events = 0, [], []
        self.maximum_penetration_m, self.peak_limit_force_nm = 0., 0.
        self._error, self._done, self._terminated, self._truncated = None, False, False, False
        self._terminal_settlement, self._timeout_settlement, self._last_step_info = None, None, None
        self._trace = {key: [] for key in ('time_s', 'qpos', 'qvel', 'target_qpos', 'ctrl', 'actuator_force_nm')}
        self._feedback()
        self._freeze_evidence()
        # Initial ctrl is applied before resetting the monitor's provenance.
        position_control(self.scene, self.gravity, self.commands)
        mujoco.mj_forward(model, data)
        self.previous_blades = blade_sections_from_scene(self.scene)
        self.previous_time = float(data.time)
        self.scorer.update(self.previous_blades, self.previous_blades, 0., 0.)
        self.monitor = TrainingContactMonitor(self.scene, self.cfg)
        self.monitor.reset(self.scene)
        contacts = duel_contact_records(self.scene, rules_version=self.cfg['rules_version'])
        faults = self._faults(contacts)
        if self.scorer.initial_overlap_events:
            faults.append('initial_blade_target_overlap')
        if any(c['category'].startswith('illegal_') and c['normal_force_n'] > self.cfg['illegal_contact_force_threshold_n'] for c in contacts):
            faults.append('invalid_initial_forbidden_contact_without_preimpact_history')
        self._settle(faults=faults)
        self.initial_qpos = data.qpos.copy()
        self._reset_done = True
        self._record_trace()
        obs = self._observations()
        return obs, {'time_s': 0., 'elapsed_s': 0., 'development_mode': True,
            'policy_families': {slot: 'A1' for slot in SLOTS},
            'choice_enabled': {slot: bool(any(obs[slot]['own_skill']['action_mask'][1:])) for slot in SLOTS},
            'admitted_to_training': False, 'runtime_config': copy.deepcopy(self.cfg)}

    def step(self, actions):
        try:
            return self._step(actions)
        except TrainingEnvironmentError as exc:
            # Collector provenance failures can occur before the pure settler.
            # They poison the transition too, never allowing a retry/reward.
            self._error = {'type': type(exc).__name__, 'message': str(exc),
                           'details': copy.deepcopy(exc.details)}
            self._done = True
            if self._reset_done:
                self._record_trace()
            raise

    def _step(self, actions):
        if not self._reset_done:
            raise RuntimeError('reset() is required before step()')
        if self._done:
            raise RuntimeError('Episode has ended or failed; reset before stepping')
        try:
            assert_runtime_profile(self.scene, self.profile)
        except ValueError as exc:
            self._settle(faults=[f'numerical_profile_changed:{exc}'])
        if not isinstance(actions, dict) or set(actions) != set(SLOTS):
            raise ValueError('Both physical slots must submit one joint-action dictionary')
        # Validate the ENTIRE dictionary before touching either controller.
        for slot in SLOTS:
            action = actions[slot]
            if (isinstance(action, (bool, np.bool_)) or not isinstance(action, numbers.Integral)
                    or int(action) not in self.controllers[slot].action_names):
                raise ValueError(f'Integer skill ID from the current A1 bank required for {slot}')
        faults = self._faults()
        if faults:
            self._settle(faults=faults)
        before = self._observations()
        choice_enabled = {slot: bool(not before[slot]['own_skill']['busy']
            and any(before[slot]['own_skill']['action_mask'][1:])) for slot in SLOTS}
        submitted = {slot: int(actions[slot]) for slot in SLOTS}
        effective = {slot: submitted[slot] if choice_enabled[slot] else 0 for slot in SLOTS}
        now = float(self.scene.data.time)
        accepted = {}
        controller_accepted = {}
        for slot in SLOTS:
            controller_accepted[slot] = bool(self.controllers[slot].request(effective[slot], now))
            accepted[slot] = bool(controller_accepted[slot] and submitted[slot] == effective[slot])
        decision = dict(time_s=now, submitted_actions=submitted, effective_actions=effective,
            choice_enabled=choice_enabled, accepted=accepted, controller_request_accepted=controller_accepted,
            new_skill_accepted={slot: bool(effective[slot] and accepted[slot]) for slot in SLOTS})
        self.decisions.append(copy.deepcopy(decision))
        start, start_tick = now, self.physics_tick
        rewards = {slot: 0. for slot in SLOTS}
        final_settlement = None
        while self.physics_tick < min(start_tick+self.decision_steps, self.max_steps):
            if self.physics_tick % self.control_steps == 0:
                self._update_control()
            faults = self._faults()
            if faults:
                self._settle(faults=faults)
            self.monitor.capture_before_step(self.scene)
            mujoco.mj_step(self.scene.model, self.scene.data)
            mujoco.mj_forward(self.scene.model, self.scene.data)
            self.physics_tick += 1
            faults = self._faults()
            if faults:
                self._settle(faults=faults)
            self._feedback()
            current_time = float(self.scene.data.time)
            current_blades = blade_sections_from_scene(self.scene)
            hits = self.scorer.update(self.previous_blades, current_blades,
                                      self.previous_time, current_time)
            report = self.monitor.inspect(self.scene, self.physics_tick)
            faults = list(report['numerical_faults'])+self._faults(report['contacts'])
            final_settlement = self._settle(verdicts=report['verdicts'],
                hits=[hit['agent'] for hit in hits], faults=faults)
            if report['verdicts'] or hits:
                self.native_events.append({'time_s': current_time, 'physics_tick': self.physics_tick,
                    'contact_report': copy.deepcopy(report), 'hits': copy.deepcopy(hits),
                    'settlement': copy.deepcopy(final_settlement)})
            self.previous_blades, self.previous_time = current_blades, current_time
            if final_settlement['terminal']:
                self._terminal_settlement = copy.deepcopy(final_settlement)
                self._done = self._terminated = True
                rewards = copy.deepcopy(final_settlement['rl_reward'])
                break
        if not self._done and self.physics_tick >= self.max_steps:
            self._done = self._truncated = True
            self.scorer.finish(float(self.scene.data.time))
            self._timeout_settlement = dict(copy.deepcopy(final_settlement),
                status='timeout_draw', terminal=False, truncated=True,
                time_s=float(self.scene.data.time), match_winner=None,
                rl_reward={slot: 0. for slot in SLOTS})
            final_settlement = self._timeout_settlement
        self._record_trace()
        info = dict(decision, time_s=float(self.scene.data.time),
            elapsed_s=float(self.scene.data.time)-start,
            physics_steps=self.physics_tick-start_tick, physics_tick=self.physics_tick,
            settlement=copy.deepcopy(final_settlement), terminal_reason=(
                self._terminal_settlement['status'] if self._terminated else 'physical_time_limit' if self._truncated else None),
            admitted_to_training=False)
        self._last_step_info = copy.deepcopy(info)
        return self._observations(), rewards, self._terminated, self._truncated, info

    def trace(self):
        if not self._reset_done:
            raise RuntimeError('No reset episode to export')
        return {key: np.asarray(values).copy() for key, values in self._trace.items()}

    def result(self):
        if not self._reset_done:
            raise RuntimeError('No reset episode to report')
        return {'status': 'invalid' if self._error else 'terminated' if self._terminated else 'truncated' if self._truncated else 'running',
            'failure': copy.deepcopy(self._error), 'requested_duration_s': self.max_duration_s,
            'completed_time_s': float(self.scene.data.time), 'physics_tick': self.physics_tick,
            'runtime_config': copy.deepcopy(self.cfg), 'reproducibility': copy.deepcopy(self._evidence),
            'initial_qpos': self.initial_qpos.tolist(),
            'qpos_resets_after_initialization': 0, 'policy_families': {slot: 'A1' for slot in SLOTS},
            'action_events': {slot: copy.deepcopy(self.controllers[slot].events) for slot in SLOTS},
            'controller_diagnostics': {slot: copy.deepcopy(self.controllers[slot].diagnostics()) for slot in SLOTS},
            'command_sha256': {slot: h.hexdigest() for slot, h in self.command_hashes.items()},
            'hits': copy.deepcopy(self.scorer.all_entries), 'geometry_first_touch_diagnostic': self.scorer.round_result,
            'round_result': copy.deepcopy(self._terminal_settlement or self._timeout_settlement),
            'decisions': copy.deepcopy(self.decisions),
            'native_events': copy.deepcopy(self.native_events), 'contact_monitor': self.monitor.summary(),
            'maximum_contact_penetration_m': self.maximum_penetration_m,
            'peak_hypothetical_limit_constraint_force_nm': self.peak_limit_force_nm,
            'unchanged_force_caps_nm': self.caps.tolist(), 'source_hashes': copy.deepcopy(self.scene.source_hashes),
            'development_mode': True, 'admitted_to_training': False, 'selfplay_or_training_run': False,
            'scope': 'Minimal A1-only simultaneous stepping; no A2 continuous adapter, learner or automatic bank admission'}

    def export(self, output):
        """Save a new result/trace directory; never overwrite prior evidence."""
        result, trace = self.result(), self.trace()
        folder = Path(output)
        folder.mkdir(parents=True, exist_ok=False)
        save_json(folder/'summary.json', result)
        np.savez_compressed(folder/'trajectory.npz', **trace)
        save_json(folder/'bank_snapshot.json', self._bank_metadata)
        np.savez_compressed(folder/'bank_snapshot.npz', **self._bank_arrays)
        (folder/'runtime_profile.json').write_bytes(self.profile.source_bytes)
        (folder/'scene.xml').write_text(self.scene.spec.to_xml())
        return folder
