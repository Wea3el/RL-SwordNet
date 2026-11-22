"""Current-observation A1 reaction diagnostics, never self-play or admission.

The defender reads current opponent tip/velocity, its own executor state and
its own observation history. It cannot access the opponent's requested action,
controller, script or future clock. Initial common-Guard rest is assumed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from .a1_control import A1Controller, A1ScriptedPolicy, A1SequencePolicy
from .a1_skills import export_a1_library, load_a1_library
from .a1_validation import defence_gate, execution_gate
from .assembly import ROOT
from .duel_scene import build_duel_scene
from .skill_validation import V2_CONFIG, freeze_experiment, run_trial, save_json


class ObservedMotionParry:
    """One fixed parry per observed departure-and-return episode.

    A threshold is a diagnostic setting, not measured perception accuracy.
    Absolute game time never triggers a skill. Time only measures how long
    an actually observed return has remained quiet before re-arming.
    """
    def __init__(self, displacement_m=.001, minimum_speed_m_s=1e-5):
        values = (displacement_m, minimum_speed_m_s)
        if (any(isinstance(v, (bool, str)) for v in values)
                or not np.isfinite(values).all() or min(values) <= 0):
            raise ValueError("Positive finite motion thresholds required")
        self.displacement_m = float(displacement_m)
        self.minimum_speed_m_s = float(minimum_speed_m_s)
        self.initial_tip, self.quiet_since = None, None
        self.armed = True
        self.events = []
        self._last_time = -np.inf

    def decide(self, obs):
        now = obs['time_s']
        tip = np.asarray(obs['opponent_tip_minus_own_target_local_m'], dtype=float)
        velocity = np.asarray(obs['opponent_tip_velocity_local_m_s'], dtype=float)
        if (isinstance(now, (bool, str)) or not np.isscalar(now) or not np.isfinite(now)
                or now < 0 or now < self._last_time-1e-9 or tip.shape != (3,)
                or velocity.shape != (3,) or not np.isfinite(tip).all() or not np.isfinite(velocity).all()):
            raise ValueError("Finite current vectors and monotonic own observation time required")
        now = float(now)
        if now-self._last_time > .021+1e-9:
            # Missing observations cannot establish a quiet return interval.
            self.quiet_since = None
        self._last_time = max(self._last_time, now)
        speed = float(np.linalg.norm(velocity))
        if self.initial_tip is None:
            self.initial_tip = tip.copy()
        displacement = float(np.linalg.norm(tip-self.initial_tip))
        if displacement <= min(.002, self.displacement_m/2) and speed < 1e-5:
            if self.quiet_since is None:
                self.quiet_since = now
            if now-self.quiet_since >= .10-1e-9:
                self.armed = True
        else:
            self.quiet_since = None
        own = obs['own_skill']
        if (self.armed and not own['busy'] and own['action_mask'][4]
                and displacement >= self.displacement_m and speed >= self.minimum_speed_m_s):
            self.armed = False
            self.events.append(dict(time_s=now, action_id=4,
                reason='current observed departure from initial rest',
                tip_displacement_m=displacement, tip_speed_m_s=speed,
                tip_minus_own_target_local_m=tip.tolist()))
            return 4
        return 0

    def diagnostics(self):
        return dict(events=copy.deepcopy(self.events),
            displacement_threshold_m=self.displacement_m,
            minimum_speed_m_s=self.minimum_speed_m_s,
            maximum_observation_gap_for_quiet_dwell_s=.021,
            opponent_action_phase_clock_or_program_inputs=False,
            scope='observation-only diagnostic, not learned policy or perception certification')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_reaction(bank, output, *, attacker='a1', attack_start_s=.75,
                      displacement_m=.001, repeat_count=2, physics_timestep_s=.002):
    """Fresh matched repeated-attack baseline and observation-only defence.

    Fixed expected action counts are used, not the number that the policy
    happened to request. Failure, return/settle and post-first-touch entries
    remain part of evidence. No result automatically authorizes training.
    """
    if (attacker not in {'a1', 'a2'} or isinstance(attack_start_s, (bool, str))
            or not np.isfinite(attack_start_s) or attack_start_s < 0
            or type(repeat_count) is not int or not 1 <= repeat_count <= 3
            or isinstance(physics_timestep_s, bool) or physics_timestep_s not in {.001, .002}):
        raise ValueError("Use either slot, finite attack start, 1..3 repeats and 1/2ms physics")
    reactive = ObservedMotionParry(displacement_m)
    bank, output = Path(bank).resolve(), Path(output).resolve()
    source_hashes = {str(p): _sha(p) for p in bank.iterdir() if p.is_file()}
    library = load_a1_library(bank)
    cfg = dict(V2_CONFIG, physics_timestep_s=physics_timestep_s, decision_period_s=.02)
    scene = build_duel_scene(cfg['base_distance_m'])
    # Conservative whole-window duration includes worst-case own settling.
    duration = max(70., float(np.ceil(attack_start_s+repeat_count*(library.skills['stab'].duration_s
        +A1Controller(library).settle_timeout_s+.12)+cfg['final_hold_s'])))
    modules = ('sim/a1_skills.py', 'sim/a1_control.py', 'sim/a1_validation.py',
        'sim/a1_defence.py', 'sim/a1_adaptation.py', 'sim/a1_reaction_validation.py',
        'sim/assembly.py', 'sim/avatar.py', 'sim/floor.py', 'sim/shinai.py',
        'sim/model.py', 'config/assembly.json')
    freeze_experiment(output, library, scene, cfg=cfg, implementation_paths=modules,
        metadata={'scope': 'restricted Stab/Right current-observation diagnostic, not final A1/A2',
            'attacker_slot': attacker, 'attack_start_s': attack_start_s,
            'repeat_count': repeat_count, 'threshold_m': displacement_m,
            'no_opponent_future_information': True, 'initial_common_guard_rest_assumed': True})
    frozen = json.loads((output/'experiment.json').read_text())['implementation_sha256']
    export_a1_library(library, output/'library')
    defender = 'a2' if attacker == 'a1' else 'a1'
    summary = dict(status='running', runtime_config=cfg, trials={},
        admitted_to_training=False, may_start_RL=False, source_bank_sha256=source_hashes,
        scope='physical observation-only reaction validation, NOT self-play')
    save_json(output/'summary.json', summary)
    runs = []
    for name, defending in (('guard_baseline', False), ('observed_defence', True)):
        policies = {attacker: A1SequencePolicy(library, [1]*repeat_count, attack_start_s, .1),
                    defender: reactive if defending else A1ScriptedPolicy(library, [])}
        result = run_trial(scene, library, policies, duration, cfg=cfg,
            output=output/name, controller_factory=A1Controller)
        expected = {attacker: [1]*repeat_count, defender: [4]*repeat_count if defending else []}
        result['a1_execution_gate'] = execution_gate(result, expected)
        result['reaction_diagnostics'] = reactive.diagnostics() if defending else None
        result['limitations'] = 'Current-observation diagnostic; no learned policy, sensor noise, A2 or hardware certification'
        save_json(output/name/'summary.json', result)
        summary['trials'][name] = {k:result[k] for k in ('status', 'failure', 'hits',
            'action_events', 'completed_time_s', 'a1_execution_gate', 'reaction_diagnostics')}
        save_json(output/'summary.json', summary)
        runs.append(result)
    summary['paired_defence'] = defence_gate(*runs, attacker=attacker, cfg=cfg)
    summary['source_bank_unchanged'] = all(_sha(p) == h for p,h in source_hashes.items())
    summary['implementations_unchanged'] = all(_sha(ROOT/p) == h for p,h in frozen.items())
    summary['status'] = ('diagnostic_complete_not_admitted' if summary['source_bank_unchanged']
        and summary['implementations_unchanged'] else 'source_changed_during_run')
    save_json(output/'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--attacker', choices=('a1','a2'), default='a1')
    parser.add_argument('--attack-start-s', type=float, default=.75)
    parser.add_argument('--threshold-mm', type=float, default=1.)
    parser.add_argument('--repeat-count', type=int, default=2)
    parser.add_argument('--physics-timestep-s', type=float, choices=(.001,.002), default=.002)
    args = parser.parse_args()
    result = validate_reaction(args.bank, args.output, attacker=args.attacker,
        attack_start_s=args.attack_start_s, displacement_m=args.threshold_mm/1000,
        repeat_count=args.repeat_count, physics_timestep_s=args.physics_timestep_s)
    print(json.dumps(result['paired_defence'], indent=2))
    print(f"{args.output.resolve()}/summary.json; may_start_RL=False")


if __name__ == '__main__':
    main()
