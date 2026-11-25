"""Candidate-only named-edge A1 adapter, not a trainer or admission gate.

Every explicitly ordered edge gets its own policy ID. Family is descriptive:
two right parries at the same guard remain two choices, never an automatic
selection using an opponent label. Only listed edges (including transfers)
are callable; no bridge, reset, interrupt, or legacy-actor conversion exists.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import numbers
from pathlib import Path

import numpy as np

from .assembly import ROOT
from .feedback_policy import GraphAction, SkillGraphExecutor
from .joint_constraints import DEFAULT_PATH
from .motion_spline import C2JointSpline


ADAPTER_SCHEMA = 'a1_named_edge_candidate_v1'
OWN_STATE_SCHEMA = 'a1_named_edge_own_execution_v1'
IMPLEMENTATIONS = ('sim/a1_variant_bank.py', 'sim/feedback_policy.py',
                   'sim/motion_spline.py', 'sim/joint_constraints.py')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key: '+key)
        result[key] = value
    return result


def _finite_vector(value, size, label):
    if (not isinstance(value, list) or len(value) != size or
        any(isinstance(x, bool) or not isinstance(x, numbers.Real) or
            not math.isfinite(x) for x in value)):
        raise ValueError(label+' requires finite numeric coordinates')
    return np.asarray(value, dtype=float)


def _parse(raw, policy_order, initial_guard):
    report = json.loads(raw, object_pairs_hook=_unique)
    if not isinstance(report, dict) or report.get('admitted_to_training') is not False:
        raise ValueError('An explicitly non-admitted candidate report is required')
    if (not isinstance(policy_order, tuple) or not policy_order or
        any(not isinstance(x, str) or not x.strip() or x == 'wait' for x in policy_order) or
        len(set(policy_order)) != len(policy_order)):
        raise ValueError('Explicit unique tuple of edge names, excluding reserved wait, required')
    source_guards, source_actions = report.get('guards'), report.get('actions')
    if not isinstance(source_guards, dict) or not source_guards or not isinstance(source_actions, dict):
        raise ValueError('Explicit guard and action dictionaries required')
    nodes = {}
    for name, row in source_guards.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(row, dict):
            raise ValueError('Explicit named guards required')
        nodes[name] = _finite_vector(row.get('qpos_rad'), 7, 'Guard '+name)
    if initial_guard not in nodes:
        raise ValueError('An existing explicit initial guard is required')
    actions, descriptions = {}, []
    for name in policy_order:
        row = source_actions.get(name)
        if (not isinstance(row, dict) or row.get('candidate_pass') is not True or
            row.get('admitted_to_training') is True):
            raise ValueError('Selected edge must be a passed candidate, not an admission: '+name)
        if row.get('candidate_stage') == 'authorized_native_research_hypothesis':
            raise ValueError('Native research permission is not a selectable library skill: '+name)
        kind, family = row.get('kind'), row.get('family')
        if kind not in ('attack', 'defence', 'transition') or not isinstance(family, str) or not family.strip():
            raise ValueError('Selected edge needs explicit kind and descriptive family: '+name)
        start, end = row.get('start_guard'), row.get('end_guard')
        if start not in nodes or end not in nodes:
            raise ValueError('Unknown exact entry/exit guard: '+name)
        times, qpos = row.get('knot_time_s'), row.get('knot_qpos')
        if not isinstance(times, list) or len(times) < 2 or not isinstance(qpos, list) or len(qpos) != len(times):
            raise ValueError('Matching explicit time and joint knots required: '+name)
        t = _finite_vector(times, len(times), 'Times '+name)
        q = np.asarray([_finite_vector(v, 7, 'Knot '+name) for v in qpos])
        curve = C2JointSpline(t, q)
        actions[name] = GraphAction(name, start, end, curve)
        descriptions.append((name, kind, family, start, end, curve.duration_s))
    # Reuses strict exact endpoints, confirmed whole-curve ranges, and measured
    # settling semantics. No unselected report edge enters this executor.
    executor = SkillGraphExecutor(nodes, actions, initial_guard)
    return executor, tuple(descriptions), tuple(sorted(set(source_actions)-set(policy_order)))


@dataclass(frozen=True)
class NamedEdgeBank:
    source_path: Path
    source_bytes: bytes
    policy_order: tuple[str, ...]
    initial_guard: str
    joint_constraints_sha256: str
    implementation_sha256: tuple
    actions: tuple
    unselected_actions: tuple

    @property
    def action_names(self):
        return ('wait',)+self.policy_order

    def identity(self):
        return {'adapter_schema': ADAPTER_SCHEMA, 'own_state_schema': OWN_STATE_SCHEMA,
                'bank_sha256': hashlib.sha256(self.source_bytes).hexdigest(),
                'action_names': list(self.action_names), 'initial_guard': self.initial_guard,
                'joint_constraints_sha256': self.joint_constraints_sha256,
                'implementation_sha256': dict(self.implementation_sha256)}

    def actor_signature(self):
        identity = self.identity()
        return {'adapter_schema': ADAPTER_SCHEMA, 'own_state_schema': OWN_STATE_SCHEMA,
                'snapshot_signature': _canonical_hash(identity),
                'policy_output_size': len(self.action_names),
                'action_names': list(self.action_names)}

    def assert_identity(self):
        if self.source_path.read_bytes() != self.source_bytes:
            raise ValueError('Source bank changed; reload and revalidate')
        if _sha(DEFAULT_PATH) != self.joint_constraints_sha256:
            raise ValueError('Confirmed joint constraints changed')
        for relative, expected in self.implementation_sha256:
            if _sha(ROOT/relative) != expected:
                raise ValueError('Adapter/executor implementation changed: '+relative)

    def assert_actor_compatible(self, signature, *, output_size):
        """Reject old 5/10-output metadata, reordered IDs and stale snapshots."""
        self.assert_identity()
        if (type(output_size) is not int or output_size != len(self.action_names) or
            signature != self.actor_signature()):
            raise ValueError('Exact new named-edge actor signature/output size required; no legacy conversion')

    def new_executor(self):
        self.assert_identity()
        return _parse(self.source_bytes, self.policy_order, self.initial_guard)[0]

    def summary(self):
        executor = self.new_executor()
        reached = {self.initial_guard}
        while True:
            extended = reached | {a.end_node for a in executor.actions.values() if a.start_node in reached}
            if extended == reached:
                break
            reached = extended
        return {'identity': self.identity(), 'actor_signature': self.actor_signature(),
                'actions': [{'id': i, 'name': name, 'kind': kind, 'family': family,
                             'start_guard': start, 'end_guard': end, 'duration_s': duration}
                            for i, (name, kind, family, start, end, duration) in enumerate(self.actions, 1)],
                'structurally_reachable_guards': sorted(reached),
                'structurally_unreachable_choices': [a.name for a in executor.actions.values()
                                                     if a.start_node not in reached],
                'wait_only_reachable_guards': sorted(n for n in reached
                    if not any(a.start_node == n for a in executor.actions.values())),
                'unselected_report_edges_not_callable': list(self.unselected_actions),
                'automatic_bridge_or_opponent_label_selection': False,
                'actual_physics_identity_bound': False, 'admitted_to_training': False,
                'scope': 'Candidate command adapter only; not a new training environment, physical validation, or policy.'}


def load_variant_bank(path, *, policy_order, initial_guard):
    path = Path(path).resolve()
    raw = path.read_bytes()
    _, descriptions, unselected = _parse(raw, policy_order, initial_guard)
    bank = NamedEdgeBank(path, raw, policy_order, initial_guard, _sha(DEFAULT_PATH),
                        tuple((name, _sha(ROOT/name)) for name in IMPLEMENTATIONS),
                        descriptions, unselected)
    bank.assert_identity()
    return bank


class NamedEdgeController:
    """Own-state-only selection of explicitly named, independently masked edges."""
    def __init__(self, bank, *, max_observation_age_s=0.):
        if not isinstance(bank, NamedEdgeBank):
            raise ValueError('An explicitly loaded named-edge candidate bank is required')
        if (isinstance(max_observation_age_s, bool) or not isinstance(max_observation_age_s, numbers.Real)
            or not math.isfinite(max_observation_age_s) or max_observation_age_s < 0):
            raise ValueError('Finite nonnegative observation age required')
        self.bank, self.executor = bank, bank.new_executor()
        self.max_observation_age_s = float(max_observation_age_s)
        self._last_time = self._observation_time = None

    def _clock(self, now):
        if (isinstance(now, bool) or not isinstance(now, numbers.Real) or not math.isfinite(now) or
            now < 0 or (self._last_time is not None and now < self._last_time-1e-12)):
            raise ValueError('Finite nonnegative non-reversing simulation clock required')
        self.bank.assert_identity()
        self._last_time = float(now)
        return float(now)

    def _fresh(self, now):
        return (self._observation_time is not None and
                now-self._observation_time <= self.max_observation_age_s+1e-12)

    def observe(self, now, qpos, qvel):
        now = self._clock(now)
        self.executor.observe(now, qpos, qvel)
        self._observation_time = now

    def mask(self, now):
        now = self._clock(now)
        available = set(self.executor.available()) if self._fresh(now) else set()
        return np.asarray([True]+[name in available for name in self.bank.policy_order], dtype=bool)

    def actor_decision_allowed(self, now):
        return bool(self.mask(now)[1:].any())

    def request_id(self, action_id, now):
        now = self._clock(now)
        if type(action_id) is not int or not 0 <= action_id < len(self.bank.action_names):
            raise ValueError('Explicit named-edge policy ID required')
        if not self.mask(now)[action_id]:
            raise ValueError('Action unavailable at the actual settled guard, or observation is stale')
        if action_id:
            self.executor.request(self.bank.action_names[action_id], now)
        # wait=0 only keeps the current target. It never interrupts/restarts.

    def target(self, now):
        return self.executor.target(self._clock(now))

    def own_status(self, now):
        now = self._clock(now)
        mask = self.mask(now)
        active = self.executor.active
        action_id = self.bank.action_names.index(active) if active is not None else 0
        description = self.bank.actions[action_id-1] if action_id else None
        duration = description[5] if description else 0.
        elapsed = max(0., now-self.executor.started_s) if active is not None else 0.
        return {'own_state_schema': OWN_STATE_SCHEMA, 'own_node': self.executor.node,
                'own_active_action_id': action_id, 'own_active_action_name': active,
                'own_active_family': description[2] if description else None,
                'own_active_kind': description[1] if description else None,
                'own_action_duration_s': duration, 'own_elapsed_s': elapsed,
                'own_reference_progress': min(1., elapsed/duration) if duration else 0.,
                'busy': active is not None, 'measured_ready': self.executor.ready,
                'own_observation_fresh': self._fresh(now), 'action_mask': mask.tolist(),
                'actor_decision_allowed': bool(mask[1:].any()),
                'admitted_to_training': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--initial-guard', required=True)
    parser.add_argument('--action', action='append', required=True,
                        help='Exact edge name; argument order fixes policy IDs after wait=0')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('Keep existing catalog/snapshot identities; choose a fresh path')
    bank = load_variant_bank(args.source, policy_order=tuple(args.action), initial_guard=args.initial_guard)
    summary = bank.summary()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'policy_choices_including_wait':len(bank.action_names),
                      'unreachable':summary['structurally_unreachable_choices'],
                      'admitted_to_training':False}))


if __name__ == '__main__':
    main()
