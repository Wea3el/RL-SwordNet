"""Candidate-only multi-guard action families, with explicit directed routes.

Nine policy choices are not nine unconditional motions.  A family resolves to
an explicitly authored edge at the measured current guard, or is unavailable.
Repositioning edges are separate diagnostic commands; this module never adds
bridges, resets qpos, guesses a route, interrupts a motion, or grants training
admission.  Endpoint settling remains compulsory.  Momentum-preserving skill
boundaries and their interaction evidence are intentionally not claimed here.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .assembly import ROOT
from .candidate_graph import _numbers, _unique_object
from .feedback_policy import GraphAction, SkillGraphExecutor
from .joint_constraints import DEFAULT_PATH, JOINT_NAMES
from .motion_spline import C2JointSpline


ACTION_NAMES = ('wait', 'cut_down_left_to_right', 'cut_down_right_to_left',
                'cut_up_left_to_right', 'cut_up_right_to_left', 'cut_down_center',
                'thrust_center', 'parry_left', 'parry_right', 'parry_high')
GUARD_NAMES = ('left_high', 'right_high', 'left_low', 'right_low', 'center')
GEOMETRY_ALIASES = dict(zip(
    ('cut_down_lr', 'cut_down_rl', 'cut_up_lr', 'cut_up_rl', 'cut_vertical',
     'thrust', 'parry_left', 'parry_right', 'parry_high'), ACTION_NAMES[1:]))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _kind(name):
    return 'defence' if name.startswith('parry_') else 'attack'


def _parse(source):
    cfg = json.loads(source, object_pairs_hook=_unique_object)
    if not isinstance(cfg, dict) or type(cfg.get('schema_version')) is not int or cfg['schema_version'] != 1:
        raise ValueError('Multi-guard schema 1 required')
    if cfg.get('bank_kind') != 'multiguard_v1_candidate' or cfg.get('policy_family') != 'A1':
        raise ValueError('Explicit A1 multi-guard candidate required')
    if cfg.get('admitted_to_training') is not False or cfg.get('legacy_actor_compatible') is not False:
        raise ValueError('Candidate cannot grant training or legacy actor compatibility')
    if cfg.get('action_ids') != {str(i): name for i, name in enumerate(ACTION_NAMES)}:
        raise ValueError('Exact stable nine-family plus wait ID map required')
    if (cfg.get('angle_unit') != 'radian' or not isinstance(cfg.get('joint_names'), list)
            or tuple(cfg['joint_names']) != JOINT_NAMES
            or cfg.get('joint_constraints') != 'config/joint_constraints_observed.json'
            or cfg.get('reference_generator') != 'sim.motion_spline.C2JointSpline'):
        raise ValueError('Confirmed ordered model coordinates and C2 generator required')
    nodes, guards, specs = cfg.get('nodes'), cfg.get('guard_nodes'), cfg.get('actions')
    if not isinstance(nodes, dict) or not nodes or any(
            not isinstance(k, str) or not k.strip() or not _numbers(q, length=7)
            for k, q in nodes.items()):
        raise ValueError('Named finite seven-joint nodes required')
    if (not isinstance(guards, dict) or set(guards) != set(GUARD_NAMES)
            or any(not isinstance(n, str) or n not in nodes for n in guards.values())
            or len(set(guards.values())) != len(GUARD_NAMES)):
        raise ValueError('Five distinct named guard nodes required')
    initial = cfg.get('initial_node')
    if not isinstance(initial, str) or initial not in guards.values():
        raise ValueError('Initial node must be an explicit guard')
    if not isinstance(specs, dict):
        raise ValueError('Explicit action dictionary required, even for an empty candidate')
    actions = {}
    for name, spec in specs.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(spec, dict):
            raise ValueError('Explicit named action objects required')
        route = spec.get('waypoint_nodes')
        if (not isinstance(route, list) or len(route) < 2
                or any(not isinstance(n, str) or n not in nodes for n in route)
                or route[0] != spec.get('start_node') or route[-1] != spec.get('end_node')
                or route[0] not in guards.values() or route[-1] not in guards.values()):
            raise ValueError(f'{name}: explicit guard-to-guard route required')
        times = spec.get('knot_time_s')
        if not _numbers(times, length=len(route)):
            raise ValueError(f'{name}: finite times matching route required')
        actions[name] = GraphAction(name, route[0], route[-1], C2JointSpline(times, [nodes[n] for n in route]))
    families = cfg.get('families')
    if not isinstance(families, dict) or set(families) != set(ACTION_NAMES[1:]):
        raise ValueError('Explicit nine-family registry required; missing families use empty routes')
    owned = set()
    routes = []
    for action_id, name in enumerate(ACTION_NAMES[1:], 1):
        family = families[name]
        if (not isinstance(family, dict) or type(family.get('id')) is not int
                or family['id'] != action_id or family.get('kind') != _kind(name)
                or not isinstance(family.get('routes'), dict)):
            raise ValueError(f'{name}: exact ID, kind and route map required')
        for entry, action in family['routes'].items():
            if (entry not in guards.values() or not isinstance(action, str) or action not in actions
                    or actions[action].start_node != entry or action in owned):
                raise ValueError(f'{name}: each edge needs one owner and its actual entry guard')
            owned.add(action)
        routes.append(tuple(family['routes'].items()))
    transitions = cfg.get('control_transitions', [])
    if (not isinstance(transitions, list) or any(not isinstance(n, str) or n not in actions for n in transitions)
            or len(set(transitions)) != len(transitions) or owned.intersection(transitions)):
        raise ValueError('Unique explicit control transitions, separate from policy families, required')
    if owned.union(transitions) != set(actions):
        raise ValueError('Every action must be an explicit family edge or control transition')
    identity = cfg.get('source_sha256', {})
    if not isinstance(identity, dict) or any(not isinstance(p, str) or not p or not isinstance(d, str)
            or len(d) != 64 or any(c not in '0123456789abcdef' for c in d) for p, d in identity.items()):
        raise ValueError('Source identity must contain paths and lowercase SHA-256 digests')
    executor = SkillGraphExecutor(nodes, actions, initial)
    def reachable_through(edges, start=initial):
        reachable = {start}
        while True:
            extended = reachable | {a.end_node for a in edges if a.start_node in reachable}
            if extended == reachable:
                return frozenset(reachable)
            reachable = extended
    reachable = reachable_through(tuple(actions.values()))
    policy_edges = tuple(actions[name] for name in owned)
    policy_reachable = reachable_through(policy_edges)
    policy_dead_ends = frozenset(n for n in policy_reachable
                                if not any(actions[name].start_node == n for name in owned))
    # Initial reachability is insufficient: an early choice can enter a closed
    # cycle which has outgoing actions but permanently loses other families.
    policy_limited_guards = tuple((guard, tuple(
        family for family, family_routes in zip(ACTION_NAMES[1:], routes)
        if family_routes and not any(entry in reachable_through(policy_edges, guard)
                                     for entry, _ in family_routes)))
        for guard in sorted(policy_reachable))
    return (cfg, executor, tuple(routes), tuple(transitions), reachable,
            policy_reachable, policy_dead_ends, policy_limited_guards)


@dataclass(frozen=True)
class MultiGuardBank:
    source_path: Path
    source_bytes: bytes
    source_sha256: str
    joint_constraints_sha256: str
    family_routes: tuple
    control_transitions: tuple
    reachable_nodes: frozenset
    policy_reachable_nodes: frozenset
    policy_dead_end_guards: frozenset
    policy_limited_guards: tuple
    source_identity: tuple

    @property
    def missing_families(self):
        return tuple(name for name, routes in zip(ACTION_NAMES[1:], self.family_routes) if not routes)

    @property
    def unreachable_families(self):
        return tuple(name for name, routes in zip(ACTION_NAMES[1:], self.family_routes)
                     if routes and not any(entry in self.reachable_nodes for entry, _ in routes))

    @property
    def policy_unreachable_families(self):
        """Families requiring an operator-only reposition are not policy reachable."""
        return tuple(name for name, routes in zip(ACTION_NAMES[1:], self.family_routes)
                     if routes and not any(entry in self.policy_reachable_nodes for entry, _ in routes))

    def assert_identity(self):
        if self.source_path.read_bytes() != self.source_bytes or _sha(DEFAULT_PATH) != self.joint_constraints_sha256:
            raise ValueError('Bank or confirmed constraints changed; reload and revalidate')
        for name, digest in self.source_identity:
            path = Path(name) if Path(name).is_absolute() else ROOT / name
            if _sha(path) != digest:
                raise ValueError('Multi-guard source identity changed: ' + name)

    def new_executor(self):
        self.assert_identity()
        return _parse(self.source_bytes)[1]


def load_multiguard_bank(path, *, require_complete=False):
    """Load partial geometry without calling it a complete or admitted library.

    Disconnected components are preserved as offline candidates, not bridged.
    require_complete checks policy-only structural reachability, dead ends and
    trapped cycles from every reachable guard, never physical gates.
    Operator-only repositioning cannot satisfy that test.
    """
    if type(require_complete) is not bool:
        raise ValueError('Explicit structural completeness flag required')
    path = Path(path).resolve()
    source = path.read_bytes()
    (cfg, _, routes, transitions, reachable, policy_reachable,
     policy_dead_ends, policy_limited_guards) = _parse(source)
    bank = MultiGuardBank(path, source, hashlib.sha256(source).hexdigest(), _sha(DEFAULT_PATH),
                          routes, transitions, reachable, policy_reachable, policy_dead_ends, policy_limited_guards,
                          tuple(cfg.get('source_sha256', {}).items()))
    bank.assert_identity()
    if require_complete and (bank.missing_families or bank.policy_unreachable_families
                             or bank.policy_dead_end_guards
                             or any(lost for _, lost in bank.policy_limited_guards)):
        raise ValueError('Incomplete or unreachable nine-family bank; no implicit bridges allowed')
    return bank


class MultiGuardController:
    """Own-state, current-guard route selector for diagnostic execution only."""
    def __init__(self, path, *, max_observation_age_s=0., require_complete=False):
        if (type(max_observation_age_s) not in (int, float)
                or not math.isfinite(max_observation_age_s) or max_observation_age_s < 0):
            raise ValueError('Finite nonnegative observation age required')
        self.bank = load_multiguard_bank(path, require_complete=require_complete)
        self.executor = self.bank.new_executor()
        self.max_observation_age_s = float(max_observation_age_s)
        self.last_observation_s = None
        self.last_request_s = None
        self.active_family_id = 0

    def _clock(self, now):
        if type(now) not in (int, float) or not math.isfinite(now) or now < 0:
            raise ValueError('Finite nonnegative simulation timestamp required')
        previous = [t for t in (self.last_observation_s, self.last_request_s) if t is not None]
        if previous and now < max(previous)-1e-12:
            raise ValueError('Stale or reversed simulation timestamp')
        return float(now)

    def _fresh(self, now):
        return (self.last_observation_s is not None
                and now-self.last_observation_s <= self.max_observation_age_s+1e-12)

    def observe(self, now, q, velocity):
        now = self._clock(now)
        self.bank.assert_identity()
        self.executor.observe(now, q, velocity)
        self.last_observation_s = now
        if self.executor.active is None:
            self.active_family_id = 0

    def mask(self, now):
        now = self._clock(now)
        available = set(self.executor.available()) if self._fresh(now) else set()
        return np.asarray([True] + [any(entry == self.executor.node and edge in available for entry, edge in routes)
                                   for routes in self.bank.family_routes], dtype=bool)

    def actor_decision_allowed(self, now):
        now = self._clock(now)
        return bool(self._fresh(now) and self.executor.ready and self.executor.active is None)

    def available_control_transitions(self, now):
        now = self._clock(now)
        available = set(self.executor.available()) if self._fresh(now) else set()
        return tuple(n for n in self.bank.control_transitions if n in available)

    def target(self, now):
        return self.executor.target(self._clock(now))

    def request_id(self, action_id, now):
        now = self._clock(now)
        self.bank.assert_identity()
        if type(action_id) is not int or not 0 <= action_id < len(ACTION_NAMES):
            raise ValueError('Integer action ID 0..9 required')
        if not self.mask(now)[action_id]:
            raise ValueError('Family unavailable at measured current guard or observation is stale')
        self.last_request_s = now
        if action_id:
            edge = dict(self.bank.family_routes[action_id-1])[self.executor.node]
            self.executor.request(edge, now)
            self.active_family_id = action_id
        # Zero retains the current reference, even while busy or not observed.

    def request_control_transition(self, name, now):
        """Explicit reposition command; not a policy-family action or auto-route."""
        now = self._clock(now)
        self.bank.assert_identity()
        if name not in self.available_control_transitions(now):
            raise ValueError('Unavailable explicit control transition')
        self.last_request_s = now
        self.executor.request(name, now)
        self.active_family_id = 0

    def own_status(self, now):
        now = self._clock(now)
        active = self.executor.active
        progress = 0. if active is None else min(1., max(0.,
            (now-self.executor.started_s)/self.executor.actions[active].curve.duration_s))
        return {'actor_decision_allowed': self.actor_decision_allowed(now),
                'available_mask': self.mask(now).tolist(), 'own_node': self.executor.node,
                'own_active_family_id': self.active_family_id, 'own_active_route': active,
                'own_reference_progress': progress,
                'own_observation_fresh': self._fresh(now),
                'available_control_transitions': self.available_control_transitions(now),
                'missing_families': self.bank.missing_families,
                'unreachable_families': self.bank.unreachable_families,
                'policy_unreachable_families': self.bank.policy_unreachable_families,
                'policy_dead_end_guards': tuple(sorted(self.bank.policy_dead_end_guards)),
                'policy_limited_guards': dict(self.bank.policy_limited_guards),
                'training_admitted': False}


def geometry_report_to_payload(report_path, *, bank_version='multiguard_geometry_candidate_01', initial_guard='center'):
    """Import explicit successful geometry paths, not evidence of real dynamics.

    No missing family is replaced by a duplicate or fabricated hold.  Failed
    candidates remain identified in import_review; no automatic transition is
    generated.  The caller saves the returned JSON and runs native gates.
    """
    path = Path(report_path).resolve()
    report = json.loads(path.read_text(), object_pairs_hook=_unique_object)
    raw_guards, raw_actions = report.get('guards'), report.get('actions')
    if not isinstance(raw_guards, dict) or set(raw_guards) != set(GUARD_NAMES) or not isinstance(raw_actions, dict):
        raise ValueError('Geometry report needs five guards and explicit action paths')
    guards = {name: name for name in GUARD_NAMES}
    nodes = {name: raw_guards[name]['qpos_rad'] for name in GUARD_NAMES}
    if initial_guard not in guards:
        raise ValueError('Known explicit initial guard required')
    families = {name: {'id': i, 'kind': _kind(name), 'routes': {}}
                for i, name in enumerate(ACTION_NAMES[1:], 1)}
    actions, review = {}, {}
    for name, row in raw_actions.items():
        family = GEOMETRY_ALIASES.get(row.get('family', name), row.get('family', name))
        if family not in families:
            raise ValueError('Unknown geometry action family: ' + str(family))
        if row.get('kind') != _kind(family):
            raise ValueError('Geometry family kind mismatch: ' + name)
        if row.get('candidate_pass') is not True:
            review[name] = 'not_imported_geometry_candidate_did_not_pass'
            continue
        start, end = row.get('start_guard'), row.get('end_guard')
        q = row.get('knot_qpos'); times = row.get('knot_time_s')
        if (start not in guards or end not in guards or not isinstance(q, list) or len(q) < 2
                or not all(_numbers(v, length=7) for v in q) or not _numbers(times, length=len(q))
                or not np.array_equal(q[0], nodes[start]) or not np.array_equal(q[-1], nodes[end])):
            raise ValueError('Geometry paths must preserve exact authored guard endpoints: ' + name)
        if start in families[family]['routes']:
            raise ValueError('Ambiguous duplicate family entry; select an explicit candidate: ' + name)
        route = [start]
        for index, pose in enumerate(q[1:-1], 1):
            key = f'path::{name}::{index}'
            if key in nodes:
                raise ValueError('Duplicate imported waypoint')
            nodes[key] = pose
            route.append(key)
        route.append(end)
        actions[name] = {'start_node': start, 'end_node': end, 'waypoint_nodes': route,
                         'knot_time_s': times, 'source_geometry_action': name,
                         'source_evidence_scope': 'geometry_only_native_and_interaction_pending'}
        families[family]['routes'][start] = name
        review[name] = 'geometry_imported_not_native_or_training_admitted'
    cfg = {'schema_version': 1, 'bank_kind': 'multiguard_v1_candidate', 'bank_version': bank_version,
           'policy_family': 'A1', 'admitted_to_training': False, 'legacy_actor_compatible': False,
           'joint_names': list(JOINT_NAMES), 'angle_unit': 'radian',
           'joint_constraints': 'config/joint_constraints_observed.json',
           'reference_generator': 'sim.motion_spline.C2JointSpline', 'initial_node': initial_guard,
           'action_ids': {str(i): name for i, name in enumerate(ACTION_NAMES)}, 'guard_nodes': guards,
           'nodes': nodes, 'actions': actions, 'families': families, 'control_transitions': [],
           'source_sha256': {str(path): _sha(path)}, 'import_review': review,
           'context_routes_not_imported': sorted(set(report.get('context_routes', {})) - set(actions)),
           'boundary_semantics': 'C2 rest endpoints plus measured settling; no momentum-boundary claim',
           'status': 'partial_candidate_not_physics_or_training_admitted'}
    _parse(json.dumps(cfg))
    return cfg


def bank_summary(bank):
    """Describe structural readiness without converting it into certification."""
    return {
        'bank': str(bank.source_path),
        'source_sha256': bank.source_sha256,
        'family_route_counts': {name: len(routes) for name, routes
                                in zip(ACTION_NAMES[1:], bank.family_routes)},
        'missing_families': list(bank.missing_families),
        'policy_unreachable_families': list(bank.policy_unreachable_families),
        'policy_dead_end_guards': sorted(bank.policy_dead_end_guards),
        'policy_limited_guards': {guard: list(lost) for guard, lost in bank.policy_limited_guards},
        'context_routes_not_imported': json.loads(bank.source_bytes).get('context_routes_not_imported', []),
        'control_transitions': list(bank.control_transitions),
        'training_admitted': False,
        'scope': 'Candidate graph only; native attack, paired defence and feedback gates remain separate.'}


def export_geometry_bank(source, output, *, initial_guard='center'):
    """Create a new inspectable candidate artifact; never replace old evidence."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Existing bank is preserved; choose a fresh output path')
    payload = geometry_report_to_payload(source, initial_guard=initial_guard)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also protects against another writer after the check.
    with output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    return load_multiguard_bank(output)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    build = commands.add_parser('import-geometry', help='Export a candidate, not a trained or admitted bank')
    build.add_argument('--source', required=True)
    build.add_argument('--output', required=True)
    build.add_argument('--initial-guard', choices=GUARD_NAMES, default='center')
    inspect = commands.add_parser('inspect', help='Check identity and report missing/unreachable routes')
    inspect.add_argument('--bank', required=True)
    inspect.add_argument('--require-complete', action='store_true', help='Structural test only, not physics admission')
    args = parser.parse_args()
    if args.command == 'import-geometry':
        bank = export_geometry_bank(args.source, args.output, initial_guard=args.initial_guard)
    else:
        bank = load_multiguard_bank(args.bank, require_complete=args.require_complete)
    print(json.dumps(bank_summary(bank), ensure_ascii=False, indent=2))
