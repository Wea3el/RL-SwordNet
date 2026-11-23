"""Load explicit robot-space candidate skills without granting training admission.

This format stores named waypoints and whole C2 curves, not isolated poses to
teleport to. It is deliberately distinct from the legacy five-action bank.
Physical, temporal and opponent-coverage evidence is still required separately.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

from .feedback_policy import GraphAction, SkillGraphExecutor
from .joint_constraints import DEFAULT_PATH, JOINT_NAMES
from .motion_spline import C2JointSpline


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate candidate field: {key}')
        result[key] = value
    return result


def _numbers(values, *, length=None):
    return (isinstance(values, list) and (length is None or len(values) == length)
            and all(type(v) in (int, float) and math.isfinite(v) for v in values))


def _executor(source):
    cfg = json.loads(source, object_pairs_hook=_unique_object)
    if not isinstance(cfg, dict) or type(cfg.get('schema_version')) is not int or cfg['schema_version'] != 1:
        raise ValueError('Candidate graph schema 1 required')
    if cfg.get('policy_family') != 'A1' or cfg.get('angle_unit') != 'radian':
        raise ValueError('Explicit A1 graph in model radians required')
    if not isinstance(cfg.get('joint_names'), list) or tuple(cfg['joint_names']) != JOINT_NAMES:
        raise ValueError('Exact ordered J0 through J6 required')
    if cfg.get('joint_constraints') != 'config/joint_constraints_observed.json':
        raise ValueError('Use the confirmed project joint constraints')
    if cfg.get('reference_generator') != 'sim.motion_spline.C2JointSpline':
        raise ValueError('Explicit C2 reference generator required')
    if cfg.get('admitted_to_training') is not False:
        raise ValueError('Candidate loading cannot authorize training admission')
    nodes, specs = cfg.get('nodes'), cfg.get('actions')
    if not isinstance(nodes, dict) or not nodes or not isinstance(specs, dict) or not specs:
        raise ValueError('Nonempty named waypoints and actions required')
    if any(not isinstance(k, str) or not k.strip() for k in (*nodes, *specs)):
        raise ValueError('Nonempty waypoint and action names required')
    if any(not _numbers(q, length=7) for q in nodes.values()):
        raise ValueError('Seven finite numeric coordinates per waypoint required')
    initial = cfg.get('initial_node')
    if not isinstance(initial, str) or initial not in nodes:
        raise ValueError('Initial node is missing')
    actions = {}
    for name, action in specs.items():
        if not isinstance(action, dict):
            raise ValueError(f'{name}: explicit action object required')
        route = action.get('waypoint_nodes')
        if (not isinstance(route, list) or len(route) < 2
                or any(not isinstance(n, str) or n not in nodes for n in route)):
            raise ValueError(f'{name}: known waypoint route required')
        if route[0] != action.get('start_node') or route[-1] != action.get('end_node'):
            raise ValueError(f'{name}: route must preserve exact endpoint identity')
        times = action.get('knot_time_s')
        if not _numbers(times, length=len(route)):
            raise ValueError(f'{name}: finite numeric times matching the route required')
        curve = C2JointSpline(times, [nodes[n] for n in route])
        actions[name] = GraphAction(name, route[0], route[-1], curve)
    # The executor enforces all raw joint bounds and the whole Bezier envelope.
    executor = SkillGraphExecutor(nodes, actions, initial)
    reachable = {initial}
    while True:
        extended = reachable | {a.end_node for a in actions.values() if a.start_node in reachable}
        if extended == reachable:
            break
        reachable = extended
    if any(a.start_node not in reachable for a in actions.values()):
        raise ValueError('An action entry is disconnected from the initial node')
    return executor


@dataclass(frozen=True)
class CandidateGraph:
    source_path: Path
    source_bytes: bytes
    source_sha256: str
    joint_constraints_sha256: str

    def new_executor(self):
        """Independent mutable execution state for each physical agent."""
        if hashlib.sha256(DEFAULT_PATH.read_bytes()).hexdigest() != self.joint_constraints_sha256:
            raise ValueError('Joint constraints changed; load and revalidate the candidate')
        return _executor(self.source_bytes)


def load_candidate_graph(path):
    path = Path(path).resolve()
    source = path.read_bytes()
    _executor(source)
    return CandidateGraph(path, source, hashlib.sha256(source).hexdigest(),
                          hashlib.sha256(DEFAULT_PATH.read_bytes()).hexdigest())
