"""Pure completion accounting for bounded physical diagnostic trials.

A diagnostic horizon is not a physical failure. Conversely, extending a
horizon cannot repair a physical or evidence fault that already occurred.
The caller supplies measured execution state; this module does not infer
completion from a motion's label or from the requested episode duration.

This schema is additive. It neither rewrites historical summaries nor changes
training-environment or match scoring. In particular, an already scored hit
must be retained separately even when its subsequent diagnostic is truncated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Mapping


@dataclass(frozen=True)
class ExecutorCompletionSnapshot:
    """Measured end state, plus the end of any active *reference*.

    ``settled`` must include the caller's measured position, speed and dwell
    checks. A reference reaching its last timestamp alone is not settling.
    Unserved script requests include future and currently unavailable ones;
    they must not disappear merely because the episode horizon was reached.
    """

    active_action: str | None = None
    reference_end_s: float | None = None
    settled: bool = False
    pending_script_requests: tuple[str, ...] = ()


@dataclass(frozen=True)
class TrialCompletionSnapshot:
    completed_time_s: float
    requested_duration_s: float
    # 'horizon' means the requested simulated window was actually reached.
    # A physical/evidence error can still take precedence at that instant.
    stop_reason: str
    executors: Mapping[str, ExecutorCompletionSnapshot]
    physical_failures: tuple[str, ...] = field(default_factory=tuple)
    evidence_failures: tuple[str, ...] = field(default_factory=tuple)
    execution_failures: tuple[str, ...] = field(default_factory=tuple)


def _nonnegative_finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative')
    return float(value)


def _names(values, name):
    if not isinstance(values, (tuple, list)) or any(
            not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f'{name} must be a sequence of nonempty names')
    return list(values)


def assess_trial_completion(snapshot: TrialCompletionSnapshot) -> dict:
    """Classify a diagnostic stop without turning truncation into failure/draw.

    Physical faults take precedence over evidence faults, execution errors,
    wall budget and unfinished work. All supplied faults and outstanding work
    remain in the returned record even when another category has precedence.

    ``latest_active_reference_end_s`` is only an earliest reference deadline
    for already started actions. It is *not* a guaranteed completion time:
    settling, queued actions and observation-triggered reactions can take
    longer or never become available. No amount of extra episode time is
    promised to cure a recovery or action-availability problem.
    """
    if not isinstance(snapshot, TrialCompletionSnapshot):
        raise TypeError('Explicit TrialCompletionSnapshot required')
    now = _nonnegative_finite(snapshot.completed_time_s, 'completed_time_s')
    horizon = _nonnegative_finite(snapshot.requested_duration_s, 'requested_duration_s')
    if horizon == 0:
        raise ValueError('requested_duration_s must be positive')
    if snapshot.stop_reason not in ('horizon', 'wall_budget', 'interrupted'):
        raise ValueError('Unknown diagnostic stop_reason')
    if snapshot.stop_reason == 'horizon' and now < horizon - 1e-9:
        raise ValueError('A horizon stop requires reaching the simulated horizon')
    if not isinstance(snapshot.executors, Mapping) or not snapshot.executors:
        raise ValueError('At least one explicit executor snapshot required')
    physical = _names(snapshot.physical_failures, 'physical_failures')
    evidence = _names(snapshot.evidence_failures, 'evidence_failures')
    execution = _names(snapshot.execution_failures, 'execution_failures')

    unfinished, active_ends = {}, []
    for slot, state in snapshot.executors.items():
        if not isinstance(slot, str) or not slot.strip() \
                or not isinstance(state, ExecutorCompletionSnapshot):
            raise ValueError('Named ExecutorCompletionSnapshot entries required')
        if not isinstance(state.settled, bool):
            raise ValueError('settled must be an explicit boolean')
        pending = _names(state.pending_script_requests, 'pending_script_requests')
        end = state.reference_end_s
        if state.active_action is None:
            if end is not None:
                raise ValueError('Inactive executor must not have a reference_end_s')
            phase = 'settled' if state.settled else 'recovery_unsettled'
        else:
            if not isinstance(state.active_action, str) or not state.active_action.strip():
                raise ValueError('active_action must be a nonempty name or None')
            end = _nonnegative_finite(end, 'reference_end_s')
            if state.settled:
                raise ValueError('An active action cannot be declared completed and settled')
            active_ends.append(end)
            phase = 'active_reference' if now < end - 1e-9 else 'recovery_unsettled'
        if phase != 'settled' or pending:
            unfinished[slot] = {
                'phase': phase,
                'active_action': state.active_action,
                'reference_end_s': end,
                'reference_remaining_s': None if end is None else max(0., end - now),
                'settled': state.settled,
                'pending_script_requests': pending,
            }

    if physical:
        category = 'physical_failure'
    elif evidence:
        category = 'evidence_failure'
    elif execution:
        category = 'execution_error'
    elif snapshot.stop_reason == 'wall_budget':
        category = 'wall_budget_exhausted'
    elif snapshot.stop_reason != 'horizon':
        category = 'interrupted'
    elif any(s['phase'] == 'active_reference' for s in unfinished.values()):
        category = 'horizon_active_reference'
    elif any(s['pending_script_requests'] for s in unfinished.values()):
        category = 'horizon_unserved_script_requests'
    elif unfinished:
        category = 'horizon_recovery_unsettled'
    else:
        category = 'complete'
    complete = category == 'complete'
    truncated = category in ('wall_budget_exhausted', 'interrupted',
        'horizon_active_reference', 'horizon_unserved_script_requests',
        'horizon_recovery_unsettled')
    return {
        'schema_version': 'trial_completion_v1',
        'termination_category': category,
        'complete': complete,
        'is_truncation': truncated,
        'completed_time_s': now,
        'requested_duration_s': horizon,
        'stop_reason': snapshot.stop_reason,
        'physical_failures': physical,
        'evidence_failures': evidence,
        'execution_failures': execution,
        'unfinished': unfinished,
        'latest_active_reference_end_s': max(active_ends) if active_ends else None,
        'reference_end_guarantees_actual_settling': False,
        'allows_timeout_draw_if_no_scored_result': complete,
    }
