"""Causal new-ready upper defence probe with immutable native provenance.

The offline timestamp selects one observed sword-state calibration example.
Runtime action choice receives only the shared current physical fields through
the existing CorridorPolicy, never an attack clock, name or future trajectory.
This one-corridor test does not certify a general reactive bank or RL policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .assembly import ROOT, _finite_json
from .a1_reactive_library import _gate_evidence, _references
from .a1_triangle_reactive import PUBLIC_FIELDS, corridor_observation, run


BASELINE = ROOT / 'outputs/compact_bank_v1_natural/native_upper01/native'
DEFENDED = ROOT / 'outputs/compact_bank_v1_new_defence/upper_dwell12_delay010/native'
GATE = DEFENDED.parent / 'comparison.json'
DEFAULT_LIBRARY = ROOT / 'outputs/compact_bank_v1_reactive/upper_library.json'


def build(output=DEFAULT_LIBRARY, calibration_s=.875):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Keep the calibrated source library immutable')
    scoring_path = ROOT / 'config/scoring_triangle_v1_candidate.json'
    scoring = json.loads(scoring_path.read_text())
    gate = json.loads(GATE.read_text())
    if gate.get('status') != 'qualified':
        raise ValueError('A qualified full-window native source is required')
    attacker, defender, summaries = _gate_evidence(gate, BASELINE, DEFENDED, scoring)
    if (attacker, defender) != ('a1', 'a2'):
        raise ValueError('This fixed provenance expects a1 attack and a2 defence')
    attack = _references(BASELINE, attacker, summaries['baseline'])
    defence = _references(DEFENDED, defender, summaries['defended'])
    if not np.array_equal(defence['qpos'][0], defence['qpos'][-1]):
        raise ValueError('The new-ready defence must be a physically closed skill')
    if not np.array_equal(attack['qpos'][0], defence['qpos'][0]):
        raise ValueError('Attack and defence need the same new-ready entry')
    calibrated = corridor_observation(BASELINE, calibration_s, observer=defender)
    physical = calibrated['physical']
    provenance = {}
    for path in (GATE, scoring_path):
        provenance[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for folder in (BASELINE, DEFENDED):
        for filename in ('scene.xml', 'trace.npz', 'summary.json', 'references.npz'):
            path = folder / filename
            provenance[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    library = {'schema_version': 1, 'candidate_only': True,
               'admitted_to_training': False, 'duration_s': 8.,
               'scoring': str(scoring_path.relative_to(ROOT)),
               'cases': {'upper': {'region': 'upper',
                                  'ready_qpos': defence['qpos'][0], 'attack': attack}},
               'skills': [{'name': 'defence_upper', 'region': 'upper',
                           'defence': defence, 'calibration': calibrated,
                           'corridor': {'tip': physical[PUBLIC_FIELDS[2]],
                                        'grip': physical[PUBLIC_FIELDS[3]],
                                        'velocity': physical[PUBLIC_FIELDS[4]]}}],
               'provenance_sha256': provenance,
               'scope': 'One observed new-ready upper corridor; no opponent future, script phase or skill ID at runtime; not a learned policy or bank admission'}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_finite_json(library), indent=2) + '\n')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['build', 'run'])
    parser.add_argument('--library', type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/compact_bank_v1_reactive/upper_observed01')
    parser.add_argument('--calibration-s', type=float, default=.875)
    parser.add_argument('--start-s', type=float, default=.76)
    parser.add_argument('--time-scale', type=float, default=1.)
    parser.add_argument('--attacker', choices=['a1', 'a2'], default='a1')
    args = parser.parse_args()
    if args.stage == 'build':
        print(build(args.library, args.calibration_s))
    else:
        run(args.library, 'upper', args.output, attacker=args.attacker,
            start_s=args.start_s, time_scale=args.time_scale)


if __name__ == '__main__':
    main()
