"""Run bounded, full-window attack/parry diagnostics on a frozen bank.

Every cell keeps an identical-attack, exact-entry waiting baseline. A missing
baseline threat is NOT a successful block. This orchestrator cannot certify a
policy, reach an arbitrary entry guard, or relax collision/scoring thresholds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .a1_coverage_audit import ATTACKS, DEFENCES as DEFENCE_FAMILIES, family
from .a1_upright_audit import audit
from .multiguard_defence_probe import run


def resolve_defences(report, requested=None):
    """Resolve explicit variants from the bank, never invent policy actions."""
    available = tuple(name for name, row in report['actions'].items()
                      if row.get('kind') == 'defence' and family(name,row) in DEFENCE_FAMILIES)
    selected = tuple(requested) if requested else available
    if not selected or len(selected) != len(set(selected)):
        raise ValueError('Select at least one distinct declared defence variant')
    if any(name not in available for name in selected):
        raise ValueError('Requested defence must exist and declare an allowed defence family')
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--attack', action='append', choices=ATTACKS)
    parser.add_argument('--defence', action='append', help='Exact defence variant name in the selected bank')
    parser.add_argument('--attacker', action='append', choices=('a1', 'a2'))
    parser.add_argument('--offset', action='append', type=float)
    args = parser.parse_args()
    source, output = Path(args.bank).resolve(), Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('A fresh batch directory is required; old results stay unchanged')
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    report = json.loads(raw)
    attacks, defences = args.attack or ATTACKS, resolve_defences(report,args.defence)
    output.mkdir(parents=True)
    slots, offsets = args.attacker or ('a1', 'a2'), args.offset or (0.,)
    manifest = {'source_bank': str(source), 'source_sha256': digest,
                'status': 'running', 'admitted_to_training': False, 'cases': []}
    for attack in attacks:
        for defence in defences:
            for slot in slots:
                for offset in offsets:
                    if source.read_bytes() != raw:
                        raise RuntimeError('Bank changed during batch')
                    name = f'{attack}__{defence}__{slot}__offset_{offset:+.3f}'
                    row = {'attack': attack, 'defence': defence, 'attacker': slot,
                           'offset_s': offset, 'directory': name}
                    manifest['cases'].append(row)
                    try:
                        result = run(source, output/name, attack_name=attack,
                                     defence_name=defence, defence_offset_s=offset, attacker=slot)
                        row.update(status=result['status'], baseline_valid_threat=result['baseline_valid_threat'],
                                   physical_defence=result['successful_physical_defence'])
                        action = report['actions'][defence]
                        requirement = action.get('orientation_requirement', {})
                        interval = action.get('active_upright_interval_s')
                        if interval is None:
                            interval = action.get('active_interval_s')
                        # The exact phase declaration must come from the bank,
                        # not from looking at which contact frames happened to pass.
                        if result['successful_physical_defence'] and interval is not None:
                            defender = 'a2' if slot == 'a1' else 'a1'
                            audit_result = audit(output/name/'defended', output/name/'upright',
                                defender=defender,
                                active_start_s=result['defence_start_s']+interval[0],
                                active_end_s=result['defence_start_s']+interval[1],
                                min_elevation_deg=float(requirement.get('minimum_elevation_deg',25.)),
                                min_grip_height_m=float(requirement.get('minimum_grip_height_m',.35)),
                                comparison=output/name/'comparison.json')
                            row['upright_pass'] = audit_result['qualified_upright_example']
                    except (ValueError, RuntimeError, KeyError) as exc:
                        row.update(status='diagnostic_not_completed', error=f'{type(exc).__name__}: {exc}')
                    (output/'batch.json').write_text(json.dumps(manifest, indent=2)+'\n')
    manifest['status'] = 'completed_diagnostics_not_training_admission'
    (output/'batch.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
