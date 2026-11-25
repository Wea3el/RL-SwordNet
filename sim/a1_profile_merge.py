"""Combine explicitly selected candidate references, never their certificates.

All inputs remain immutable. Identical scoring, blade geometry and exact guard
coordinates are required; conflicting guards cannot be silently renamed or
bridged. The output is a diagnostic bank, not a training admission.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from .motion_spline import C2JointSpline


def merge(base, replacements, output, *, additions=()):
    base, output = Path(base).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preserve existing banks and evidence')
    sources = {base: base.read_bytes()}
    parent = json.loads(sources[base])
    if any(row.get('candidate_stage') == 'authorized_native_research_hypothesis'
           for row in parent['actions'].values()):
        raise ValueError('A native-only research hypothesis cannot be a library base')
    bank = {key: copy.deepcopy(parent[key]) for key in (
        'schema_version', 'guards', 'actions', 'scoring', 'blade_geometry',
        'blade_profile_contract') if key in parent}
    selected = []
    added = []
    seen = set()
    requests = [(name,path,False)for name,path in replacements]
    requests += [(name,path,True)for name,path in additions]
    for name, path, is_addition in requests:
        if name in seen or (is_addition == (name in bank['actions'])):
            raise ValueError('Replacement requires an existing action; addition requires a new distinct name')
        seen.add(name)
        path = Path(path).resolve()
        sources.setdefault(path, path.read_bytes())
        source = json.loads(sources[path])
        if source['scoring'] != bank['scoring'] or source['blade_geometry'] != bank['blade_geometry']:
            raise ValueError('Different scoring or blade geometry requires separate reconciliation')
        row = copy.deepcopy(source['actions'][name])
        if row.get('candidate_stage') == 'authorized_native_research_hypothesis':
            raise ValueError('Native research permission cannot promote an action into the library')
        if is_addition and (row.get('kind') != 'defence' or
                row.get('family') not in ('parry_left','parry_right','parry_high')):
            raise ValueError('New diagnostic subskills must declare an existing defence family')
        for field, knot in (('start_guard', 0), ('end_guard', -1)):
            guard = row[field]
            node = source['guards'][guard]
            if guard in bank['guards'] and bank['guards'][guard]['qpos_rad'] != node['qpos_rad']:
                raise ValueError('Conflicting exact guard: '+guard)
            if not np.array_equal(row['knot_qpos'][knot], node['qpos_rad']):
                raise ValueError('Reference endpoint does not match its declared guard')
            bank['guards'][guard] = copy.deepcopy(node)
        C2JointSpline(row['knot_time_s'], row['knot_qpos'])
        bank['actions'][name] = row
        (added if is_addition else selected).append({'action': name, 'source': str(path),
            'sha256': hashlib.sha256(sources[path]).hexdigest()})
    for row in bank['actions'].values():
        row['admitted_to_training'] = False
        row['prior_certificate_inherited'] = False
    bank.update(status='explicitly_merged_diagnostic_candidates_only',
        admitted_to_training=False, source_certificates_inherited=False,
        composition={'base_source': str(base), 'base_sha256': hashlib.sha256(sources[base]).hexdigest(),
            'selected_replacements': selected,
            'selected_additions': added,
            'action_space_expanded': bool(added),
            'runtime_and_snapshot_migration_required': bool(added),
            'parent_metadata_not_reapplied_to_changed_actions': True,
            'scope': 'Exact reference assembly only. Independently bind any paired evidence; no policy or connectivity certificate.'})
    bank['outgoing_actions'] = {}
    for name, row in bank['actions'].items():
        if row.get('candidate_pass', False):
            bank['outgoing_actions'].setdefault(row['start_guard'], []).append(name)
    if any(path.read_bytes() != raw for path, raw in sources.items()):
        raise RuntimeError('A source changed during composition')
    output.mkdir(parents=True)
    (output/'report.json').write_text(json.dumps(bank, ensure_ascii=False, indent=2)+'\n')
    return bank


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', required=True)
    p.add_argument('--replace', nargs=2, action='append', metavar=('ACTION', 'SOURCE'), default=[])
    p.add_argument('--add', nargs=2, action='append', metavar=('ACTION', 'SOURCE'), default=[])
    p.add_argument('--output', required=True)
    a = p.parse_args()
    merge(a.base, a.replace, a.output, additions=a.add)
