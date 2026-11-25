"""Inspect a frozen A1 pose package; never train, integrate or admit it.

This is deliberately incompatible with the legacy five-action A1Library.
Named graph edges keep their exact C2 knots and distinct endpoint contexts.
Hash/limit checks do not certify collisions, defence, arbitrary successors,
or compatibility with a future shared A1/A2 training environment.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np

from .feedback_policy import GraphAction, SkillGraphExecutor
from .joint_constraints import DEFAULT_PATH, load_joint_constraints, model_joint_ranges_rad
from .motion_spline import C2JointSpline


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGE = ROOT / 'assets/a1_pose_bank_v1'
DEFAULT_CONSTRAINTS_PATH = DEFAULT_PATH
MANIFEST_STATUS = 'development_material_release_not_training_admission'
REFERENCE_SPEED_CEILING_RAD_S = 2.
MAX_FILE_BYTES = 64 * 1024 * 1024


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _pairs(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f'Duplicate JSON key: {name}')
        result[name] = value
    return result


def _json(raw, label):
    def invalid(value):
        raise ValueError(f'Nonfinite JSON value in {label}: {value}')
    try:
        value = json.loads(raw, object_pairs_hook=_pairs, parse_constant=invalid)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f'Invalid JSON in {label}') from exc
    if not isinstance(value, dict):
        raise ValueError(f'{label} must be a JSON object')
    return value


def _relative_name(name):
    if not isinstance(name, str) or not name or '\\' in name:
        raise ValueError('Package file names must be nonempty relative POSIX paths')
    part = PurePosixPath(name)
    if part.is_absolute() or '..' in part.parts or name != part.as_posix() or name == '.':
        raise ValueError(f'Unsafe package path: {name}')
    return name


def _local_file(root, name):
    _relative_name(name)
    path = root / name
    if not path.resolve().is_relative_to(root) or path.is_symlink():
        raise ValueError(f'Package path escapes its root or is a symlink: {name}')
    if not path.is_file():
        raise ValueError(f'Missing package file: {name}')
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError(f'Package file is too large: {name}')
    return path


def _numeric(value, shape, label):
    """Reject strings and Boolean coercion before constructing numeric arrays."""
    try:
        raw = np.asarray(value, dtype=object)
        if raw.shape != shape or any(type(x) not in (int, float) for x in raw.flat):
            raise ValueError(f'{label}: real numeric values with shape {shape} required')
        result = raw.astype(float)
    except (TypeError, OverflowError) as exc:
        raise ValueError(f'{label}: finite real numeric values required') from exc
    if not np.isfinite(result).all():
        raise ValueError(f'{label}: finite real numeric values required')
    return result


def _catalog(raw, constraints):
    cfg = _json(raw, 'catalog')
    if type(cfg.get('schema_version')) is not int or cfg['schema_version'] != 1 \
            or cfg.get('admitted_to_training') is not False:
        raise ValueError('Candidate catalog schema 1 with admitted_to_training=false required')
    guards, specs, order = cfg.get('guards'), cfg.get('actions'), cfg.get('policy_order')
    if not isinstance(guards, dict) or not guards or not isinstance(specs, dict) or not specs:
        raise ValueError('Nonempty explicit guards and actions required')
    if not isinstance(order, list) or any(not isinstance(n, str) for n in order) \
            or len(order) != len(set(order)) or set(order) != set(specs) or 'wait' in specs:
        raise ValueError('policy_order must name every edge exactly once; wait is reserved ID 0')
    if not isinstance(cfg.get('initial_guard'), str) or cfg['initial_guard'] not in guards:
        raise ValueError('Catalog initial_guard must be an explicit node')
    bounds = model_joint_ranges_rad(constraints)
    nodes = {}
    for name, row in guards.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(row, dict):
            raise ValueError('Named guard objects required')
        values = row.get('qpos_rad')
        if not isinstance(values, list) or len(values) != 7 or any(type(x) not in (int, float) for x in values):
            raise ValueError(f'{name}: seven real CAD-radian coordinates required')
        q = _numeric(values, (7,), name)
        if np.any(q < bounds[:, 0]) or np.any(q > bounds[:, 1]):
            raise ValueError(f'{name}: node violates the packaged confirmed joint limits')
        nodes[name] = q.copy()
    actions, summaries = {}, []
    for index, name in enumerate(order, 1):
        row = specs[name]
        if not name.strip() or not isinstance(row, dict) or row.get('kind') not in ('attack', 'defence', 'transition'):
            raise ValueError(f'{name}: explicit edge kind required')
        start, end = row.get('start_guard'), row.get('end_guard')
        if not isinstance(start, str) or not isinstance(end, str) or start not in nodes or end not in nodes:
            raise ValueError(f'{name}: unknown explicit edge endpoint')
        times = row.get('knot_time_s')
        if not isinstance(times, list) or len(times) < 2:
            raise ValueError(f'{name}: at least two explicit knot times required')
        _numeric(times, (len(times),), f'{name} times')
        _numeric(row.get('knot_qpos'), (len(times), 7), f'{name} qpos')
        curve = C2JointSpline(times, row['knot_qpos'])
        if curve.knot_time_s[0] != 0. or not np.array_equal(curve.knot_qpos[0], nodes[start]) \
                or not np.array_equal(curve.knot_qpos[-1], nodes[end]):
            raise ValueError(f'{name}: curve must preserve exact zero-based endpoint identity')
        if np.any(curve.bezier_controls < bounds[:, 0]) or np.any(curve.bezier_controls > bounds[:, 1]):
            raise ValueError(f'{name}: whole C2 control envelope violates confirmed joint limits')
        peak = float(max(curve.analytic_extrema()['peak_speed_rad_s']))
        if peak > REFERENCE_SPEED_CEILING_RAD_S + 1e-9:
            raise ValueError(f'{name}: reference speed exceeds 2 rad/s')
        if 'coefficients' in row:
            coeff = _numeric(row['coefficients'], curve.coefficients.shape, f'{name} coefficients')
            if not np.array_equal(coeff, curve.coefficients):
                raise ValueError(f'{name}: stored coefficients differ from the exact C2 curve')
        actions[name] = GraphAction(name, start, end, curve)
        summaries.append(dict(action_id=index, name=name, kind=row['kind'], start_node=start,
            end_node=end, duration_s=curve.duration_s, peak_reference_speed_rad_s=peak,
            physical_material_validated=row.get('physical_material_validated', False),
            valid_in_tested_sequence_only=row.get('valid_in_tested_sequence_only', False)))
    aliases = cfg.get('exact_guard_aliases', {})
    if not isinstance(aliases, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                                           or v not in nodes for k, v in aliases.items()):
        raise ValueError('Guard aliases must refer to existing exact nodes')
    # Do not collapse equal-q nodes: they may have different certified predecessors.
    return cfg, nodes, actions, summaries


@dataclass(frozen=True)
class A1Release:
    """Immutable input bytes; every executor owns fresh mutable graph state."""
    package_dir: Path
    manifest_bytes: bytes
    file_bytes: tuple[tuple[str, bytes], ...]

    @property
    def manifest(self):
        return _json(self.manifest_bytes, 'manifest')

    @property
    def action_names(self):
        manifest = self.manifest
        cfg = _json(dict(self.file_bytes)[manifest['catalog_file']], 'catalog')
        return ('wait', *cfg['policy_order'])

    def inspect(self):
        self._assert_current()
        manifest = self.manifest
        contents = dict(self.file_bytes)
        constraints = load_joint_constraints(self.package_dir / manifest['joint_constraints_file'])
        cfg, nodes, _, rows = _catalog(contents[manifest['catalog_file']], constraints)
        return dict(status='verified_package_candidate_not_training_admission',
            package_dir=str(self.package_dir), manifest_status=manifest['status'],
            manifest_sha256=_sha(self.manifest_bytes), files_sha256=dict(manifest['files_sha256']),
            initial_node=cfg['initial_guard'], named_nodes=len(nodes),
            distinct_joint_poses=len({tuple(q) for q in nodes.values()}),
            policy_outputs_including_wait=len(rows) + 1, actions=rows,
            legacy_actor_compatible=False, admitted_to_training=False,
            native_steps=0, limitations=list(manifest.get('limitations', [])),
            scope='Hashes, exact endpoints, C2 joint limits and reference speed only; no new collision, interaction, policy or training certificate.')

    def _assert_current(self):
        current = load_release(self.package_dir)
        if current.manifest_bytes != self.manifest_bytes or current.file_bytes != self.file_bytes:
            raise ValueError('Package changed after loading; reload explicitly')

    def new_executor(self, initial_node=None):
        """Choose only an explicit node; never infer a bridge or readiness."""
        self._assert_current()
        manifest = self.manifest
        contents = dict(self.file_bytes)
        if _sha(DEFAULT_CONSTRAINTS_PATH.read_bytes()) != manifest['files_sha256'][manifest['joint_constraints_file']]:
            raise ValueError('Current executor constraints differ from the packaged constraints')
        constraints = load_joint_constraints(self.package_dir / manifest['joint_constraints_file'])
        cfg, nodes, actions, _ = _catalog(contents[manifest['catalog_file']], constraints)
        start = cfg['initial_guard'] if initial_node is None else initial_node
        if not isinstance(start, str) or start not in nodes:
            raise ValueError('initial_node must be an explicit packaged node, not an inferred alias')
        return SkillGraphExecutor(nodes, actions, start)


def load_release(package=DEFAULT_PACKAGE):
    """Verify a self-contained package, without reading original outputs paths."""
    root = Path(package).resolve()
    if not root.is_dir():
        raise ValueError('Release package must be a directory, not a legacy bank or catalog file')
    raw = _local_file(root, 'manifest.json').read_bytes()
    manifest = _json(raw, 'manifest')
    if type(manifest.get('schema_version')) is not int or manifest['schema_version'] != 1 \
            or manifest.get('status') != MANIFEST_STATUS \
            or manifest.get('legacy_actor_compatible') is not False or manifest.get('admitted_to_training') is not False:
        raise ValueError('Explicit development-only, legacy-incompatible manifest schema 1 required')
    hashes = manifest.get('files_sha256')
    if not isinstance(hashes, dict) or not hashes or len(hashes) > 128 \
            or any(not _digest(value) for value in hashes.values()) or 'manifest.json' in hashes:
        raise ValueError('files_sha256 must cover package content, excluding the manifest itself')
    required = [manifest.get(k) for k in ('catalog_file', 'cases_file', 'joint_constraints_file')]
    if any(not isinstance(p, str) or p not in hashes for p in required) or len(set(required)) != 3:
        raise ValueError('Distinct catalog, cases and joint-constraints files must be hash-bound')
    files = {}
    for name, expected in hashes.items():
        content = _local_file(root, name).read_bytes()
        if _sha(content) != expected:
            raise ValueError(f'Package SHA256 mismatch: {name}')
        files[name] = content
    inventory = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()} - {'manifest.json'}
    if inventory != set(hashes):
        raise ValueError('Every package file must be covered by files_sha256')
    provenance = manifest.get('source_fingerprints', {})
    if not isinstance(provenance, dict) or any(not isinstance(k, str) or not k or not _digest(v) for k, v in provenance.items()):
        raise ValueError('Invalid informational source_fingerprints')
    for name in provenance:
        _relative_name(name)  # Containment syntax only; never read external original outputs.
    cases = _json(files[manifest['cases_file']], 'cases')
    if type(cases.get('schema_version')) is not int or cases['schema_version'] != 1:
        raise ValueError('Cases schema 1 required; cases are evidence metadata, not an admission gate')
    constraints = load_joint_constraints(root / manifest['joint_constraints_file'])
    _, nodes, actions, _ = _catalog(files[manifest['catalog_file']], constraints)
    actual_counts = dict(catalog_edges=len(actions), named_nodes=len(nodes),
        unique_q=len({tuple(q) for q in nodes.values()}), policy_outputs=len(actions) + 1)
    counts = manifest.get('counts', {})
    if not isinstance(counts, dict) or any(k in counts and (type(counts[k]) is not int or counts[k] != value)
                                           for k, value in actual_counts.items()):
        raise ValueError('Manifest catalog counts do not match exact package content')
    return A1Release(root, raw, tuple(sorted(files.items())))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('inspect',))
    parser.add_argument('--package', type=Path, default=DEFAULT_PACKAGE)
    args = parser.parse_args(argv)
    try:
        result = load_release(args.package).inspect()
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
