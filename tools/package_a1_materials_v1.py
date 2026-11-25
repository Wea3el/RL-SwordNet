"""Package authored A1 numeric materials, not raw motion data or a trainer.

The package has no runtime dependency on ignored outputs. Original evidence is
represented by checked fingerprints and compact summaries, not redistributed
native traces. Existing sources and failed experiments are never changed.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from sim.motion_spline import C2JointSpline

SOURCE=ROOT/'outputs/pose_set_v1'
DEFAULT=ROOT/'assets/a1_pose_bank_v1'


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    p=Path(p)
    if p.exists():raise FileExistsError('Keep existing package: '+str(p))
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def relative(p):return str(Path(p).resolve().relative_to(ROOT))
def verify(folder,hashes):
    for n,h in hashes.items():
        if sha(folder/n)!=h:raise ValueError('Evidence hash mismatch: '+str(folder/n))


def path_strings(value,where=''):
    found=[]
    if isinstance(value,dict):
        for k,v in value.items():
            if isinstance(k,str) and (k.startswith('/') or k.startswith('outputs/')):found.append((where+'.<key>',k))
            found.extend(path_strings(v,where+'.'+str(k)))
    elif isinstance(value,list):
        for i,v in enumerate(value):found.extend(path_strings(v,where+'['+str(i)+']'))
    elif isinstance(value,str) and (value.startswith('/') or value.startswith('outputs/')):found.append((where,value))
    return found


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT)
    args=parser.parse_args();out=args.output.resolve()
    if out.exists():raise FileExistsError('Use a new package directory')
    files={n:SOURCE/n for n in ('six_attack_materials_v3.json','six_defence_materials_v2.json',
        'integrated_catalog_candidate_v4.json','integrated_catalog_structure_v4.json','integrated_static_matrix_v4.json')}
    attack,defence,catalog=(read(files[n]) for n in ('six_attack_materials_v3.json','six_defence_materials_v2.json','integrated_catalog_candidate_v4.json'))
    assert attack['scoring']==defence['scoring']==catalog['scoring']
    assert attack['common_physical_identity']==defence['common_physical_identity']
    provenance=dict((relative(p),sha(p)) for p in files.values())
    inspected=dict(files)
    for rows in catalog['action_provenance'].values():
        for row in rows:
            p=ROOT/row['path'];assert sha(p)==row['sha256'];provenance[row['path']]=row['sha256'];inspected[row['path']]=p
    core_keys=('kind','family','start_guard','end_guard','knot_time_s','knot_qpos','target_region',
        'candidate_pass','physical_material_validated','valid_in_tested_sequence_only','measured_ready_required',
        'readiness','supported_sequence','scope','candidate_stage')
    packed={k:copy.deepcopy(catalog[k]) for k in ('schema_version','guards','scoring','initial_guard','policy_order','exact_guard_aliases')}
    packed.update(status='development_material_release_not_training_admission',actions={},admitted_to_training=False,
        action_mask_note=catalog['action_mask_note'],legacy_actor_compatible=False)
    for name,a in catalog['actions'].items():
        row={k:copy.deepcopy(a[k]) for k in core_keys if k in a}
        row['admitted_to_training']=False
        row['evidence_level']=('geometry_and_ideal_candidate_not_native_admitted' if a['kind']=='transition' else
            'tested_sequence_only' if a.get('valid_in_tested_sequence_only') else 'matched_nominal_material')
        old=C2JointSpline(a['knot_time_s'],a['knot_qpos']);new=C2JointSpline(row['knot_time_s'],row['knot_qpos'])
        assert np.array_equal(old.coefficients,new.coefficients)
        row['coefficients']=new.coefficients.tolist();packed['actions'][name]=row
    assert not path_strings(packed),'Runtime catalog must not depend on outputs/absolute paths'
    evidence=[];cases={};native_files={}
    for kind,bank in [('attack',attack),('defence',defence)]:
        for name,a in bank['actions'].items():
            proof=ROOT/a['proof'];assert sha(proof)==a['proof_sha256'];pr=read(proof)
            provenance[a['proof']]=a['proof_sha256'];inspected[a['proof']]=proof
            native=ROOT/a['baseline_native' if kind=='attack' else 'native']
            hashes=(a['native_sha256'] if kind=='attack' else pr['defended_triangle_audit']['native_trace_source_sha256'])
            verify(native,hashes)
            summary=read(native/'summary.json');assert summary['status']=='completed' and not summary['failure']
            for fn,h in hashes.items():native_files[relative(native/fn)]=dict(sha256=h,bytes=(native/fn).stat().st_size)
            entry=dict(material=name,kind=kind,proof_sha256=a['proof_sha256'],native_file_fingerprints=hashes,
                original_archive_directory=relative(native),full_window_s=float(a.get('full_observation_s',a.get('full_recorded_duration_s'))),
                physical_material_validated=True,admitted_to_training=False,
                proof_status=pr.get('status','native_triangle_audit'),
                qualification_scope='successful waiting-control attack' if kind=='attack' else 'matched nominal attack and complete defence/recovery')
            if kind=='attack':entry['actual_entries']=a['actual_entries']
            else:
                assert pr['status']=='qualified' and pr['successful_physical_defence']
                entry.update(first_pre_hit_contact_s=pr.get('first_pre_hit_weapon_contact_s'),
                    pre_hit_impulse_ns=pr.get('pre_hit_weapon_impulse_ns'),
                    all_regions_checked=pr.get('all_regions_checked'),
                    validated_attack_family=a['validated_attack_family_only'],
                    validated_attack_start_s=a['validated_attack_start_s'],validated_defence_start_s=a['validated_defence_start_s'])
                refs=native/'references.npz';programs={}
                with np.load(refs,allow_pickle=False) as z:
                    for slot,start in [('a1',a['validated_attack_start_s']),('a2',a['validated_defence_start_s'])]:
                        ts=z[slot+'_knot_time_s'];q=z[slot+'_knot_qpos'];c=C2JointSpline(ts,q)
                        assert np.array_equal(c.coefficients,z[slot+'_coefficients'])
                        programs[slot]=dict(start_s=start,curve=dict(knot_time_s=ts.tolist(),knot_qpos=q.tolist(),coefficients=c.coefficients.tolist()))
                cases[name]=dict(duration_s=a['full_recorded_duration_s'],attacker_slot='a1',defender_slot='a2',
                    initial_qpos={s:p['curve']['knot_qpos'][0] for s,p in programs.items()},programs=programs,
                    baseline_defender='hold exact initial q throughout',native_references_sha256=sha(refs),
                    proof_sha256=a['proof_sha256'],scope='Exact historical pair fixture, not an autonomous reaction policy',
                    admitted_to_training=False)
                provenance[relative(refs)]=sha(refs)
            evidence.append(entry)
    static=read(files['integrated_static_matrix_v4.json']);structure=read(files['integrated_catalog_structure_v4.json'])
    assert static['clear_count']==169 and static['total_pairs']==169
    assert structure['source_catalog_sha256']==sha(files['integrated_catalog_candidate_v4.json'])
    paths=[]
    for name,p in inspected.items():
        for where,s in path_strings(read(p)):
            target=Path(s) if s.startswith('/') else ROOT/s
            paths.append(dict(document=name,field=where,original_path=s,is_absolute=s.startswith('/'),exists=target.exists()))
    missing=[r for r in paths if not r['exists']]
    uniqueq=len({tuple(v['qpos_rad']) for v in packed['guards'].values()})
    counts=dict(attack_families=len(attack['actions']),nominal_defence_materials=len(defence['actions']),
        catalog_edges=len(packed['actions']),named_nodes=len(packed['guards']),unique_q=uniqueq,
        policy_outputs=len(packed['policy_order'])+1,transition_edges=sum(a['kind']=='transition' for a in packed['actions'].values()),
        closed_defence_edges=sum(a['kind']=='defence' and not a.get('valid_in_tested_sequence_only') for a in packed['actions'].values()),
        sequence_bound_defence_edges=sum(a['kind']=='defence' and a.get('valid_in_tested_sequence_only',False) for a in packed['actions'].values()))
    save(out/'catalog.json',packed)
    save(out/'cases.json',dict(schema_version=1,cases=cases,runtime_config=attack['common_physical_identity']['runtime_config'],
        scoring=attack['scoring'],native_steps=0,admitted_to_training=False))
    save(out/'physical_contract.json',dict(schema_version=1,**attack['common_physical_identity'],
        torque_caps_nm=read(ROOT/'config/assembly.json')['runtime']['torque_caps_nm'],
        command_contract='Current direct spline target -> shared position servo; no A2 lag governor deployed',
        admitted_to_training=False))
    save(out/'evidence_summary.json',dict(schema_version=1,counts=counts,materials=evidence,
        static_named_node_pairs=dict(clear=169,total=169),
        scope='Six attack threats and six nominal matched defences; 18 transitions remain candidate-level. Stage edges inherit only their tested sequence, not arbitrary switches.',
        raw_native_traces_included=False,admitted_to_training=False))
    save(out/'dependency_audit.json',dict(schema_version=1,inspected_json_documents=list(inspected),
        discovered_path_count=len(paths),absolute_path_count=sum(p['is_absolute'] for p in paths),
        missing_paths=missing,paths=paths,raw_native_archive=native_files,
        raw_native_archive_bytes=sum(r['bytes'] for r in native_files.values()),
        runtime_catalog_outputs_dependencies=0,raw_replay_self_contained=False,
        note='Path inventory is for selected catalog/provenance/proof documents, not all historical experiments. Original archive paths are audit metadata only.'))
    source_notes={
        'left_cut_pair': ['outputs/pose_set_v1/attacks/fresh_branch_left_v2d.py',
            'outputs/pose_set_v1/attacks/soft_margin_left_v2e.py',
            'outputs/pose_set_v1/attacks/controlled_withdraw_left_v2f.py'],
        'right_cut_pair': ['tools/compact_bank_pose_search.py','sim/multiguard_geometry.py',
            'outputs/pose_set_v1/right_retracted_v3.py'],
        'center_cut_and_thrust': ['config/a1_triangle_motion_v1.json','tools/compact_bank_pose_search.py',
            'sim/multiguard_geometry.py'],
        'solver_helpers_only_not_motion_input': ['plan1_skills/matching.py'],
        'excluded_upstream_motion_notice': ['motion_skills/plan1/SOURCES.md']}
    save(out/'source_audit.json',dict(schema_version=1,
        technical_conclusion='Selected numeric robot curves are authored task-space/IK/C2 candidates, not copied upstream ASE motion samples.',
        basis='Left pair uses five new Cartesian targets; compact endpoint paths use explicit target optimization; right arcs and vertical cut use generated Cartesian grip/axis paths. Historical seed poses and generic IK helpers are not a copied motion sequence.',
        source_paths_and_hashes={group:{p:sha(ROOT/p) for p in paths} for group,paths in source_notes.items()},
        excluded=['motion_skills raw/derived motion arrays','outputs ASE download/retarget caches','raw native traces','videos'],
        limitation='Technical provenance review, not a grant of third-party motion rights; retain the existing upstream notice. No original or derived ASE motion file is distributed in this package.'))
    constraints=ROOT/'config/joint_constraints_observed.json';(out/'joint_constraints_observed.json').write_bytes(constraints.read_bytes())
    provenance[relative(constraints)]=sha(constraints);provenance[relative(Path(__file__))]=sha(Path(__file__))
    save(out/'manifest.json',dict(schema_version=1,status='development_material_release_not_training_admission',
        catalog_file='catalog.json',cases_file='cases.json',joint_constraints_file='joint_constraints_observed.json',
        physical_contract_file='physical_contract.json',evidence_file='evidence_summary.json',
        files_sha256={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()},source_fingerprints=provenance,
        counts=counts,legacy_actor_compatible=False,admitted_to_training=False,
        limitations=['Not a trained A1 policy or complete adversarial graph certificate.',
            'Fast left down-to-up reversal and lower105 repair failed; those motions are not added to this catalog.',
            'Transition edges are explicit screened candidates, not all independently native-admitted.',
            'Whole-window nominal evidence does not imply every timing, swapped slot or arbitrary A2 trajectory is defended.',
            'No raw ASE/Reallusion motion files, native traces or videos are included.'],
        data_source_scope='Team-authored task-space IK/C2 robot candidates; solver helpers do not import upstream motion data. See source audit supplied with release.',
        data_files_self_contained=True,requires_repository_simulator=True,raw_replay_self_contained=False))
    print(json.dumps(dict(output=relative(out),counts=counts,manifest_sha256=sha(out/'manifest.json'),
        package_bytes=sum(p.stat().st_size for p in out.iterdir()),missing_dependency_paths=len(missing)),indent=2))


if __name__=='__main__':main()
