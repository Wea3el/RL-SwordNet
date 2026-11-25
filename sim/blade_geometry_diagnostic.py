"""Generate read-only profile/envelope evidence; does not step physics."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .blade_geometry import blade_profile_points, blade_cylinder_sections, blade_geometry_metadata
from .shinai import load_profile


def run(output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Keep prior diagnostic evidence')
    output.mkdir(parents=True)
    profile=load_profile();points=blade_profile_points(profile);sections=blade_cylinder_sections(profile)
    old_bounds=np.linspace(points[0,0],points[-1,0],4)
    old=[]
    for a,b in zip(old_bounds[:-1],old_bounds[1:]):
        rows=points[(points[:,0]>a)&(points[:,0]<b),1]
        radius=float(max(np.r_[np.interp([a,b],points[:,0],points[:,1]),rows]))
        old.append((float(a),float(b),radius))
    z=np.unique(np.r_[np.linspace(points[0,0],points[-1,0],7001),points[:,0]])
    actual=np.interp(z,points[:,0],points[:,1])
    def envelope(intervals):
        result=np.zeros_like(z)
        for a,b,radius in intervals:
            active=(z>=a-1e-14)&(z<=b+1e-14)
            result[active]=np.maximum(result[active],radius)
        return result
    old_r=envelope(old);new_r=envelope([(s.start_m,s.end_m,s.radius_m)for s in sections])
    body_length=(points[-1,0]-points[0,0])*1000
    x=(z-points[0,0])*1000
    fig,(ax,err)=plt.subplots(2,1,figsize=(12,6.4),sharex=True,gridspec_kw={'height_ratios':[2.2,1]})
    ax.fill_between(x,-1000*old_r,1000*old_r,color='#ef9b48',alpha=.23,label='Old: 3 collision cylinders')
    ax.fill_between(x,-1000*new_r,1000*new_r,color='#31a178',alpha=.30,label='New: 105 error-bounded cylinders')
    ax.plot(x,1000*actual,color='#18384b',lw=1.3,label='Visible axial profile (circle envelope)')
    ax.plot(x,-1000*actual,color='#18384b',lw=1.3)
    ax.set_ylim(-27,31);ax.set_ylabel('Radius from sword axis (mm)')
    ax.set_title('Collision silhouette versus visible blade profile — geometry only',loc='left',fontweight='bold')
    ax.legend(loc='upper right',frameon=False,fontsize=9,ncol=1)
    err.plot(x,1000*(old_r-actual),color='#dc7b24',lw=1.4,label='Old excess')
    err.plot(x,1000*(new_r-actual),color='#13825a',lw=1.1,label='New excess')
    err.axhline(.25,color='#13825a',ls='--',lw=.8,label='0.25 mm radial budget')
    err.set_ylabel('Radial excess (mm)');err.set_xlabel('Distance beyond blade-side guard (mm)')
    err.set_xlim(0,body_length);err.set_ylim(-.05,max(5.,float(np.max(old_r-actual))*1100))
    err.legend(loc='upper right',frameon=False,fontsize=9,ncol=3)
    for a in (ax,err):
        a.spines[['top','right']].set_visible(False);a.grid(alpha=.15)
    fig.text(.07,.015,'The 64-sided visible mesh is up to 0.0268 mm inside the circle envelope. '
             'Closed flat cylinders share end planes; no axial gaps. No defence success is implied.',fontsize=8.5,color='#4c5861')
    fig.tight_layout(rect=[0,.04,1,1]);image=output/'blade_collision_profile_comparison.png'
    fig.savefig(image,dpi=180);plt.close(fig)
    evidence={'schema_version':1,'status':'geometry_error_and_bounded_performance_evidence_only',
        'new_geometry':blade_geometry_metadata(profile),'old_three_cylinders':[
            {'start_m':a,'end_m':b,'radius_m':r}for a,b,r in old],
        'sampled_old_max_radial_excess_m':float(np.max(old_r-actual)),
        'sampled_new_max_radial_excess_m':float(np.max(new_r-actual)),
        'new_no_profile_undercoverage_at_samples':bool(np.min(new_r-actual)>=-1e-12),
        'sampled_profile_count':len(z),'image':str(image),
        'profile_error_method':'Per-linear-interval analytic bounds in blade_geometry plus 7001-point diagnostic plot; not CAD tessellation error.',
        'bounded_benchmarks_pre_vectorization':{
            'measurement_context':'Same process/environment, exact left_guard_positive in both slots; no contact at endpoint. Static benchmark only.',
            'new_duel_build_s':4.772180596999533,'new_forward_200_calls_s':.026264217998686945,
            'new_step_200_calls_s':.025191902001097333,'integrated_duration_s':.4,
            'new_three_scorers_static_200_steps_s':6.290019469000981,
            'old_three_scorers_static_200_steps_s':.19622610900114523,
            'scorer_failed':False,'scorer_optimization_pending_at_measurement':True,
            'warning':'Do not extrapolate zero-contact step timing to moving weapon contact or use this as defence validation.'},
        'source_sha256':{str(Path(__file__).resolve()):hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            str(Path(__file__).with_name('blade_geometry.py').resolve()):hashlib.sha256(Path(__file__).with_name('blade_geometry.py').read_bytes()).hexdigest()},
        'admitted_to_training':False,'prior_contact_or_defence_results_inherited':False}
    (output/'geometry_audit01.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:evidence[k]for k in ('sampled_old_max_radial_excess_m','sampled_new_max_radial_excess_m','image')},indent=2))
    return evidence


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    run(p.parse_args().output)
