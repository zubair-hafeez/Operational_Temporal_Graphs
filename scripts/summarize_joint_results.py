"""Produce auditable score tables and paired time-block uncertainty from runs."""
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from otg.uncertainty import paired_moving_block_interval
ROOT=Path(__file__).resolve().parents[1]
METRICS=['fraction_crps','coverage_90','width_90_fraction','interval_score_90_fraction','buffer_cost_ratio3','buffer_cost_ratio9','trajectory_energy','simultaneous_alarm_fraction_crps']


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['borg','bpi'],required=True);ap.add_argument('--data-tag',default='selected_v2');ap.add_argument('--draws',type=int,default=512)
    ap.add_argument('--period',choices=['development','final'],default='development');a=ap.parse_args();frames=[];sources=[]
    filename='full_cohort_scores' if a.period=='development' else 'final_cohort_scores'
    for family in ['gaussian','commonshock']:
        for seed in range(3):
            p=ROOT/'runs'/f'joint_{a.dataset}_{a.data_tag}_{family}_seed{seed}'/f'{filename}_m{a.draws}.parquet'
            d=pd.read_parquet(p);d['family']=family;d['seed']=seed;frames.append(d);sources.append({'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    d=pd.concat(frames,ignore_index=True);out=ROOT/'reports'/f'summary_{a.dataset}_{a.data_tag}_{a.period}_m{a.draws}';out.mkdir(exist_ok=False)
    d.to_parquet(out/'all_scores.parquet',index=False)
    # First average horizons/groups and random seeds inside the same forecast
    # origin. Treating seeds or node counts as independent evidence is invalid.
    origin=d.groupby(['family','mode','cutoff_us','projection'])[METRICS].mean().reset_index()
    table=origin.groupby(['family','mode','projection'])[METRICS].mean().reset_index();table.to_csv(out/'projection_metrics.csv',index=False)
    macro=origin.groupby(['family','mode','cutoff_us'])[METRICS].mean().reset_index();macro.groupby(['family','mode'])[METRICS].mean().reset_index().to_csv(out/'macro_metrics.csv',index=False)
    uncertainty=[]
    for scope,frame in [('macro',macro)]+[(p,origin[origin.projection==p]) for p in origin.projection.unique()]:
        for family in ['gaussian','commonshock']:
            for metric in ['fraction_crps','buffer_cost_ratio9','trajectory_energy']:
                pivot=frame[frame.family==family].pivot(index='cutoff_us',columns='mode',values=metric).sort_index()
                for model,reference in [('global','independent'),('static','independent'),('static','global'),('context','static'),('static','static_permuted')]:
                    diff=(pivot[model]-pivot[reference]).to_numpy();mean_reference=float(pivot[reference].mean())
                    base={'projection':scope,'family':family,'metric':metric,'model':model,'reference':reference,'mean_difference':float(diff.mean()),'relative_improvement':float(-diff.mean()/mean_reference) if mean_reference else None,'n_origins':len(diff)}
                    for block in ([4,8] if a.dataset=='borg' else [14,28]):
                        row={**base,'block_length_origins':block}
                        if len(diff)<2*block:row.update(ci95_low=None,ci95_high=None,uncertainty_note='Fewer than two blocks; temporal evidence too short for this bootstrap interval.')
                        else:row.update(paired_moving_block_interval(diff,block,repeats=10000,seed=20261014))
                        uncertainty.append(row)
    pd.DataFrame(uncertainty).to_csv(out/'paired_block_intervals.csv',index=False)
    report={'config':vars(a),'sources':sources,'seeds':[0,1,2],'n_origins':int(d.cutoff_us.nunique()),'status':'complete',
        'interpretation':('Development scores are selected on development and are not unbiased final estimates. ' if a.period=='development' else 'Final scores use frozen development checkpoints without adaptation. ')+'Intervals are descriptive paired circular moving-block bootstrap intervals with seed-averaged forecast origins. No task-level IID or seed-as-new-data inference.',
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(table.to_string(index=False));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
