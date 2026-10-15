"""Choose a common, strong marginal forecast before dependence comparisons.

Compare every completed HGB feature/calibration arm and three-seed means of
each neural architecture. Never choose a lucky neural seed. This is explicitly
development selection, not a preregistered or unbiased performance estimate.
"""
from __future__ import annotations
import argparse,hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scripts.prepare_joint_development import sample_pairs
ROOT=Path(__file__).resolve().parents[1]


def load_choice(dataset,choice,fold,q):
    state=choice['state'];variant=choice['variant']
    if choice['kind']=='hgb':
        path=ROOT/'runs'/f'{dataset}_marginals_{choice["tag"]}_state{state}_{choice["arm"]}_seed0'/fold/'calibration_variants.npz'
        z=np.load(path);return z['query_ids'],z[variant]
    ensemble=[];ids=None
    for seed in range(3):
        path=ROOT/'runs'/f'temporal_{dataset}_{choice["architecture"]}_seed{seed}'/fold/'predictions.npz'
        z=np.load(path);index=z['query_ids'];keep=q.state.to_numpy()[index]==state
        if ids is None:ids=index[keep]
        elif not np.array_equal(ids,index[keep]):raise ValueError('Seed targets differ')
        ensemble.append(z[variant][keep])
    return ids,np.mean(ensemble,axis=0)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['borg','bpi'],required=True);a=ap.parse_args()
    version='history_v1' if a.dataset=='borg' else 'horizons_v1';source=ROOT/'data/processed'/f'{a.dataset}_development_{version}'
    meta=json.loads((source/'metadata.json').read_text());q=pd.read_parquet(source/'queries.parquet');x=np.load(source/'features.npy',mmap_mode='r')
    dev='development20_22' if a.dataset=='borg' else 'development_october';folds=['oof8_10','oof12_14','oof16_18',dev] if a.dataset=='borg' else ['oof_june','oof_july','oof_august',dev]
    horizons=[1,5,15,60] if a.dataset=='borg' else [1,3,7,14];unit='m' if a.dataset=='borg' else 'd';y=q[[f'exit_{h}{unit}' for h in horizons]].to_numpy()
    # Fail before creating any output if the declared neural comparison is
    # incomplete. All three architectures and seeds must be represented.
    for architecture in ['mixer','gru','mlp']:
        for seed in range(3):
            for fold in folds:
                p=ROOT/'runs'/f'temporal_{a.dataset}_{architecture}_seed{seed}'/fold/'manifest.json'
                if not p.exists() or json.loads(p.read_text())['status']!='complete':raise RuntimeError(f'Incomplete required baseline: {p}')
    candidates=[];chosen=[]
    for state in map(int,sorted(q.state.unique())):
        choices=[]
        for tag in (['v1','history_v1'] if a.dataset=='borg' else ['v1']):
            for arm in ['personal','global','relational']:
                for variant in ['raw_sorted','individual_platt_sorted','shared_population','shared_cutoff']:
                    choices.append({'kind':'hgb','tag':tag,'arm':arm,'variant':variant,'state':state})
        for architecture in ['mixer','gru','mlp']:
            for variant in ['raw','temperature']:choices.append({'kind':'temporal_ensemble','architecture':architecture,'seeds':[0,1,2],'variant':variant,'state':state})
        for choice in choices:
            ids,p=load_choice(a.dataset,choice,dev,q);frame=q.iloc[ids];w=frame.sampling_weight.to_numpy()/frame.groupby('cutoff_us').sampling_weight.transform('sum').to_numpy()
            choice['development_integrated_brier']=float(np.average(((p-y[ids])**2).mean(1),weights=w))
            choice['brier_by_horizon']=np.average((p-y[ids])**2,weights=w,axis=0).tolist()
            choice['prediction_mean_by_horizon']=np.average(p,weights=w,axis=0).tolist();choice['actual_mean_by_horizon']=np.average(y[ids],weights=w,axis=0).tolist()
        candidates+=choices;chosen.append(min(choices,key=lambda c:c['development_integrated_brier']))
    out=ROOT/'data/processed'/f'{a.dataset}_joint_selected_v2';out.mkdir(parents=True,exist_ok=False);records=[];probabilities=[];start=time.monotonic()
    for choice in chosen:
        for fold in folds:
            ids,p=load_choice(a.dataset,choice,fold,q);part=q.iloc[ids].copy();part['original_query_id']=ids;part['part']='validation' if fold==dev else 'train';part['forward_fold']=fold
            records.append(part);probabilities.append(p)
    joint=pd.concat(records,ignore_index=True);p=np.concatenate(probabilities);order=np.argsort(joint.cutoff_us,kind='stable');joint=joint.iloc[order].reset_index(drop=True);p=p[order]
    if joint.original_query_id.duplicated().any():raise ValueError('Duplicate targets')
    joint['query_id']=np.arange(len(joint));xx=np.asarray(x[joint.original_query_id.to_numpy()]);fit=joint.part=='train';mean=xx[fit].mean(0);scale=xx[fit].std(0);scale[scale<1e-5]=1
    joint.to_parquet(out/'queries.parquet',index=False,compression='zstd');np.save(out/'features.npy',np.clip((xx-mean)/scale,-10,10).astype(np.float32));np.save(out/'cdf.npy',p.astype(np.float32));np.savez(out/'scaler.npz',mean=mean,scale=scale)
    relations=['job','machine'] if a.dataset=='borg' else ['resource'];rng=np.random.default_rng(20261014);pair_counts={}
    for part in ['train','validation']:
        pairs=sample_pairs(joint[joint.part==part],relations,rng);np.save(out/f'pairs_{part}.npy',pairs);pair_counts[part]=len(pairs)
    report={'config':vars(a),'selected_marginals':chosen,'candidate_marginals':candidates,'selection_rule':'Equal-cutoff integrated Brier; all HGB history/calibration arms and three-seed neural ensembles. No best-seed selection. Development selection, final untouched.',
        'source_dataset':str(source.relative_to(ROOT)),'features':meta['features'],'feature_groups':meta['feature_groups'],'relations':relations,'horizons':horizons,'horizon_unit':'minutes' if a.dataset=='borg' else 'days',
        'n_queries':len(joint),'parts':joint.part.value_counts().to_dict(),'pairs':pair_counts,'source_metadata_sha256':hashlib.sha256((source/'metadata.json').read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'cdf_sha256':hashlib.sha256((out/'cdf.npy').read_bytes()).hexdigest(),'queries_sha256':hashlib.sha256((out/'queries.parquet').read_bytes()).hexdigest(),'elapsed_s':time.monotonic()-start}
    (out/'metadata.json').write_text(json.dumps(report,indent=2)+'\n');(ROOT/'reports'/f'{a.dataset}_selected_marginals_v2.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'selected':chosen,'queries':len(joint),'pairs':pair_counts}),flush=True)


if __name__=='__main__':main()
