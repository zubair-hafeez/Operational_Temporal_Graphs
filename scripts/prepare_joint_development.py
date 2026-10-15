"""Assemble frozen marginal probabilities and observed incidence for joint fits."""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]


def sample_pairs(frame,relation_columns,rng,per_stratum=1000):
    result=[]
    for t,f in frame.groupby('cutoff_us',sort=True):
        ids=f.index.to_numpy()
        # Current query cohorts are outcome-independent; pair sampling never
        # uses the future event labels or fitted event probabilities.
        left=rng.choice(ids,per_stratum);right=rng.choice(ids,per_stratum)
        while np.any(left==right):
            equal=left==right;right[equal]=rng.choice(ids,equal.sum())
        result.extend(zip(left,right))
        for column in relation_columns:
            groups=[np.asarray(ix) for key,ix in f.groupby(column).groups.items() if key>=0 and len(ix)>=2]
            if not groups:continue
            for weighted in [False,True]:
                probabilities=np.array([len(g)*(len(g)-1) for g in groups],np.float64)
                selected=rng.choice(len(groups),per_stratum,p=probabilities/probabilities.sum() if weighted else None)
                for k in selected:result.append(tuple(rng.choice(groups[k],2,replace=False)))
    return np.asarray(result,np.int32)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['borg','bpi'],required=True);ap.add_argument('--history',action='store_true');a=ap.parse_args()
    version='history_v1' if a.history else 'horizons_v1';tag='history_v1' if a.history else 'v1'
    src=ROOT/'data/processed'/f'{a.dataset}_development_{version}';m=json.loads((src/'metadata.json').read_text())
    audits=json.loads((ROOT/'reports'/f'{a.dataset}_{tag}_calibration_audit.json').read_text())
    q=pd.read_parquet(src/'queries.parquet');x=np.load(src/'features.npy',mmap_mode='r');selected=[];records=[];probabilities=[]
    out=ROOT/'data/processed'/f'{a.dataset}_joint_{tag}';out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    for state in map(int,sorted(q.state.unique())):
        choices=[]
        for r in audits:
            if r['state']==state and r['fold'].startswith('development'):
                choices.extend((metric['cutoff_mean_integrated_brier'],r['arm'],name) for name,metric in r['metrics'].items())
        if len(choices)!=12:raise ValueError('Need all three completed marginal arms and four calibration variants')
        score,arm,variant=min(choices)
        chosen={'state':state,'arm':arm,'calibration_variant':variant,'development_cutoff_mean_integrated_brier':score}
        selected.append(chosen);run=ROOT/'runs'/f'{a.dataset}_marginals_{tag}_state{state}_{arm}_seed0'
        for folder in sorted(run.iterdir()):
            pred=folder/'calibration_variants.npz'
            if not pred.exists():raise RuntimeError(f'Incomplete forward fold: {folder}')
            z=np.load(pred);ids=z['query_ids'];part=q.iloc[ids].copy();part['original_query_id']=ids
            part['part']='validation' if folder.name.startswith('development') else 'train';part['forward_fold']=folder.name
            records.append(part);probabilities.append(z[variant])
    joint=pd.concat(records,ignore_index=True);p=np.concatenate(probabilities)
    order=np.argsort(joint.cutoff_us.to_numpy(),kind='stable');joint=joint.iloc[order].reset_index(drop=True);p=p[order]
    if joint.original_query_id.duplicated().any():raise ValueError('Overlapping fitting targets')
    joint['query_id']=np.arange(len(joint));xx=np.asarray(x[joint.original_query_id.to_numpy()])
    fit=joint.part=='train';mean=xx[fit].mean(0);scale=xx[fit].std(0);scale[scale<1e-5]=1
    xx=np.clip((xx-mean)/scale,-10,10).astype(np.float32)
    relations=['job','machine'] if a.dataset=='borg' else ['resource']
    joint.to_parquet(out/'queries.parquet',index=False,compression='zstd');np.save(out/'features.npy',xx);np.save(out/'cdf.npy',p.astype(np.float32));np.savez(out/'scaler.npz',mean=mean,scale=scale)
    rng=np.random.default_rng(20261014)
    pairs={}
    for part in ['train','validation']:
        pairs[part]=sample_pairs(joint[joint.part==part],relations,rng);np.save(out/f'pairs_{part}.npy',pairs[part])
    report={'config':vars(a),'selected_marginals':selected,'selection_rule':'Lowest development integrated Brier, equal horizons and equal cutoff weight, among three feature arms and four calibration variants.',
        'features':m['features'],'feature_groups':m['feature_groups'],'relations':relations,'n_queries':len(joint),'parts':joint.part.value_counts().to_dict(),'pairs':{k:len(v) for k,v in pairs.items()},
        'horizons':[1,5,15,60] if a.dataset=='borg' else [1,3,7,14],'horizon_unit':'minutes' if a.dataset=='borg' else 'days',
        'pair_sampling':'Per cutoff: 1000 uniform-node pairs; for each known relation, 1000 uniform-group and 1000 size*(size-1)-weighted group pairs. Includes cross-state pairs; no outcome/risk sampling.',
        'source_dataset':str(src.relative_to(ROOT)),'source_metadata_sha256':hashlib.sha256((src/'metadata.json').read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'queries_sha256':hashlib.sha256((out/'queries.parquet').read_bytes()).hexdigest(),'cdf_sha256':hashlib.sha256((out/'cdf.npy').read_bytes()).hexdigest(),'elapsed_s':time.monotonic()-start}
    (out/'metadata.json').write_text(json.dumps(report,indent=2)+'\n');(ROOT/'reports'/f'{a.dataset}_joint_{tag}_build.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['features','feature_groups']}),flush=True)


if __name__=='__main__':main()
