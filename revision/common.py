from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from otg.copula import StratifiedFactors

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def bpi_origins():
    data = ROOT/'data/processed/bpi_joint_selected_v2'
    q = pd.read_parquet(data/'queries.parquet')
    p = np.load(data/'cdf.npy')
    x = np.load(data/'features.npy')
    result = []
    for t, ids in q.groupby('cutoff_us', sort=True).indices.items():
        z=q.iloc[ids].reset_index(drop=True)
        result.append(dict(t=int(t), period='train' if z.part.iloc[0]=='train' else 'development', q=z, p=p[ids], x=x[ids]))
    cm=json.loads((data/'final_cohorts/manifest.json').read_text())
    for c in cm['cohorts']:
        f=data/'final_cohorts'/c['folder']
        result.append(dict(t=int(c['cutoff_us']),period='retrospective_final',q=pd.read_parquet(f/'queries.parquet'),p=np.load(f/'cdf.npy'),x=np.load(f/'features.npy')))
    for o in result:
        o['y']=o['q'][['exit_1d','exit_3d','exit_7d','exit_14d']].to_numpy(np.int8)
        assert np.all(o['q'].sampling_weight.to_numpy()==1), 'Full cohorts required'
        assert o['q']['case'].nunique()==len(o['q'])
    return sorted(result,key=lambda o:o['t'])


def frozen_loadings(origin, seed, dataset='bpi'):
    run=ROOT/'runs'/f'joint_{dataset}_selected_v2_gaussian_seed{seed}'
    m=json.loads((run/'manifest.json').read_text())
    q=origin['q'];relation_names=['resource'] if dataset=='bpi' else ['job','machine']
    relation=q[relation_names].to_numpy(np.int64)
    state=torch.as_tensor(np.searchsorted(m['states'],q.state.to_numpy()),dtype=torch.long)
    x=torch.as_tensor(origin['x']);available=torch.as_tensor(relation>=0,dtype=torch.float32)
    result={}
    with torch.no_grad():
        for mode in ['global','static']:
            model=StratifiedFactors(m['input_dim'],len(m['states']),m['n_relations'],mode=mode,hidden=m['hidden'])
            model.load_state_dict(torch.load(run/f'{mode}.pt',weights_only=True,map_location='cpu'));model.eval()
            result[mode]=model(x,state,available)[0].numpy()
    independent=np.zeros_like(result['global']);independent[:,-1]=1
    return {'independent':independent,**result}
