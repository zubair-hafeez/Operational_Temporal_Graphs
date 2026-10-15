import json
import numpy as np
import pandas as pd
import torch
from revision.common import ROOT
from revision.calibration import DAY_US
from revision.fit_bpi2012_marginals import boundary
from otg.copula import StratifiedFactors


def origins_2012(period='development',calibration='identity'):
    source=ROOT/'data/processed'/f'revision_bpi2012_{period}_v1';marg=ROOT/'runs'/('revision_bpi2012_marginals_v1' if period=='development' else 'revision_bpi2012_final_predictions_v1')
    q=pd.read_parquet(source/'queries.parquet');p=np.load(marg/f'selected_{calibration}_cdf.npy');raw=np.load(marg/'selected_identity_cdf.npy');origins=[]
    for t,ids in q.groupby('cutoff_us').indices.items():
        if period=='final':
            if t<boundary('2012-02-01'):continue
            role='final'
        elif boundary('2011-12-01')<=t and t+14*DAY_US<=boundary('2012-01-01'):role='train'
        elif boundary('2012-01-01')<=t and t+14*DAY_US<=boundary('2012-02-01'):role='development'
        else:continue
        z=q.iloc[ids].reset_index(drop=True);origins.append({'t':int(t),'period':role,'q':z,'p':p[ids],'raw_reference':raw[ids],'y':z[['exit_1d','exit_3d','exit_7d','exit_14d']].to_numpy(np.int8)})
    return origins


def loads_2012(origin,seed,calibration):
    out=ROOT/'runs'/f'revision_bpi2012_joint_v1_seed{seed}';q=origin['q'];x=torch.zeros((len(q),1));st=torch.zeros(len(q),dtype=torch.long);av=torch.as_tensor((q.resource.to_numpy()>=0)[:,None],dtype=torch.float32);result={}
    with torch.no_grad():
        for mode in ['global','static']:
            model=StratifiedFactors(1,1,1,mode=mode);model.load_state_dict(torch.load(out/f'{calibration}_{mode}.pt',weights_only=True,map_location='cpu'));model.eval();result[mode]=model(x,st,av)[0].numpy()
    independent=np.zeros_like(result['global']);independent[:,-1]=1
    return {'independent':independent,**result}
