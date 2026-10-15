"""Compare exported temporal-model inference with saved GPU predictions."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from otg.forecast import FrozenForecast
ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--device',default='cuda:1');a=ap.parse_args();torch.set_num_threads(4)
    q=pd.read_parquet(ROOT/'data/processed/bpi_development_horizons_v1/queries.parquet');x=np.load(ROOT/'data/processed/bpi_development_horizons_v1/features.npy',mmap_mode='r')
    ix=np.load(ROOT/'data/processed/bpi_temporal_histories_k16_v1/queries_indices.npy',mmap_mode='r')
    choice={'kind':'temporal_ensemble','architecture':'mixer','seeds':[0,1,2],'state':1,'variant':'raw'}
    zs=[np.load(ROOT/f'runs/temporal_bpi_mixer_seed{s}/development_october/predictions.npz') for s in range(3)];ids=zs[0]['query_ids'][:1000];expected=np.mean([z['raw'][:1000] for z in zs],axis=0)
    rows=[]
    with threadpool_limits(limits=4):
        for device in ['cpu',a.device]:
            f=FrozenForecast(ROOT,'bpi',[choice],[1,3,7,14],device);p=f.predict(q.iloc[ids],x[ids],ix[ids]);delta=np.abs(p-expected)
            rows.append({'device':device,'max_abs_error':float(delta.max()),'mean_abs_error':float(delta.mean())})
            if device==a.device and delta.max()>2e-5:raise ValueError(f'GPU round trip mismatch: {rows[-1]}')
    report={'audit':'Exported inference vs saved development predictions; BPI Mixer three-seed mean, first1000 targets.','results':rows,'policy':'Use CUDA for temporal-model cohort inference, matching the declared training environment. CPU and GPU trigonometric time encodings can differ at large float32 time arguments.'}
    (ROOT/'reports/temporal_inference_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
