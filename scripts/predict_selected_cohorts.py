"""Prepare full development cohorts for the frozen common marginal selection."""
from __future__ import annotations
import argparse,hashlib,json,os,shutil,sys,time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from otg.forecast import FrozenForecast
from otg.protocol import verify_final_protocol
ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['borg','bpi'],required=True);ap.add_argument('--device',default='cpu')
    ap.add_argument('--period',choices=['development','final'],default='development');a=ap.parse_args();torch.set_num_threads(4)
    if a.period=='final':verify_final_protocol(ROOT)
    data=ROOT/'data/processed'/f'{a.dataset}_joint_selected_v2';m=json.loads((data/'metadata.json').read_text())
    source=ROOT/m['source_dataset'] if a.period=='development' else ROOT/'data/processed'/f'{a.dataset}_final_cohorts_v1';sm=json.loads((source/'metadata.json').read_text())
    if sm['features']!=m['features']:raise ValueError('Frozen feature order changed')
    out=data/('cohorts' if a.period=='development' else 'final_cohorts');out.mkdir(exist_ok=False);scaler=np.load(data/'scaler.npz');rows=[];start=time.monotonic()
    if a.dataset=='bpi' and a.period=='development':
        q=pd.read_parquet(data/'queries.parquet');p=np.load(data/'cdf.npy');x=np.load(data/'features.npy')
        for t,frame in q[q.part=='validation'].groupby('cutoff_us'):
            name=f'cohort_{int(t)}';folder=out/name;folder.mkdir();ids=frame.index.to_numpy()
            frame.to_parquet(folder/'queries.parquet',index=False);np.save(folder/'cdf.npy',p[ids]);np.save(folder/'features.npy',x[ids]);rows.append({'folder':name,'cutoff_us':int(t),'n':len(frame)})
    else:
        history=None if a.period=='development' else ROOT/'data/processed'/f'{a.dataset}_final_histories_k16_v1'
        forecast=FrozenForecast(ROOT,a.dataset,m['selected_marginals'],m['horizons'],a.device,history_directory=history)
        with threadpool_limits(limits=4):
            for c in sm['cohorts']:
                if a.period=='development' and c['elapsed_day']<20:continue
                old=source/c['folder'];q=pd.read_parquet(old/'queries.parquet');x=np.load(old/'features.npy')
                indices=None
                if forecast.assets is not None:indices=np.load(forecast.history/f'{c["folder"]}_indices.npy',mmap_mode='r')
                p=forecast.predict(q,x,indices);folder=out/c['folder'];folder.mkdir();shutil.copy2(old/'queries.parquet',folder/'queries.parquet')
                np.save(folder/'cdf.npy',p);np.save(folder/'features.npy',np.clip((x-scaler['mean'])/scaler['scale'],-10,10).astype(np.float32))
                rows.append({'folder':c['folder'],'cutoff_us':c['cutoff_us'],'n':len(q),'elapsed_day':float(q.elapsed_day.iloc[0])});print(json.dumps({'cohort':c['folder'],'elapsed_s':time.monotonic()-start}),flush=True)
    for row in rows:row['cdf_sha256']=hashlib.sha256((out/row['folder']/'cdf.npy').read_bytes()).hexdigest()
    report={'config':vars(a),'cohorts':rows,'selected_metadata_sha256':hashlib.sha256((data/'metadata.json').read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'forecast_module_sha256':hashlib.sha256((ROOT/'otg/forecast.py').read_bytes()).hexdigest(),'elapsed_s':time.monotonic()-start}
    (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
