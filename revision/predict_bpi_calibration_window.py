"""Recover September forecasts from the frozen current-version BPI ensemble."""
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from otg.forecast import FrozenForecast
from revision.common import ROOT,write_json,digest


def main():
    torch.set_num_threads(4);threadpool_limits(limits=4);start=time.monotonic()
    out=ROOT/'data/processed/revision_bpi_calibration_window_v1';out.mkdir(exist_ok=False)
    meta=json.loads((ROOT/'data/processed/bpi_joint_selected_v2/metadata.json').read_text());source=ROOT/meta['source_dataset'];history=ROOT/'data/processed/bpi_temporal_histories_k16_v1'
    q=pd.read_parquet(source/'queries.parquet');begin=pd.Timestamp('2016-09-01',tz='UTC').value//1000;end=pd.Timestamp('2016-10-01',tz='UTC').value//1000
    ids=np.flatnonzero((q.cutoff_us>=begin)&(q.cutoff_us+14*86400_000_000<=end));subset=q.iloc[ids].reset_index(drop=True)
    x=np.load(source/'features.npy',mmap_mode='r')[ids];indices=np.load(history/'queries_indices.npy',mmap_mode='r')[ids]
    forecast=FrozenForecast(ROOT,'bpi',meta['selected_marginals'],meta['horizons'],'cpu');p=forecast.predict(subset,x,indices)
    subset.to_parquet(out/'queries.parquet',index=False);np.save(out/'cdf.npy',p)
    write_json(out/'manifest.json',{'n':len(subset),'origins':subset.cutoff_us.nunique(),'interpretation':'Same model version as October and final. September window also used for early stopping/temperature in original training; not an independent validation score. CPU inference; original CPU/GPU discrepancy audit retained.','script_sha256':digest(__file__),'cdf_sha256':digest(out/'cdf.npy'),'elapsed_s':time.monotonic()-start})
    print(json.dumps({'n':len(subset),'origins':subset.cutoff_us.nunique(),'elapsed_s':time.monotonic()-start}),flush=True)


if __name__=='__main__':main()
