"""Bounded matched-node graph pilot on existing Borg development evidence.

This is exploratory and sampled, not a replacement for the original full-cohort
evaluation. Existing engineered features are shared by every comparator.
"""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from extension.prepare_pilots import OUT,REPORT,DAY


def main():
    out=OUT/"borg";out.mkdir(exist_ok=False)
    source=ROOT/"data/processed/borg_development_history_v1"
    hist=ROOT/"data/processed/borg_temporal_histories_k16_v1"
    q=pd.read_parquet(source/"queries.parquet")
    x=np.load(source/"features.npy",mmap_mode="r")
    history_index=np.load(hist/"queries_indices.npy",mmap_mode="r")
    event_time=np.load(hist/"event_times.npy",mmap_mode="r")
    event_type=np.load(hist/"event_categories.npy",mmap_mode="r")
    event_value=np.load(hist/"event_values.npy",mmap_mode="r")
    bounds=np.array([0,14,17,20,22])*DAY+600
    rows=[];xs=[];ys=[];rs=[];ts=[];es=[];hs=[]
    times=np.sort(q.cutoff_us.unique())[::6]
    rng=np.random.default_rng(20261021);offset=0
    for t in times:
        origin=t/1e6
        part=next((name for k,name in enumerate(["train","calibration","dependence","validation"])
                   if bounds[k]<=origin and origin+3600<bounds[k+1]),None)
        if part is None:continue
        ids=q.index[q.cutoff_us.eq(t)].to_numpy()
        ids=np.sort(rng.choice(ids,min(2048,len(ids)),replace=False))
        indices=np.asarray(history_index[ids,1:,-8:])
        valid=indices!=0
        if np.any(event_time[indices][valid]>t):raise ValueError("Future event in stored history")
        tokens=np.zeros((*indices.shape,5),np.float32)
        tokens[...,0]=np.log1p(np.maximum(0,(t-event_time[indices])/1e6/3600))*valid
        tokens[...,1]=event_type[indices,0]/10*valid
        tokens[...,2:4]=event_value[indices,:2]*valid[...,None]
        tokens[...,4]=valid
        xs.append(np.asarray(x[ids]));ys.append(q.loc[ids,["exit_1m","exit_5m","exit_15m","exit_60m"]].to_numpy(np.float32))
        rs.append(q.loc[ids,["job","machine"]].to_numpy(np.int64));ts.append(np.full(len(ids),origin));es.append(q.loc[ids,"task"].to_numpy());hs.append(tokens)
        rows.append({"origin":origin,"part":part,"offset":offset,"size":len(ids)});offset+=len(ids)
    np.savez_compressed(out/"cohorts.npz",x=np.concatenate(xs),y=np.concatenate(ys),relations=np.concatenate(rs),time=np.concatenate(ts),entity=np.concatenate(es),history=np.concatenate(hs))
    report={"source":"borg","stage":"retrospective development pilot on sampled queries", "features":int(xs[0].shape[1]),"queries":offset,"cohorts":rows,
            "relationships":["job","machine"],"bounds":bounds.tolist(),"horizons_seconds":[60,300,900,3600],
            "sample":"up to 2048 nodes uniformly from each existing sampled query cohort, every sixth stored origin; fixed seed20261021",
            "features_shared":"all existing 176 engineered history/context features; no claim of no relational facts in MLP",
            "history":"last8 observed job/machine events; lag/type/two observed continuous attributes/mask",
            "limitation":"original query sampling is state stratified; this is a matched sampled-node comparison, not a population estimate",
            "source_manifest_sha256":hashlib.sha256((source/"metadata.json").read_bytes()).hexdigest(),"code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/"manifest.json").write_text(json.dumps(report,indent=2)+"\n")
    (REPORT/"prepared_borg.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"queries":offset,"origins":len(rows)}),flush=True)


if __name__=="__main__":main()
