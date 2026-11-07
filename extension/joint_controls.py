"""Three-seed joint evaluation with direct counts, exact scores, and block CIs."""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS","4")
os.environ.setdefault("OPENBLAS_NUM_THREADS","4")
import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from extension.run_pilots import projections, scenario, count_score, crps
from revision.counts import fit_dm, parameters, sample_dm
from revision.exact_counts import exact_projected_scores
from revision.joint_profile import profile_scores
from otg.scores import ensemble_crps


def group_records(p,y,r):
    result={}
    groups=projections(r)
    for kind in sorted(set(k for k,_ in groups)):
        members=[ix for k,ix in groups if k==kind]
        n=np.array([len(ix) for ix in members])
        mean=np.stack([p[ix].mean(0) for ix in members])
        prob=np.diff(np.column_stack([np.zeros(len(n)),mean,np.ones(len(n))]),axis=1).clip(1e-6)
        prob/=prob.sum(-1,keepdims=True)
        truth=np.stack([y[ix].sum(0) for ix in members])
        target=np.diff(np.column_stack([np.zeros(len(n)),truth,n]),axis=1).astype(int)
        result[kind]={"n":n,"prob":prob,"truth":truth,"target":target,"members":members}
    return result


def group_scores(samples,records,p,fair=False):
    summaries=[]
    for kind,g in records.items():
        s=samples[kind];n=g["n"];truth=g["truth"]
        fraction=s/n[None,:,None]
        lo,hi=np.quantile(s,[.05,.95],axis=0)
        reserve=np.quantile(n[None,:,None]-s,.9,axis=0)
        actual=n[:,None]-truth
        mean=np.stack([p[ix].sum(0) for ix in g["members"]])
        variance=np.stack([(p[ix]*(1-p[ix])).sum(0) for ix in g["members"]])
        scale=np.sqrt(1+variance) if fair else np.sqrt(np.maximum(1,variance))
        profile=profile_scores(s,truth,n,mean,scale)
        count_crps=ensemble_crps(fraction,truth/n[:,None]) if fair else crps(fraction,truth/n[:,None])
        summaries.append({"projection":kind,"count_crps":float(count_crps.mean()),
                          "coverage90":float(((truth>=lo)&(truth<=hi)).mean()),
                          "width90":float(((hi-lo)/n[:,None]).mean()),
                          "reserve_loss":float(((9*np.maximum(actual-reserve,0)+np.maximum(reserve-actual,0))/n[:,None]).mean()),
                          **{k:float(np.mean([row[k] for row in profile])) for k in ["joint_profile_energy","joint_variogram_fair","peak_slowdown_crps"]}})
    return summaries


def bootstrap_difference(table,reference,candidate,block_days,seed=811,repeats=2000):
    wide=table.pivot(index="origin",columns="model",values="count_crps").sort_index()
    d=(wide[candidate]-wide[reference]).to_numpy()
    if len(d)<3:return {"origins":len(d),"mean_difference":float(d.mean())}
    cadence=np.median(np.diff(wide.index.to_numpy()))
    length=min(len(d),max(1,int(round(block_days*86400/cadence))))
    rng=np.random.default_rng(seed)
    starts=rng.integers(len(d),size=(repeats,int(np.ceil(len(d)/length))))
    idx=(starts[:,:,None]+np.arange(length))%len(d)
    means=d[idx.reshape(repeats,-1)[:,:len(d)]].mean(-1)
    return {"origins":len(d),"block_days":block_days,"block_origins":length,"mean_difference":float(d.mean()),
            "interval95":np.quantile(means,[.025,.975]).tolist(),"warning":"finite-period exploratory interval; not independent organizational replication"}


def main(a):
    start=time.monotonic()
    src=ROOT/"data/processed/extension_20261021"/a.source
    out=ROOT/"runs/extension_20261021"/f"{a.source}_joint_{a.tag}"
    out.mkdir(parents=True,exist_ok=False)
    code=out/"code";code.mkdir()
    for path in [Path(__file__),ROOT/"extension/run_pilots.py",ROOT/"extension/models.py"]:shutil.copy2(path,code/path.name)
    m=json.loads((src/"manifest.json").read_text());z=np.load(src/"cohorts.npz")
    y,r=z["y"],z["relations"]
    cohorts=m["cohorts"]
    for c in cohorts:c["ix"]=np.arange(c["offset"],c["offset"]+c["size"])
    parts={p:[c for c in cohorts if c["part"]==p] for p in ["calibration","dependence","validation"]}
    paths=[ROOT/"runs/extension_20261021"/f"{a.source}_seed{seed}_{a.marginal_tag}" for seed in a.seeds]
    sets=[np.load(path/"predictions.npz") for path in paths]
    names=sorted(set.intersection(*(set(s.files) for s in sets)))
    ensembles={name:np.mean([s[name] for s in sets],axis=0) for name in names}
    marginal_rows=[]
    for name,p in ensembles.items():
        for part in ["calibration","validation"]:
            for c in parts[part]:
                ix=c["ix"];marginal_rows.append({"model":name,"part":part,"origin":c["origin"],"brier":float(((p[ix]-y[ix])**2).mean())})
    marginal=pd.DataFrame(marginal_rows);marginal.to_csv(out/"marginal_ensemble.csv",index=False)
    selected=marginal.loc[marginal.part.eq("calibration")].groupby("model").brier.mean().idxmin()
    p=ensembles[selected];np.save(out/"selected_cdf.npy",p)
    groups={c["origin"]:group_records(p[c["ix"]],y[c["ix"]],r[c["ix"]]) for c in cohorts if c["part"] in parts}
    def objective(w,df=None):
        scores=[]
        for i,c in enumerate(parts["dependence"]):
            ix=c["ix"];sample=scenario(p[ix],r[ix],w,a.fit_draws,9100+i,df)
            scores.append(count_score(sample,y[ix],projections(r[ix])))
        return np.mean(scores)
    empty=[0.]*(r.shape[1]+1);grid=[0,.05,.15,.3,.5,.7]
    gw=min(([v]+empty[1:] for v in grid),key=objective)
    models={"independent":(empty,None,False),"gaussian_global":(gw,None,False)}
    for j,rel in enumerate(m["relationships"]):
        for global_on in [False,True]:
            candidates=[]
            for v in grid:
                w=gw.copy() if global_on else empty.copy();w[j+1]=v
                if sum(w)<=.9:candidates.append(w)
            models[f"gaussian_{'global_' if global_on else ''}{rel}"]=(min(candidates,key=objective),None,False)
    w=gw.copy()
    for sweep in range(2):
        for j in range(len(w)):
            candidates=[]
            for v in grid:
                trial=w.copy();trial[j]=v
                if sum(trial)<=.9:candidates.append(trial)
            w=min(candidates,key=objective)
    models["gaussian_typed"]=(w,None,False)
    models["gaussian_shuffled"]=(w,None,True)
    candidates=[(objective(v,df),v,df) for v in [gw,w] for df in [4,8,16]]
    _,tw,tdf=min(candidates,key=lambda v:v[0]);models["student_t"]=(tw,tdf,False)
    for name,weights in [("student_t_global",gw),("student_t_typed",w)]:
        df=min([4,8,16],key=lambda d:objective(weights,d))
        models[name]=(weights,df,False)
    print(json.dumps({"marginal":selected,"joint_configs":models}),flush=True)
    dm={}
    for bias in [False,True]:
        name="direct_bias_dm" if bias else "direct_dm"
        dm[name]={}
        for kind in sorted({key for c in parts["dependence"] for key in groups[c["origin"]]}):
            records=[groups[c["origin"]][kind] for c in parts["dependence"] if kind in groups[c["origin"]]]
            dm[name][kind]=fit_dm(records,bias)
    rows=[]
    for name in list(models)+list(dm)+["direct_residual"]:
        for c in parts["validation"]:
            ix=c["ix"];g=groups[c["origin"]]
            for seed in [0,1,2]:
                rng=np.random.default_rng(1200+seed)
                if name in models:
                    weights,df,shuffled=models[name]
                    sample=scenario(p[ix],r[ix],weights,a.draws,1400+seed,df,shuffled)
                    counts={kind:np.stack([sample[:,idx].sum(1) for idx in v["members"]],axis=1) for kind,v in g.items()}
                elif name in dm:
                    counts={kind:sample_dm(v["n"],parameters(v["prob"],dm[name][kind]),a.draws,rng).cumsum(-1)[:,:,:4] for kind,v in g.items()}
                else:
                    counts={}
                    for kind,v in g.items():
                        residual=[groups[d["origin"]][kind]["truth"]/groups[d["origin"]][kind]["n"][:,None]-groups[d["origin"]][kind]["prob"].cumsum(-1)[:,:4]
                                  for d in parts["dependence"] if kind in groups[d["origin"]]]
                        sampled=[]
                        for _ in range(a.draws):
                            source=residual[rng.integers(len(residual))]
                            sampled.append(source[rng.integers(len(source),size=len(v["n"]))])
                        q=np.clip(v["prob"].cumsum(-1)[None,:,:4]+np.array(sampled),0,1)
                        counts[kind]=np.rint(np.maximum.accumulate(q,axis=-1)*v["n"][None,:,None]).astype(int)
                for scores in group_scores(counts,g,p[ix]):rows.append({"model":name,"seed":seed,"origin":c["origin"],**scores})
            if name in dm:
                exact=exact_projected_scores(g,{kind:parameters(v["prob"],dm[name][kind]) for kind,v in g.items()})
                for kind in g:
                    subset=[s for s in exact if s["projection"]==kind]
                    rows.append({"model":name+"_exact","seed":-1,"origin":c["origin"],"projection":kind,
                                 "count_crps":np.mean([s["fraction_crps"] for s in subset]),"coverage90":np.mean([s["coverage_90"] for s in subset]),
                                 "width90":np.mean([s["width_90_fraction"] for s in subset]),"reserve_loss":np.mean([s["buffer_cost_ratio9"] for s in subset])})
        print(json.dumps({"joint_scored":name}),flush=True)
    table=pd.DataFrame(rows);table.to_csv(out/"projection_scores.csv",index=False)
    metrics=["count_crps","coverage90","width90","reserve_loss","joint_profile_energy","joint_variogram_fair","peak_slowdown_crps"]
    origin=table.groupby(["model","origin"])[metrics].mean().reset_index();origin.to_csv(out/"origin_scores.csv",index=False)
    macro=origin.groupby("model")[metrics].mean();macro.to_csv(out/"macro_scores.csv")
    intervals=[]
    for ref,candidate in [("gaussian_global","gaussian_typed"),("gaussian_shuffled","gaussian_typed"),("direct_bias_dm_exact","gaussian_typed")]:
        for block in [1,3,7]:intervals.append({"reference":ref,"candidate":candidate,**bootstrap_difference(origin,ref,candidate,block)})
    report={"config":vars(a),"selected_marginal":selected,"marginal_selection":"three-seed mean CDF, calibration-origin mean Brier", "joint_models":models,"direct_fits":dm,
            "macro":macro.to_dict("index"),"paired_intervals":intervals,"elapsed_seconds":time.monotonic()-start,
            "stage":"development validation only; no fresh confirmation", "data_manifest_sha256":hashlib.sha256((src/"manifest.json").read_bytes()).hexdigest(),
            "direct_limitations":"separately sampled groups need not agree across overlapping projections; residual trajectories clipped and rounded",
            "model_source_hashes":{str(path.relative_to(ROOT)):hashlib.sha256((path/"report.json").read_bytes()).hexdigest() for path in paths}}
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"macro":report["macro"],"elapsed_seconds":report["elapsed_seconds"]}),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--source",required=True);ap.add_argument("--tag",default="controls_v1")
    ap.add_argument("--marginal-tag",default="replication_v2");ap.add_argument("--seeds",type=int,nargs="+",default=[0,1,2])
    ap.add_argument("--draws",type=int,default=512);ap.add_argument("--fit-draws",type=int,default=256)
    main(ap.parse_args())
