"""Verify freeze, prepare new-period cohorts, apply fixed predictors and score."""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS","4");os.environ.setdefault("OPENBLAS_NUM_THREADS","4")
import argparse,datetime,hashlib,json,sys,time
from pathlib import Path
import joblib,numpy as np,pandas as pd,torch
from scipy.special import softmax
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from extension.freeze_confirmation import PROTOCOL,digest
from extension.prepare_pilots import build
from extension.models import CohortNet,local_codes,relation_summary
from extension.run_pilots import scenario
from extension.joint_controls import group_records,group_scores,bootstrap_difference
from revision.counts import parameters,sample_dm
from revision.exact_counts import exact_projected_scores


def predict(source,folder,device):
    z=np.load(folder/"cohorts.npz");m=json.loads((folder/"manifest.json").read_text())
    x,y,r,history=z["x"],z["y"],z["relations"],z["history"]
    cohorts=m["cohorts"]
    for c in cohorts:c["ix"]=np.arange(c["offset"],c["offset"]+c["size"])
    results={}
    for seed in [0,1,2]:
        run=ROOT/"runs/extension_20261021"/f"{source}_seed{seed}_replication_v2"
        config=json.loads((run/"report.json").read_text());scaler=np.load(run/"scaler.npz")
        xx=np.clip((x-scaler["mean"])/scaler["scale"],-10,10)
        context=np.empty((len(x),x.shape[1]*(2+r.shape[1])+r.shape[1]),np.float32);codes=np.empty_like(r)
        for c in cohorts:
            ix=c["ix"];context[ix]=relation_summary(xx[ix],r[ix]);codes[ix]=local_codes(r[ix])
        for name,cfg in config["marginal_configs"].items():
            if name.startswith("hgb_"):
                features=xx if name=="hgb_personal" else context
                if name=="hgb_temporal":features=np.column_stack([context,history.reshape(len(x),-1)])
                model=joblib.load(run/f"{name}.joblib")
                prob=np.full((len(x),5),1e-8);prob[:,model.classes_.astype(int)]=model.predict_proba(features);prob/=prob.sum(-1,keepdims=True)
            else:
                architecture="graph" if name.startswith("graph_") else name
                model=CohortNet(x.shape[1],r.shape[1],architecture).to(device)
                model.load_state_dict(torch.load(run/f"{name}.pt",map_location=device,weights_only=True)["state_dict"]);model.eval()
                permutation_rng=np.random.default_rng(9187)
                prob=np.empty((len(x),5))
                with torch.no_grad():
                    for c in cohorts:
                        ix=c["ix"];local=codes[ix].copy()
                        if name.startswith("graph_relation"):
                            for j in range(r.shape[1]):
                                if j!=int(name.removeprefix("graph_relation")):local[:,j]=np.arange(len(ix))
                        elif name=="graph_shuffled":
                            for j in range(r.shape[1]):local[:,j]=permutation_rng.permutation(local[:,j])
                        prob[ix]=model(torch.as_tensor(xx[ix],device=device),torch.as_tensor(local,device=device),torch.as_tensor(history[ix],device=device)).softmax(-1).cpu().numpy()
            cdf=softmax(np.log(np.clip(prob,1e-7,1))/cfg["temperature"],-1).cumsum(-1)[:,:4]
            results[name]=results.get(name,0)+cdf/3
        print(json.dumps({"source":source,"predicted_seed":seed}),flush=True)
    return results,z,m,cohorts


def main(a):
    torch.set_num_threads(4)
    if a.device.startswith("cuda"):torch.cuda.set_per_process_memory_fraction(.16,torch.device(a.device))
    protocol=json.loads(PROTOCOL.read_text())
    if a.source not in protocol["sources"]:raise ValueError("Not a frozen source")
    for name,expected in protocol["files"].items():
        if digest(ROOT/name)!=expected:raise ValueError(f"Frozen asset changed: {name}")
    print(json.dumps({"freeze_verified":len(protocol["files"])}),flush=True)
    out=ROOT/"runs/extension_20261021"/f"{a.source}_confirmation"
    out.mkdir(exist_ok=False);start=time.monotonic()
    (out/"opened.json").write_text(json.dumps({"utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"protocol_sha256":digest(PROTOCOL),"source":a.source},indent=2)+"\n")
    build(a.source,evaluation=True)
    folder=ROOT/"data/processed/extension_20261021"/f"{a.source}_final"
    predictions,z,m,cohorts=predict(a.source,folder,a.device)
    np.savez_compressed(out/"predictions.npz",**predictions)
    marg=[]
    for name,p in predictions.items():
        for c in cohorts:
            ix=c["ix"]
            for h in range(4):marg.append({"model":name,"origin":c["origin"],"horizon":h,"brier":float(((p[ix,h]-z["y"][ix,h])**2).mean()),"mean_prediction":float(p[ix,h].mean()),"mean_observed":float(z["y"][ix,h].mean())})
    pd.DataFrame(marg).to_csv(out/"marginal_scores.csv",index=False)
    fit=ROOT/"runs/extension_20261021"/f"{a.source}_joint_controls_v2"
    report=json.loads((fit/"report.json").read_text());p=predictions[report["selected_marginal"]]
    y,r=z["y"],z["relations"];rows=[]
    development=ROOT/"data/processed/extension_20261021"/a.source
    dz=np.load(development/"cohorts.npz");dm=json.loads((development/"manifest.json").read_text())
    dp=np.load(fit/"selected_cdf.npy");residuals={}
    for c in dm["cohorts"]:
        if c["part"]!="dependence":continue
        ix=np.arange(c["offset"],c["offset"]+c["size"])
        for kind,g in group_records(dp[ix],dz["y"][ix],dz["relations"][ix]).items():
            residuals.setdefault(kind,[]).append(g["truth"]/g["n"][:,None]-g["prob"].cumsum(-1)[:,:4])
    aliases={}
    for name in list(report["joint_models"])+list(report["direct_fits"])+["direct_residual"]:
        if name in report["joint_models"]:
            w,df,shuffled=report["joint_models"][name]
            key=json.dumps([[float(v) for v in w],df,bool(shuffled and sum(w[1:])>0)])
            if key in aliases:
                original=aliases[key]
                rows.extend([{**row,"model":name} for row in rows if row["model"]==original])
                print(json.dumps({"source":a.source,"final_scored":name,"identical_to":original}),flush=True)
                continue
            aliases[key]=name
        for c in cohorts:
            ix=c["ix"];g=group_records(p[ix],y[ix],r[ix])
            for seed in protocol["sampling_seeds"]:
                if name in report["joint_models"]:
                    w,df,shuffled=report["joint_models"][name]
                    sample=scenario(p[ix],r[ix],w,protocol["draws"],1400+seed,df,shuffled)
                    counts={kind:np.stack([sample[:,idx].sum(1) for idx in v["members"]],axis=1) for kind,v in g.items()}
                elif name in report["direct_fits"]:
                    rng=np.random.default_rng(1200+seed)
                    counts={kind:sample_dm(v["n"],parameters(v["prob"],report["direct_fits"][name][kind]),protocol["draws"],rng).cumsum(-1)[:,:,:4] for kind,v in g.items()}
                else:
                    rng=np.random.default_rng(1200+seed);counts={}
                    for kind,v in g.items():
                        draws=[]
                        for _ in range(protocol["draws"]):
                            template=residuals[kind][rng.integers(len(residuals[kind]))]
                            draws.append(template[rng.integers(len(template),size=len(v["n"]))])
                        cdf=np.clip(v["prob"].cumsum(-1)[None,:,:4]+np.array(draws),0,1)
                        counts[kind]=np.rint(np.maximum.accumulate(cdf,axis=-1)*v["n"][None,:,None]).astype(int)
                for score in group_scores(counts,g,p[ix],fair=True):rows.append({"model":name,"seed":seed,"origin":c["origin"],**score})
            if name in report["direct_fits"]:
                exact=exact_projected_scores(g,{kind:parameters(v["prob"],report["direct_fits"][name][kind]) for kind,v in g.items()})
                for kind in g:
                    sub=[s for s in exact if s["projection"]==kind]
                    rows.append({"model":name+"_exact","seed":-1,"origin":c["origin"],"projection":kind,
                                 "count_crps":np.mean([s["fraction_crps"] for s in sub]),"coverage90":np.mean([s["coverage_90"] for s in sub]),
                                 "width90":np.mean([s["width_90_fraction"] for s in sub]),"reserve_loss":np.mean([s["buffer_cost_ratio9"] for s in sub])})
        print(json.dumps({"source":a.source,"final_scored":name}),flush=True)
    table=pd.DataFrame(rows);table.to_csv(out/"projection_scores.csv",index=False)
    metrics=["count_crps","coverage90","width90","reserve_loss","joint_profile_energy","joint_variogram_fair","peak_slowdown_crps"]
    origin=table.groupby(["model","origin"])[metrics].mean().reset_index();origin.to_csv(out/"origin_scores.csv",index=False)
    macro=origin.groupby("model")[metrics].mean();macro.to_csv(out/"macro_scores.csv")
    intervals=[]
    for candidate,ref in protocol["primary_comparisons"]:
        for block in [1,3,7]:intervals.append({"candidate":candidate,"reference":ref,**bootstrap_difference(origin,ref,candidate,block)})
    result={"source":a.source,"origins":len(cohorts),"queries":len(y),"selected_marginal":report["selected_marginal"],"macro":macro.to_dict("index"),
            "paired_intervals":intervals,"protocol_sha256":digest(PROTOCOL),"elapsed_seconds":time.monotonic()-start,"status":"complete"}
    (out/"report.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps({"source":a.source,"complete":True,"elapsed_seconds":result["elapsed_seconds"]}),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--source",required=True);ap.add_argument("--device",default="cpu");main(ap.parse_args())
