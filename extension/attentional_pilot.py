"""Bounded adaptation of the published TACTiS AttentionalCopula module.

The upstream copula is unchanged. We supply current-cohort context in place of
its time-series encoder, train randomized discrete PIT values, and rank-adjust
generated uniforms to enforce the same empirical marginals for comparisons.
This is not a full reproduction of TACTiS/TACTiS-2 or exact discrete likelihood.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS","4")
os.environ.setdefault("OPENBLAS_NUM_THREADS","4")
import argparse,copy,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"third_party/tactis"))
from tactis.model.decoder import AttentionalCopula
from extension.run_pilots import scenario,projections,count_score


class AdaptedCopula(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.embed=nn.Sequential(nn.Linear(width,32),nn.ReLU(),nn.Linear(32,32))
        self.copula=AttentionalCopula(input_dim=32,attention_heads=2,attention_layers=1,attention_dim=16,
                                     mlp_layers=1,mlp_dim=32,resolution=10,dropout=0.)

    def encode(self,x):
        z=self.embed(x)
        # A deterministic context token prevents fully masked attention rows.
        return z.mean(1,keepdim=True),z

    def loss(self,x,u):
        hist,pred=self.encode(x)
        return self.copula.loss(hist,torch.full(hist.shape[:2],.5,device=x.device),pred,u).mean()/x.shape[1]

    def sample(self,x,draws):
        hist,pred=self.encode(x)
        return self.copula.sample(draws,hist,torch.full(hist.shape[:2],.5,device=x.device),pred)


def main(a):
    torch.set_num_threads(4);torch.manual_seed(a.seed)
    if a.device.startswith("cuda"):torch.cuda.set_per_process_memory_fraction(.16,torch.device(a.device))
    start=time.monotonic();out=ROOT/"runs/extension_20261021"/f"{a.source}_attentional_seed{a.seed}"
    out.mkdir(exist_ok=False)
    src=ROOT/"data/processed/extension_20261021"/a.source
    z=np.load(src/"cohorts.npz");m=json.loads((src/"manifest.json").read_text())
    marginal=ROOT/"runs/extension_20261021"/f"{a.source}_seed0_pilot_v1"
    cfg=json.loads((marginal/"report.json").read_text());p=np.load(marginal/"predictions.npz")[cfg["selected_marginal"]]
    x,y,r=z["x"],z["y"],z["relations"]
    cohorts=m["cohorts"]
    for c in cohorts:c["ix"]=np.arange(c["offset"],c["offset"]+c["size"])
    dep=[c for c in cohorts if c["part"]=="dependence"];val=[c for c in cohorts if c["part"]=="validation"]
    # Purge the maximum horizon between dependence fit and checkpoint windows.
    boundary=dep[max(1,len(dep)*2//3)]["origin"]
    fit=[c for c in dep if c["origin"]+max(m["horizons_seconds"])<boundary]
    checkpoint=[c for c in dep if c["origin"]>=boundary]
    if not fit or not checkpoint:raise ValueError("Need nonoverlapping dependence fit/checkpoint windows")
    fit_ids=np.concatenate([c["ix"] for c in fit]);mean=x[fit_ids].mean(0);scale=x[fit_ids].std(0).clip(.1)
    features=np.column_stack([np.clip((x-mean)/scale,-10,10),p]).astype(np.float32)
    net=AdaptedCopula(features.shape[1]).to(a.device);opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.001)
    rng=np.random.default_rng(a.seed)
    def batch(cohort,n):
        ix=rng.choice(cohort["ix"],n,replace=False)
        bins=(1-y[ix]).sum(-1).astype(int);edges=np.column_stack([np.zeros(n),p[ix],np.ones(n)])
        low=edges[np.arange(n),bins];high=edges[np.arange(n),bins+1]
        u=(low+(high-low)*rng.random(n)).astype(np.float32)
        return torch.as_tensor(features[ix][None],device=a.device),torch.as_tensor(u[None],device=a.device)
    best=np.inf;state=None;log=[]
    for epoch in range(a.epochs):
        net.train();losses=[]
        for _ in range(32):
            c=fit[rng.integers(len(fit))];xx,uu=batch(c,min(a.nodes,c["size"]))
            opt.zero_grad(set_to_none=True);loss=net.loss(xx,uu)
            if not torch.isfinite(loss):raise ValueError("Nonfinite upstream copula loss")
            loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2);opt.step();losses.append(loss.item())
        net.eval()
        with torch.no_grad():
            checks=[net.loss(*batch(c,min(a.nodes,c["size"]))).item() for c in checkpoint for _ in range(4)]
        value=float(np.mean(checks));log.append({"epoch":epoch+1,"train_loss":float(np.mean(losses)),"checkpoint_loss":value})
        if value<best:best=value;state=copy.deepcopy(net.state_dict())
    net.load_state_dict(state);net.eval();torch.save(state,out/"model.pt")
    rows=[];max_margin_error=0.
    for index,c in enumerate(val):
        ix=np.random.default_rng(9180+index).choice(c["ix"],min(a.nodes,c["size"]),replace=False)
        groups=projections(r[ix])
        with torch.no_grad():u=net.sample(torch.as_tensor(features[ix][None],device=a.device),a.draws)[0].cpu().numpy().T
        if not np.isfinite(u).all():raise ValueError("Nonfinite samples")
        # Ranks retain sampled dependence but give a deterministic uniform grid per node.
        rank=np.argsort(np.argsort(u,axis=0),axis=0)
        uniform=(rank+.5)/a.draws
        samples={"attentional_rank_calibrated":uniform[:,:,None]<=p[ix][None],"attentional_raw":u[:,:,None]<=p[ix][None]}
        max_margin_error=max(max_margin_error,float(np.abs(samples["attentional_rank_calibrated"].mean(0)-p[ix]).max()))
        for name in ["independent","gaussian_global","gaussian_typed","student_t"]:
            w,df,shuffled=cfg["joint_configs"][name]
            samples[name]=scenario(p[ix],r[ix],w,a.draws,1900+index,df,shuffled)
        for name,s in samples.items():
            rows.append({"model":name,"origin":c["origin"],"nodes":len(ix),"count_crps":count_score(s,y[ix],groups),
                         "marginal_max_abs_error":float(np.abs(s.mean(0)-p[ix]).max())})
    table=pd.DataFrame(rows);table.to_csv(out/"scores.csv",index=False)
    report={"config":vars(a),"model":"unchanged upstream TACTiS AttentionalCopula; custom cohort context and randomized-PIT objective",
            "source":"https://proceedings.mlr.press/v162/drouin22a.html","upstream_commit":json.loads((ROOT/"third_party/tactis_commit.json").read_text()),
            "upstream_decoder_sha256":hashlib.sha256((ROOT/"third_party/tactis/tactis/model/decoder.py").read_bytes()).hexdigest(),
            "training_log":log,"training_origins":len(fit),"checkpoint_origins":len(checkpoint),"validation_origins":len(val),
            "scores":table.groupby("model").count_crps.mean().to_dict(),"rank_marginal_max_error":max_margin_error,
            "elapsed_seconds":time.monotonic()-start,"limitations":["small matched node subsets, not complete-cohort result","discrete randomized PIT surrogate","single seed exploratory","rank adjustment is an adaptation"]}
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--source",default="uci");ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--nodes",type=int,default=64);ap.add_argument("--draws",type=int,default=256)
    ap.add_argument("--epochs",type=int,default=20);ap.add_argument("--device",default="cpu")
    main(ap.parse_args())
