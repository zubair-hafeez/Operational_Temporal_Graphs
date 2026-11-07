"""Development-only marginal and joint controls, with comparable cohort scores."""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import argparse
import copy
import hashlib
import json
import platform
import shutil
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import ndtr, softmax
from scipy.stats import t as student_t
from sklearn.ensemble import HistGradientBoostingClassifier
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from extension.models import CohortNet, local_codes, relation_summary


def crps(samples, truth):
    """Empirical CRPS, draws along first dimension."""
    ordered = np.sort(samples, axis=0)
    m = len(samples)
    w = (2*np.arange(1, m+1)-m-1).reshape((m,) + (1,)*(samples.ndim-1))
    return np.abs(samples-truth).mean(0) - (ordered*w).sum(0)/m**2


def calibrate(prob, truth, cohorts):
    best = None
    for temp in [.7, 1., 1.4, 2.]:
        p = softmax(np.log(np.clip(prob, 1e-7, 1))/temp, -1).cumsum(-1)[:, :4]
        score = np.mean([np.square(p[c["ix"]]-truth[c["ix"]]).mean() for c in cohorts])
        if best is None or score < best[0]:
            best = score, temp
    return best[1]


def projections(codes):
    groups = [("population", np.arange(len(codes)))]
    for j in range(codes.shape[1]):
        for value in np.unique(codes[:, j]):
            ix = np.flatnonzero(codes[:, j] == value)
            if value >= 0 and len(ix) >= 5:
                groups.append((f"relation_{j}", ix))
    return groups


def scenario(cdf, relations, weights, draws, seed, df=None, shuffled=False):
    rng = np.random.default_rng(seed)
    r = relations.copy()
    if shuffled:
        permutation_rng = np.random.default_rng(seed + 18371)
        for j in range(r.shape[1]):
            known = r[:, j] >= 0
            r[known, j] = permutation_rng.permutation(r[known, j])
    codes = local_codes(r)
    z = np.sqrt(max(0, 1-sum(weights))) * rng.standard_normal((draws, len(cdf)))
    z += np.sqrt(weights[0]) * rng.standard_normal((draws, 1))
    for j in range(r.shape[1]):
        z += np.sqrt(weights[j+1]) * rng.standard_normal((draws, int(codes[:, j].max())+1))[:, codes[:, j]]
    if df is None:
        u = ndtr(z)
    else:
        u = student_t.cdf(z / np.sqrt(rng.chisquare(df, (draws, 1))/df), df)
    return u[:, :, None] <= cdf[None]


def count_score(sample, truth, groups):
    scores = []
    for kind, ix in groups:
        value = crps(sample[:, ix].sum(1)/len(ix), truth[ix].sum(0)/len(ix)).mean()
        scores.append((kind, value))
    # Equal weight per projection type, then per group/horizon.
    return float(np.mean([np.mean([v for k,v in scores if k == kind]) for kind in sorted(set(k for k,v in scores))]))


def main(a):
    torch.set_num_threads(4)
    if a.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(.16, torch.device(a.device))
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    source = ROOT / "data/processed/extension_20261021" / a.source
    out = ROOT / "runs/extension_20261021" / f"{a.source}_seed{a.seed}_{a.tag}"
    out.mkdir(parents=True, exist_ok=False)
    code = out / "code"
    code.mkdir()
    code_hashes = {}
    for filename in ["run_pilots.py", "models.py", "prepare_pilots.py"]:
        src = ROOT / "extension" / filename
        shutil.copy2(src, code / filename)
        code_hashes[filename] = hashlib.sha256(src.read_bytes()).hexdigest()
    start = time.monotonic()
    z = np.load(source / "cohorts.npz")
    x, y, r, history = z["x"], z["y"], z["relations"], z["history"]
    meta = json.loads((source / "manifest.json").read_text())
    cohorts = meta["cohorts"]
    for c in cohorts:
        c["ix"] = np.arange(c["offset"], c["offset"] + c["size"])
    parts = {p:[c for c in cohorts if c["part"] == p] for p in ["train", "calibration", "dependence", "validation"]}
    if any(not v for v in parts.values()):
        raise ValueError("Every chronological partition needs eligible origins")
    train = np.concatenate([c["ix"] for c in parts["train"]])
    mean, scale = x[train].mean(0), x[train].std(0).clip(.1)
    xx = np.clip((x-mean)/scale, -10, 10)
    target = (1-y).sum(-1).astype(np.int64)
    context = np.empty((len(x), x.shape[1]*(2+r.shape[1])+r.shape[1]), np.float32)
    codes = np.empty_like(r)
    for c in cohorts:
        ix = c["ix"]
        context[ix] = relation_summary(xx[ix], r[ix]); codes[ix] = local_codes(r[ix])
    weight = np.zeros(len(x))
    for c in parts["train"]:
        weight[c["ix"]] = 1/c["size"]
    weight *= len(train)/len(parts["train"])
    probabilities, configs = {}, {}
    for name, features in [("hgb_personal", xx), ("hgb_relational", context),
                           ("hgb_temporal", np.column_stack([context, history.reshape(len(x), -1)]))]:
        model = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15, l2_regularization=10,
                                               early_stopping=False, random_state=a.seed)
        model.fit(features[train], target[train], sample_weight=weight[train])
        p = np.full((len(x), 5), 1e-8)
        p[:, model.classes_.astype(int)] = model.predict_proba(features)
        p /= p.sum(-1, keepdims=True)
        probabilities[name] = p
        import joblib
        joblib.dump(model, out/f"{name}.joblib")
        configs[name] = {"kind": "HGB", "max_iter":100, "max_leaf_nodes":15}
        print(json.dumps({"model":name,"status":"fit"}), flush=True)
    tensors = [torch.as_tensor(v, device=a.device) for v in [xx, codes, history, target]]
    architectures = ["mlp", "set", "graph", "temporal_attention"]
    if a.ablations:
        architectures += ["graph_shuffled"] + [f"graph_relation{j}" for j in range(r.shape[1])]
    for arch in architectures:
        torch.manual_seed(a.seed)
        architecture = "graph" if arch.startswith("graph_") else arch
        network_codes = codes.copy()
        if arch.startswith("graph_relation"):
            active = int(arch.removeprefix("graph_relation"))
            for c in cohorts:
                for j in range(r.shape[1]):
                    if j != active:network_codes[c["ix"], j] = np.arange(c["size"])
        elif arch == "graph_shuffled":
            perm_rng = np.random.default_rng(9187)
            for c in cohorts:
                for j in range(r.shape[1]):
                    network_codes[c["ix"], j] = perm_rng.permutation(network_codes[c["ix"], j])
        tensors[1] = torch.as_tensor(network_codes, device=a.device)
        net = CohortNet(x.shape[1], r.shape[1], architecture).to(a.device)
        optimizer = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.001)
        best, state, epoch_chosen = np.inf, None, None
        for epoch in range(a.epochs):
            net.train()
            for ci in rng.permutation(len(parts["train"])):
                ix = parts["train"][ci]["ix"]
                optimizer.zero_grad(set_to_none=True)
                logits = net(tensors[0][ix], tensors[1][ix], tensors[2][ix])
                loss = torch.nn.functional.cross_entropy(logits, tensors[3][ix])
                loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 2.); optimizer.step()
            net.eval()
            with torch.no_grad():
                score = np.mean([float(((net(tensors[0][c["ix"]],tensors[1][c["ix"]],tensors[2][c["ix"]]).softmax(-1).cumsum(-1)[:,:4]
                                      - torch.as_tensor(y[c["ix"]],device=a.device))**2).mean()) for c in parts["calibration"]])
            if score < best:
                best, state, epoch_chosen = score, copy.deepcopy(net.state_dict()), epoch+1
        net.load_state_dict(state); net.eval()
        p = np.empty((len(x), 5))
        with torch.no_grad():
            for c in cohorts:
                ix = c["ix"]
                p[ix] = net(tensors[0][ix], tensors[1][ix], tensors[2][ix]).softmax(-1).cpu().numpy()
        torch.save({"state_dict":state,"architecture":arch,"features":x.shape[1],"relations":r.shape[1]}, out/f"{arch}.pt")
        probabilities[arch] = p
        configs[arch] = {"epoch":epoch_chosen,"width":48,"independent_implementation":True}
        print(json.dumps({"model":arch,"calibration_brier":best,"epoch":epoch_chosen}), flush=True)
    rows, cdfs = [], {}
    for name, prob in probabilities.items():
        temp = calibrate(prob, y, parts["calibration"])
        cdfs[name] = softmax(np.log(np.clip(prob, 1e-7, 1))/temp, -1).cumsum(-1)[:, :4]
        configs[name]["temperature"] = temp
        for c in cohorts:
            ix = c["ix"]
            rows.append({"model":name,"part":c["part"],"origin":c["origin"],"size":c["size"],
                         "brier":float(np.square(cdfs[name][ix]-y[ix]).mean())})
    marginal = pd.DataFrame(rows)
    marginal.to_csv(out/"marginal_scores.csv",index=False)
    chosen = marginal.loc[marginal.part.eq("calibration")].groupby("model").brier.mean().idxmin()
    cdf = cdfs[chosen]
    np.savez_compressed(out/"predictions.npz", **cdfs)
    np.savez(out/"scaler.npz",mean=mean,scale=scale)
    print(json.dumps({"selected_marginal":chosen,"validation":marginal.loc[marginal.part.eq("validation")].groupby("model").brier.mean().to_dict()}),flush=True)
    if a.marginals_only:
        report={"config":vars(a),"selected_marginal":chosen,"marginal_configs":configs,
                "elapsed_seconds":time.monotonic()-start,"code_hashes":code_hashes,
                "python":platform.python_version(),"torch":torch.__version__,
                "stage":"development marginal fits; joint fits pending", "prepared_sha256":hashlib.sha256((source/"manifest.json").read_bytes()).hexdigest()}
        (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
        return
    projected = {c["origin"]:projections(r[c["ix"]]) for c in cohorts}
    def objective(weights, df=None):
        values = []
        for i,c in enumerate(parts["dependence"]):
            ix = c["ix"]
            sample = scenario(cdf[ix],r[ix],weights,a.draws,a.seed*10000+i,df)
            values.append(count_score(sample,y[ix],projected[c["origin"]]))
        return float(np.mean(values))
    base = [0.]*(r.shape[1]+1)
    models = {"independent":(base,None,False)}
    global_trials = [(objective([v]+base[1:]),v) for v in [0,.05,.15,.3,.5]]
    global_value = min(global_trials)[1]
    global_weights = [global_value]+base[1:]
    models["gaussian_global"] = (global_weights,None,False)
    for j in range(r.shape[1]):
        for add_global in [False,True]:
            trials=[]
            for v in [0,.05,.15,.3,.5]:
                w = global_weights.copy() if add_global else base.copy(); w[j+1]=v
                if sum(w)<=.85:trials.append((objective(w),w))
            w = min(trials,key=lambda v:v[0])[1]
            name = f"gaussian_{'global_' if add_global else ''}{meta['relationships'][j]}"
            models[name]=(w,None,False)
    w = global_weights.copy()
    for j in range(r.shape[1]):
        trials=[]
        for v in [0,.05,.15,.3,.5]:
            candidate=w.copy();candidate[j+1]=v
            if sum(candidate)<=.85:trials.append((objective(candidate),candidate))
        w=min(trials,key=lambda v:v[0])[1]
    models["gaussian_typed"]=(w,None,False)
    models["gaussian_shuffled"]=(w,None,True)
    trials=[]
    for df in [4,8]:
        for candidate in [global_weights,w]:trials.append((objective(candidate,df),candidate,df))
    _,tw,tdf=min(trials,key=lambda v:v[0]);models["student_t"]=(tw,tdf,False)
    joint_rows=[]
    for name,(weights,df,shuffled) in models.items():
        for part in ["dependence","validation"]:
            for i,c in enumerate(parts[part]):
                ix=c["ix"];sample=scenario(cdf[ix],r[ix],weights,a.draws,a.seed*10000+i,df,shuffled)
                groups=projected[c["origin"]]
                kinds={}
                # Per-projection count CRPS, central 90% coverage, upper-quantile reserve proxy.
                for kind,g in groups:
                    counts=sample[:,g].sum(1);truth=y[ix][g].sum(0)
                    lo,hi=np.quantile(counts,[.05,.95],axis=0)
                    remaining=len(g)-counts;actual=len(g)-truth
                    reserve=np.quantile(remaining,.9,axis=0)
                    values=[crps(counts/len(g),truth/len(g)).mean(),((truth>=lo)&(truth<=hi)).mean(),
                            (9*np.maximum(actual-reserve,0)+np.maximum(reserve-actual,0)).mean()/len(g)]
                    kinds.setdefault(kind,[]).append(values)
                macro=np.mean([np.mean(v,axis=0) for v in kinds.values()],axis=0)
                joint_rows.append({"model":name,"part":part,"origin":c["origin"],"count_crps":macro[0],"coverage90":macro[1],"reserve_loss":macro[2]})
        print(json.dumps({"joint_model":name,"weights":weights,"df":df}),flush=True)
    joint=pd.DataFrame(joint_rows);joint.to_csv(out/"joint_scores.csv",index=False)
    report={"config":vars(a),"selected_marginal":chosen,"marginal_configs":configs,"joint_configs":models,
            "marginal_validation":marginal.loc[marginal.part.eq("validation")].groupby("model").brier.mean().to_dict(),
            "joint_validation":joint.loc[joint.part.eq("validation")].groupby("model")[["count_crps","coverage90","reserve_loss"]].mean().to_dict("index"),
            "elapsed_seconds":time.monotonic()-start,"stage":"exploratory, not final confirmation",
            "prepared_sha256":hashlib.sha256((source/"manifest.json").read_bytes()).hexdigest(),
            "script_sha256":code_hashes["run_pilots.py"],"code_hashes":code_hashes,
            "limitations":["single pilot seed", "small Monte Carlo ensemble", "no direct count comparator yet", "named architecture implementations are independent adaptations"]}
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--source",required=True);ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--epochs",type=int,default=12);ap.add_argument("--draws",type=int,default=128)
    ap.add_argument("--device",default="cpu");ap.add_argument("--tag",default="pilot_v1")
    ap.add_argument("--marginals-only",action="store_true");ap.add_argument("--ablations",action="store_true")
    main(ap.parse_args())
