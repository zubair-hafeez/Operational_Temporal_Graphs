"""Within-relation residual dependence versus size-preserving permutations."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]


def pair_moment(values,codes):
    _,r=np.unique(codes,return_inverse=True)
    n=np.bincount(r);total=np.zeros((len(n),values.shape[1]));squares=total.copy()
    np.add.at(total,r,values);np.add.at(squares,r,values**2)
    keep=n>=5
    if not keep.any():return np.nan
    return float(((total[keep]**2-squares[keep])/(n[keep,None]*(n[keep,None]-1))).mean())


def main(a):
    src=ROOT/"data/processed/extension_20261021"/a.source
    z=np.load(src/"cohorts.npz");m=json.loads((src/"manifest.json").read_text())
    run=ROOT/"runs/extension_20261021"/f"{a.source}_seed0_{a.tag}"
    cfg=json.loads((run/"report.json").read_text());p=np.load(run/"predictions.npz")[cfg["selected_marginal"]]
    rng=np.random.default_rng(592);rows=[]
    for c in m["cohorts"]:
        if c["part"] not in ["dependence","validation"]:continue
        ix=np.arange(c["offset"],c["offset"]+c["size"])
        residual=(z["y"][ix]-p[ix])/np.sqrt(p[ix]*(1-p[ix])+.01)
        residual-=residual.mean(0)
        for j,name in enumerate(m["relationships"]):
            codes=z["relations"][ix,j];known=codes>=0
            values=residual[known];codes=codes[known]
            if len(values)<5:continue
            observed=pair_moment(values,codes)
            null=[pair_moment(values,rng.permutation(codes)) for _ in range(100)]
            rows.append({"origin":c["origin"],"part":c["part"],"relation":name,"observed_pair_product":observed,
                         "shuffled_pair_product":float(np.mean(null)),"excess":observed-float(np.mean(null))})
    out=ROOT/"reports/extension_20261021"/f"relation_diagnostic_{a.source}.csv"
    pd.DataFrame(rows).to_csv(out,index=False)
    print(pd.DataFrame(rows).groupby(["part","relation"])[["observed_pair_product","shuffled_pair_product","excess"]].mean().to_string(),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--source",required=True);ap.add_argument("--tag",default="pilot_v1");main(ap.parse_args())
