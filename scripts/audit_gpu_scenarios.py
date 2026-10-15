"""GPU sampler checks before using acceleration for research scores."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from otg.cohort_scores import projections
from otg.torch_scenarios import projected_counts_torch
ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--device',default='cuda:1');a=ap.parse_args();rows=[]
    for family in ['gaussian','commonshock']:
        rel=np.zeros((10,1),np.int64);state=np.ones(10);p=np.tile([.2,.5,.8],(10,1));loads=np.tile([0,1,0],(10,1));projection=projections(state,rel,['job'])
        values=projected_counts_torch(p,loads,rel,projection,200000,42,family,a.device,chunk_size=3)['job']
        assert np.all((values==0)|(values==10));error=float(np.abs(values.mean(0)/10-p[:1]).max());assert error<.005
        again=projected_counts_torch(p,loads,rel,projection,200000,42,family,a.device,chunk_size=3)['job'];assert np.array_equal(values,again)
        rows.append({'family':family,'max_marginal_error':error,'shared_across_chunks':True,'exact_repeatability':True})
    report={'device':a.device,'torch':torch.__version__,'draws':200000,'results':rows,'accumulation':'int32 scatter-add; integer sum is invariant to parallel scheduling'}
    (ROOT/'reports/gpu_scenario_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
