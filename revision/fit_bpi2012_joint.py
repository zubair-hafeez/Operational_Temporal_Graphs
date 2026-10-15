"""Fit classical dependence controls on December; January checkpoint selection."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from revision.common import ROOT,write_json,digest
from revision.calibration import DAY_US
from revision.fit_bpi2012_marginals import boundary
from scripts.prepare_joint_development import sample_pairs
from otg.copula import StratifiedFactors,pair_nll


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--device',default='cpu');ap.add_argument('--seed',type=int,default=0);ap.add_argument('--epochs',type=int,default=50);a=ap.parse_args();torch.set_num_threads(3);torch.use_deterministic_algorithms(True);device=torch.device(a.device);start=time.monotonic()
    out=ROOT/'runs'/f'revision_bpi2012_joint_v1_seed{a.seed}';out.mkdir(exist_ok=False)
    source=ROOT/'data/processed/revision_bpi2012_development_v1';marg=ROOT/'runs/revision_bpi2012_marginals_v1';q=pd.read_parquet(source/'queries.parquet');t=q.cutoff_us.to_numpy();n=len(q)
    masks={'train':(t>=boundary('2011-12-01'))&(t+14*DAY_US<=boundary('2012-01-01')),'validation':(t>=boundary('2012-01-01'))&(t+14*DAY_US<=boundary('2012-02-01'))}
    rng=np.random.default_rng(2012+a.seed);pairs={key:sample_pairs(q[mask],['resource'],rng) for key,mask in masks.items()};r=q.resource.to_numpy();same={key:torch.as_tensor(((r[v[:,0]]==r[v[:,1]])&(r[v[:,0]]>=0))[:,None],dtype=torch.float32,device=device) for key,v in pairs.items()};pairs={k:torch.as_tensor(v,dtype=torch.long,device=device) for k,v in pairs.items()}
    x=torch.zeros((n,1),device=device);st=torch.zeros(n,dtype=torch.long,device=device);available=torch.as_tensor((r>=0)[:,None],dtype=torch.float32,device=device);y=torch.as_tensor(q[['exit_1d','exit_3d','exit_7d','exit_14d']].to_numpy(),device=device,dtype=torch.long);logs=[];fit_info=[]
    for calibration in ['identity','static','rolling60']:
        p=torch.as_tensor(np.clip(np.load(marg/f'selected_{calibration}_cdf.npy'),1e-6,1-1e-6),device=device)
        for mode in ['global','static']:
            torch.manual_seed(a.seed);model=StratifiedFactors(1,1,1,mode=mode).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=.02,weight_decay=0);best=float('inf');best_epoch=-1
            for epoch in range(a.epochs):
                order=torch.randperm(len(pairs['train']),device=device);train_sum=0
                for ix in order.split(4096):
                    ij=pairs['train'][ix];i,j=ij[:,0],ij[:,1];hi=torch.randint(4,(len(i),),device=device);hj=torch.randint(4,(len(i),),device=device)
                    li,_=model(x[i],st[i],available[i]);lj,_=model(x[j],st[j],available[j]);loss=pair_nll(li,lj,p[i,hi],p[j,hj],same['train'][ix],y[i,hi],y[j,hj]).mean();optimizer.zero_grad();loss.backward();optimizer.step();train_sum+=float(loss)*len(i)
                value=0
                with torch.no_grad():
                    for lo in range(0,len(pairs['validation']),4096):
                        ij=pairs['validation'][lo:lo+4096];i,j=ij[:,0],ij[:,1];pos=torch.arange(lo,lo+len(i),device=device);hi=pos%4;hj=(pos//4)%4;li,_=model(x[i],st[i],available[i]);lj,_=model(x[j],st[j],available[j]);value+=float(pair_nll(li,lj,p[i,hi],p[j,hj],same['validation'][lo:lo+len(i)],y[i,hi],y[j,hj]).sum())
                value/=len(pairs['validation']);logs.append({'calibration':calibration,'mode':mode,'epoch':epoch,'train_nll':train_sum/len(order),'validation_nll':value})
                if value<best-1e-7:best=value;best_epoch=epoch;torch.save(model.state_dict(),out/f'{calibration}_{mode}.pt')
                if epoch-best_epoch>=8:break
            fit_info.append({'calibration':calibration,'mode':mode,'best_epoch':best_epoch,'validation_pair_nll':best});print(json.dumps({**fit_info[-1],'elapsed_s':time.monotonic()-start}),flush=True)
    write_json(out/'training_log.json',logs);write_json(out/'manifest.json',{'status':'complete','config':vars(a),'fits':fit_info,'pairs':{k:len(v) for k,v in pairs.items()},'individual_manifest_sha256':digest(marg/'manifest.json'),'script_sha256':digest(__file__),'interpretation':'December conditional dependence fits with January early stopping; February final outcomes remain unavailable. Same marginal version, separate dependence fit per calibration treatment.','elapsed_s':time.monotonic()-start})


if __name__=='__main__':main()
