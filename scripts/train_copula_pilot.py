"""Train matched static/global/context factor models on disjoint-fit pilots."""
from __future__ import annotations
import argparse,hashlib,json,os,platform,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from otg.copula import FactorLoadings,pair_nll
ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['borg','bpi'],required=True)
    ap.add_argument('--seed',type=int,default=0);ap.add_argument('--device',default='cpu')
    ap.add_argument('--family',choices=['gaussian','commonshock'],default='gaussian')
    ap.add_argument('--epochs',type=int,default=60);ap.add_argument('--batch-size',type=int,default=4096)
    ap.add_argument('--modes',nargs='+',default=['global','static','context']);a=ap.parse_args()
    torch.set_num_threads(4);torch.manual_seed(a.seed);np.random.seed(a.seed)
    torch.use_deterministic_algorithms(True);device=torch.device(a.device)
    data=ROOT/'data/processed'/f'{a.dataset}_copula_pilot'
    tag='copula' if a.family=='gaussian' else 'commonshock'
    out=ROOT/'runs'/f'learned_{tag}_{a.dataset}_seed{a.seed}';out.mkdir(parents=True,exist_ok=False)
    q=pd.read_parquet(data/'queries.parquet');features=np.load(data/'features.npy')
    x=torch.as_tensor(features,device=device);p=torch.as_tensor(q.p.to_numpy(np.float32),device=device)
    y=torch.as_tensor(q['exit'].to_numpy(np.int64),device=device)
    relation=q.relation.to_numpy();states=q.state.to_numpy()
    pair_sets={name:np.load(data/f'pairs_{name}.npy') for name in ['dependence','validation']}
    logs=[];metrics=[];start=time.monotonic()
    manifest={'config':vars(a),'torch':torch.__version__,'python':platform.python_version(),'device':str(device),'device_name':torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'model_sha256':hashlib.sha256((ROOT/'otg/copula.py').read_bytes()).hexdigest(),'data_metadata_sha256':hashlib.sha256((data/'metadata.json').read_bytes()).hexdigest(),
        'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'status':'running','checkpoint_selection':'Minimum validation pair composite negative log likelihood; patience 10 epochs.'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for state in map(int,sorted(q.state.unique())):
        pairs={k:torch.as_tensor(v[states[v[:,0]]==state],device=device,dtype=torch.long) for k,v in pair_sets.items()}
        same={k:torch.as_tensor(relation[v[:,0].cpu().numpy()]==relation[v[:,1].cpu().numpy()],device=device,dtype=torch.float32) for k,v in pairs.items()}
        for mode in a.modes:
            torch.manual_seed(a.seed);model=FactorLoadings(x.shape[1],mode=mode).to(device)
            optimizer=torch.optim.AdamW(model.parameters(),lr=.002 if mode=='context' else .03,weight_decay=1e-4 if mode=='context' else 0)
            best=float('inf');best_epoch=-1
            for epoch in range(a.epochs):
                model.train();order=torch.randperm(len(pairs['dependence']),device=device)
                losses=[]
                for index in order.split(a.batch_size):
                    ij=pairs['dependence'][index];i,j=ij[:,0],ij[:,1]
                    loss=pair_nll(model(x[i]),model(x[j]),p[i],p[j],same['dependence'][index],y[i],y[j],family=a.family).mean()
                    optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();losses.append((loss.item(),len(index)))
                model.eval();vl=[]
                with torch.no_grad():
                    for lo in range(0,len(pairs['validation']),a.batch_size):
                        ij=pairs['validation'][lo:lo+a.batch_size];i,j=ij[:,0],ij[:,1]
                        v=pair_nll(model(x[i]),model(x[j]),p[i],p[j],same['validation'][lo:lo+a.batch_size],y[i],y[j],family=a.family)
                        vl.append((v.sum().item(),len(v)))
                value=sum(v[0] for v in vl)/sum(v[1] for v in vl)
                row={'state':state,'mode':mode,'epoch':epoch,'train_pair_nll':sum(v*n for v,n in losses)/sum(n for _,n in losses),'validation_pair_nll':value,'elapsed_s':time.monotonic()-start}
                logs.append(row)
                if value<best-1e-7:
                    best=value;best_epoch=epoch;torch.save(model.state_dict(),out/f'{mode}_state{state}.pt')
                if epoch%10==0:print(json.dumps(row),flush=True)
                if epoch-best_epoch>=10:break
            model.load_state_dict(torch.load(out/f'{mode}_state{state}.pt',map_location=device,weights_only=True));model.eval()
            with torch.no_grad():loads=torch.cat([model(chunk) for chunk in x.split(16384)]).cpu().numpy()
            np.save(out/f'{mode}_state{state}_loadings.npy',loads)
            metrics.append({'state':state,'mode':mode,'best_epoch':best_epoch,'validation_pair_nll':best,'parameters':sum(t.numel() for t in model.parameters()),'elapsed_s':time.monotonic()-start})
            (out/'training_log.json').write_text(json.dumps(logs,indent=2)+'\n');(out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    manifest['status']='complete';manifest['elapsed_s']=time.monotonic()-start
    manifest['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(metrics),flush=True)


if __name__=='__main__':main()
