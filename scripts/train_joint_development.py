"""Multi-horizon crossed-factor learning from forward out-of-sample marginals."""
from __future__ import annotations
import argparse,hashlib,json,os,platform,sys,time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from otg.copula import StratifiedFactors,pair_nll
ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['borg','bpi'],required=True);ap.add_argument('--history',action='store_true')
    ap.add_argument('--family',choices=['gaussian','commonshock'],default='gaussian');ap.add_argument('--seed',type=int,default=0);ap.add_argument('--device',default='cpu')
    ap.add_argument('--epochs',type=int,default=50);ap.add_argument('--batch-size',type=int,default=4096);ap.add_argument('--shrink',type=float,default=.01)
    ap.add_argument('--modes',nargs='+',choices=['global','static','context'],default=['global','static','context'])
    ap.add_argument('--data-tag',choices=['v1','history_v1','selected_v2']);a=ap.parse_args()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);device=torch.device(a.device)
    tag=a.data_tag or ('history_v1' if a.history else 'v1');data=ROOT/'data/processed'/f'{a.dataset}_joint_{tag}';meta=json.loads((data/'metadata.json').read_text())
    out=ROOT/'runs'/f'joint_{a.dataset}_{tag}_{a.family}_seed{a.seed}';out.mkdir(parents=True,exist_ok=False)
    q=pd.read_parquet(data/'queries.parquet');xx=np.load(data/'features.npy');x=torch.as_tensor(xx,device=device)
    p=torch.as_tensor(np.clip(np.load(data/'cdf.npy'),1e-6,1-1e-6),device=device)
    unit='m' if a.dataset=='borg' else 'd';yy=q[[f'exit_{h}{unit}' for h in meta['horizons']]].to_numpy(np.int64);y=torch.as_tensor(yy,device=device)
    states=sorted(q.state.unique());state=torch.as_tensor(np.searchsorted(states,q.state.to_numpy()),device=device,dtype=torch.long)
    relation=q[meta['relations']].to_numpy(np.int64);available=torch.as_tensor(relation>=0,device=device,dtype=torch.float32)
    pairs={part:torch.as_tensor(np.load(data/f'pairs_{part}.npy'),device=device,dtype=torch.long) for part in ['train','validation']}
    same={}
    for part,ij in pairs.items():
        arr=ij.cpu().numpy();same[part]=torch.as_tensor((relation[arr[:,0]]==relation[arr[:,1]])&(relation[arr[:,0]]>=0),device=device,dtype=torch.float32)
    manifest={'config':vars(a),'states':list(map(int,states)),'n_relations':len(meta['relations']),'input_dim':x.shape[1],'hidden':32,'status':'running','torch':torch.__version__,'python':platform.python_version(),
        'data_metadata_sha256':hashlib.sha256((data/'metadata.json').read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'model_sha256':hashlib.sha256((ROOT/'otg/copula.py').read_bytes()).hexdigest(),
        'selection':'Minimum validation cross-horizon pair NLL; patience 8. Random horizon pairs during fitting, fixed balanced horizon-pair pattern during validation.',
        'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'device_name':torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');logs=[];metrics=[];start=time.monotonic()
    n_h=p.shape[1]
    for mode in a.modes:
        torch.manual_seed(a.seed);model=StratifiedFactors(x.shape[1],len(states),len(meta['relations']),mode=mode).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001 if mode=='context' else .02,weight_decay=1e-4 if mode=='context' else 0)
        best=float('inf');best_epoch=-1
        for epoch in range(a.epochs):
            model.train();order=torch.randperm(len(pairs['train']),device=device);train_sum=train_reg=0.;train_n=0
            for index in order.split(a.batch_size):
                ij=pairs['train'][index];i,j=ij[:,0],ij[:,1];h_i=torch.randint(n_h,(len(i),),device=device);h_j=torch.randint(n_h,(len(i),),device=device)
                li,ri=model(x[i],state[i],available[i]);lj,rj=model(x[j],state[j],available[j])
                loss=pair_nll(li,lj,p[i,h_i],p[j,h_j],same['train'][index],y[i,h_i],y[j,h_j],family=a.family).mean();reg=(ri+rj)/2
                objective=loss+a.shrink*reg;optimizer.zero_grad();objective.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
                train_sum+=loss.item()*len(i);train_reg+=reg.item()*len(i);train_n+=len(i)
            model.eval();validation_sum=0.;validation_n=0
            with torch.no_grad():
                for lo in range(0,len(pairs['validation']),a.batch_size):
                    ij=pairs['validation'][lo:lo+a.batch_size];i,j=ij[:,0],ij[:,1];position=torch.arange(lo,lo+len(i),device=device)
                    h_i=position%n_h;h_j=(position//n_h)%n_h
                    li,_=model(x[i],state[i],available[i]);lj,_=model(x[j],state[j],available[j])
                    loss=pair_nll(li,lj,p[i,h_i],p[j,h_j],same['validation'][lo:lo+len(i)],y[i,h_i],y[j,h_j],family=a.family)
                    validation_sum+=loss.sum().item();validation_n+=len(i)
            value=validation_sum/validation_n;row={'mode':mode,'epoch':epoch,'train_pair_nll':train_sum/train_n,'mean_context_offset_squared':train_reg/train_n,'validation_pair_nll':value,'elapsed_s':time.monotonic()-start}
            logs.append(row)
            if value<best-1e-7:best=value;best_epoch=epoch;torch.save(model.state_dict(),out/f'{mode}.pt')
            if epoch%5==0:print(json.dumps(row),flush=True)
            if epoch-best_epoch>=8:break
        model.load_state_dict(torch.load(out/f'{mode}.pt',map_location=device,weights_only=True));model.eval()
        with torch.no_grad():
            loads=[]
            for lo in range(0,len(x),16384):loads.append(model(x[lo:lo+16384],state[lo:lo+16384],available[lo:lo+16384])[0].cpu().numpy())
        np.save(out/f'{mode}_loadings.npy',np.concatenate(loads))
        metrics.append({'mode':mode,'best_epoch':best_epoch,'validation_pair_nll':best,'parameters':sum(v.numel() for v in model.parameters()),'elapsed_s':time.monotonic()-start})
        (out/'training_log.json').write_text(json.dumps(logs,indent=2)+'\n');(out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    manifest.update(status='complete',elapsed_s=time.monotonic()-start,finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(metrics),flush=True)


if __name__=='__main__':main()
