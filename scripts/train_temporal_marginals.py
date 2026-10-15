"""Forward temporal neural survival baselines; untouched final remains locked."""
from __future__ import annotations
import argparse,hashlib,json,os,platform,sys,time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from otg.temporal import TemporalSurvival,logits_to_cdf
from scripts.fit_borg_development_marginals import FOLDS
ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['borg','bpi'],required=True);ap.add_argument('--architecture',choices=['mixer','gru','mlp'],required=True)
    ap.add_argument('--seed',type=int,default=0);ap.add_argument('--device',default='cpu');ap.add_argument('--epochs',type=int,default=30);ap.add_argument('--batch-size',type=int,default=2048)
    ap.add_argument('--folds',nargs='+');a=ap.parse_args();torch.set_num_threads(4);torch.use_deterministic_algorithms(True);device=torch.device(a.device)
    history=ROOT/'data/processed'/f'{a.dataset}_temporal_histories_k16_v1';hm=json.loads((history/'metadata.json').read_text());source=ROOT/hm['source_dataset']
    sm=json.loads((source/'metadata.json').read_text());q=pd.read_parquet(source/'queries.parquet');xx=np.load(source/'features.npy')
    if a.dataset=='borg':folds=FOLDS;horizons=[1,5,15,60];unit='m';max_horizon=1/24
    else:
        origin=pd.Timestamp('2016-01-01',tz='UTC')
        def day(s):return (pd.Timestamp(s,tz='UTC')-origin).total_seconds()/86400
        folds={'oof_june':tuple(map(day,['2016-05-01','2016-06-01','2016-07-01'])),'oof_july':tuple(map(day,['2016-06-01','2016-07-01','2016-08-01'])),
               'oof_august':tuple(map(day,['2016-07-01','2016-08-01','2016-09-01'])),'development_october':tuple(map(day,['2016-09-01','2016-10-01','2016-11-01']))}
        horizons=[1,3,7,14];unit='d';max_horizon=14
    requested=a.folds or list(folds)
    if set(requested)-set(folds):raise ValueError('Unknown fold')
    labels=q[[f'exit_{h}{unit}' for h in horizons]].to_numpy(np.int64);exit_bin=(1-labels).sum(1)
    yy=torch.as_tensor(labels,device=device,dtype=torch.float32);target=torch.as_tensor(exit_bin,device=device,dtype=torch.long)
    # Equal forecast-time weight within state, and equal state weight overall.
    weight=q.sampling_weight.to_numpy(np.float64);weight/=q.groupby(['cutoff_us','state']).sampling_weight.transform('sum').to_numpy()
    w=torch.as_tensor(weight,device=device,dtype=torch.float32)
    t=q.elapsed_day.to_numpy();qt=torch.as_tensor(q.cutoff_us.to_numpy(),device=device)
    indices=categories=values=event_times=None
    if a.architecture!='mlp':
        indices=torch.as_tensor(np.load(history/'queries_indices.npy').astype(np.int64),device=device)
        categories=torch.as_tensor(np.load(history/'event_categories.npy').astype(np.int64),device=device)
        values=torch.as_tensor(np.load(history/'event_values.npy'),device=device)
        event_times=torch.as_tensor(np.load(history/'event_times.npy'),device=device)
    out=ROOT/'runs'/f'temporal_{a.dataset}_{a.architecture}_seed{a.seed}';out.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    for fold,(fit_end,cal_end,target_end) in folds.items():
        if fold not in requested:continue
        folder=out/fold;folder.mkdir(exist_ok=False)
        fit=np.flatnonzero(t+max_horizon<=fit_end);cal=np.flatnonzero((t>=fit_end)&(t+max_horizon<=cal_end));test=np.flatnonzero((t>=cal_end)&(t+max_horizon<=target_end))
        mean=xx[fit].mean(0);scale=xx[fit].std(0);scale[scale<1e-5]=1
        x=torch.as_tensor(np.clip((xx-mean)/scale,-10,10).astype(np.float32),device=device);np.savez(folder/'scaler.npz',mean=mean,scale=scale)
        torch.manual_seed(a.seed);model=TemporalSurvival(x.shape[1],hm['vocab_sizes'],hm['event_continuous_dim'],len(hm['scopes']),architecture=a.architecture).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4);best=float('inf');best_epoch=-1;logs=[]
        def forward(ids):
            if a.architecture=='mlp':return model(x[ids])
            ix=indices[ids];mask=ix!=0;lag=((qt[ids,None,None]-event_times[ix]).clamp_min(0)/1e6).float()*mask
            return model(x[ids],categories[ix],values[ix],lag,mask)
        def predict(ids):
            result=[];model.eval()
            with torch.no_grad():
                for lo in range(0,len(ids),a.batch_size):result.append(forward(torch.as_tensor(ids[lo:lo+a.batch_size],device=device)).cpu().numpy())
            return np.concatenate(result)
        manifest={'config':vars(a),'fold':fold,'boundaries_days':[fit_end,cal_end,target_end],'fit':len(fit),'calibration':len(cal),'target':len(test),'width':64,'horizons':horizons,'horizon_unit':unit,
          'input_dim':x.shape[1],'n_scopes':len(hm['scopes']),'vocab_sizes':hm['vocab_sizes'],'continuous_dim':hm['event_continuous_dim'],'status':'running',
          'model_sha256':hashlib.sha256((ROOT/'otg/temporal.py').read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'history_metadata_sha256':hashlib.sha256((history/'metadata.json').read_bytes()).hexdigest(),
          'selection':'Minimum earlier calibration-window integrated Brier; patience 5; temperature fitted in same earlier window. Raw and temperature variants reported on development.',
          'python':platform.python_version(),'torch':torch.__version__,'device':str(device)}
        (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        fit_tensor=torch.as_tensor(fit,device=device);fit_weight=w[fit_tensor].mean()
        for epoch in range(a.epochs):
            model.train();order=fit_tensor[torch.randperm(len(fit),device=device)];loss_sum=0.
            for ids in order.split(a.batch_size):
                logits=forward(ids);loss=(torch.nn.functional.cross_entropy(logits,target[ids],reduction='none')*w[ids]/fit_weight).mean()
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();loss_sum+=loss.item()*len(ids)
            raw_cal=predict(cal);cdf=logits_to_cdf(torch.as_tensor(raw_cal)).numpy();score=float(np.average(((cdf-labels[cal])**2).mean(1),weights=weight[cal]))
            row={'fold':fold,'epoch':epoch,'train_nll':loss_sum/len(fit),'calibration_integrated_brier':score,'elapsed_s':time.monotonic()-start};logs.append(row);print(json.dumps(row),flush=True)
            if score<best-1e-6:best=score;best_epoch=epoch;torch.save(model.state_dict(),folder/'model.pt')
            if epoch-best_epoch>=5:break
        model.load_state_dict(torch.load(folder/'model.pt',weights_only=True,map_location=device));raw_cal=predict(cal);raw_test=predict(test)
        def temperature_objective(log_temperature):
            cdf=logits_to_cdf(torch.as_tensor(raw_cal)/np.exp(log_temperature)).numpy()
            return np.average(((cdf-labels[cal])**2).mean(1),weights=weight[cal])
        optimum=minimize_scalar(temperature_objective,bounds=(-1.5,1.5),method='bounded');temperature=float(np.exp(optimum.x))
        predictions={name:logits_to_cdf(torch.as_tensor(raw_test)/temp).numpy() for name,temp in [('raw',1.),('temperature',temperature)]}
        np.savez_compressed(folder/'predictions.npz',query_ids=test,logits=raw_test,**predictions);metrics=[]
        for state in sorted(q.state.unique()):
            keep=q.state.to_numpy()[test]==state;ww=weight[test][keep]
            for name,cdf in predictions.items():
                metrics.append({'state':int(state),'variant':name,'integrated_brier':float(np.average(((cdf[keep]-labels[test][keep])**2).mean(1),weights=ww)),
                                'brier_by_horizon':np.average((cdf[keep]-labels[test][keep])**2,axis=0,weights=ww).tolist()})
        (folder/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n');(folder/'training_log.json').write_text(json.dumps(logs,indent=2)+'\n')
        manifest.update(status='complete',best_epoch=best_epoch,temperature=temperature,calibration_integrated_brier=best,elapsed_s=time.monotonic()-start,parameters=sum(p.numel() for p in model.parameters()))
        (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'fold':fold,'metrics':metrics,'elapsed_s':time.monotonic()-start}),flush=True)


if __name__=='__main__':main()
