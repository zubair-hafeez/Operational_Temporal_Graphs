"""Inference for frozen marginal selections, reusable on untouched cohorts."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
from scipy.special import expit,logit
import torch
from .temporal import TemporalSurvival,logits_to_cdf


class FrozenForecast:
    def __init__(self,root,dataset,selections,horizons,device='cpu',history_directory=None):
        self.root=Path(root);self.dataset=dataset;self.horizons=horizons;self.device=torch.device(device)
        self.fold='development20_22' if dataset=='borg' else 'development_october';self.unit='m' if dataset=='borg' else 'd';self.models=[]
        self.history=Path(history_directory) if history_directory else self.root/'data/processed'/f'{dataset}_temporal_histories_k16_v1'
        for choice in selections:
            if choice['kind']=='hgb':
                base=self.root/'runs'/f'{dataset}_marginals_{choice["tag"]}_state{choice["state"]}_{choice["arm"]}_seed0'/self.fold
                fitted=[joblib.load(base/f'horizon{h}{self.unit}.joblib') for h in horizons];audit=json.loads((base/'calibration_audit.json').read_text())
                self.models.append((choice,[(fitted,audit)]))
            else:
                members=[]
                for seed in choice['seeds']:
                    base=self.root/'runs'/f'temporal_{dataset}_{choice["architecture"]}_seed{seed}'/self.fold
                    m=json.loads((base/'manifest.json').read_text());scaler=np.load(base/'scaler.npz')
                    model=TemporalSurvival(m['input_dim'],m['vocab_sizes'],m['continuous_dim'],m['n_scopes'],width=m['width'],architecture=choice['architecture']).to(self.device)
                    model.load_state_dict(torch.load(base/'model.pt',weights_only=True,map_location=self.device));model.eval()
                    members.append((model,scaler['mean'],scaler['scale'],m['temperature'] if choice['variant']=='temperature' else 1.))
                self.models.append((choice,members))
        self.assets=None
        if any(c['kind']=='temporal_ensemble' and c['architecture']!='mlp' for c,_ in self.models):
            self.assets={
                'categories':torch.as_tensor(np.load(self.history/'event_categories.npy').astype(np.int64),device=self.device),
                'values':torch.as_tensor(np.load(self.history/'event_values.npy'),device=self.device),
                'times':torch.as_tensor(np.load(self.history/'event_times.npy'),device=self.device)}

    def predict(self,q,x,indices=None,batch_size=2048):
        result=np.zeros((len(q),len(self.horizons)),np.float32)
        for choice,members in self.models:
            ids=np.flatnonzero(q.state.to_numpy()==choice['state'])
            if not len(ids):continue
            if choice['kind']=='hgb':
                fitted,audit=members[0];raw=[];individual=[]
                for model in fitted:
                    pp=np.clip(model['model'].predict_proba(x[ids][:,model['cols']])[:,1],1e-5,1-1e-5)
                    raw.append(pp);individual.append(model['calibrator'].predict_proba(logit(pp)[:,None])[:,1])
                raw=np.sort(np.column_stack(raw),axis=1);variant=choice['variant']
                if variant=='raw_sorted':p=raw
                elif variant=='individual_platt_sorted':p=np.sort(np.column_stack(individual),axis=1)
                else:
                    a,b=audit['params'][variant];p=expit(a*logit(np.clip(raw,1e-6,1-1e-6))+b)
                result[ids]=p
            else:
                predictions=[]
                with torch.no_grad():
                    for model,mean,scale,temperature in members:
                        parts=[]
                        for lo in range(0,len(ids),batch_size):
                            ix=ids[lo:lo+batch_size];xx=torch.as_tensor(np.clip((x[ix]-mean)/scale,-10,10).astype(np.float32),device=self.device)
                            if choice['architecture']=='mlp':logits=model(xx)
                            else:
                                if indices is None:raise ValueError('Temporal model requires causal event indices')
                                event=torch.as_tensor(indices[ix].astype(np.int64),device=self.device);mask=event!=0
                                qt=torch.as_tensor(q.cutoff_us.to_numpy()[ix],device=self.device)
                                lag=((qt[:,None,None]-self.assets['times'][event]).clamp_min(0)/1e6).float()*mask
                                logits=model(xx,self.assets['categories'][event],self.assets['values'][event],lag,mask)
                            parts.append(logits_to_cdf(logits/temperature).cpu().numpy())
                        predictions.append(np.concatenate(parts))
                result[ids]=np.mean(predictions,axis=0)
        if not np.isfinite(result).all() or np.any(np.diff(result,axis=1)<-1e-6):raise ValueError('Invalid predicted CDF')
        return result
