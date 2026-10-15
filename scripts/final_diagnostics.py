"""Predeclared final marginal diagnostics and cutoff-feature strata."""
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from otg.protocol import verify_final_protocol
ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['borg','bpi'],required=True);a=ap.parse_args();_,protocol_hash=verify_final_protocol(ROOT)
    data=ROOT/'data/processed'/f'{a.dataset}_joint_selected_v2';meta=json.loads((data/'metadata.json').read_text());cm=json.loads((data/'final_cohorts/manifest.json').read_text());scaler=np.load(data/'scaler.npz');rows=[];reliability=[];cohorts=[]
    unit='m' if a.dataset=='borg' else 'd';names=meta['features']
    for c in cm['cohorts']:
        folder=data/'final_cohorts'/c['folder'];q=pd.read_parquet(folder/'queries.parquet');p=np.load(folder/'cdf.npy');x=np.load(ROOT/'data/processed'/f'{a.dataset}_final_cohorts_v1'/c['folder']/'features.npy');y=q[[f'exit_{h}{unit}' for h in meta['horizons']]].to_numpy();states=q.state.to_numpy()
        cohorts.append({'cutoff_us':c['cutoff_us'],'n':len(q),'n_states':q.state.nunique(),**{f'n_{r}_groups':q.loc[q[r]>=0,r].nunique() for r in meta['relations']}})
        strata={'all':np.ones(len(q),bool)}
        if a.dataset=='borg':
            for name in ['unknown_state_age','imputed_ever']:
                k=names.index(name);known=x[:,k]>.5
                strata[name+'_yes']=known;strata[name+'_no']=~known
        for stratum,keep in strata.items():
            for state in sorted(np.unique(states)):
                ids=keep&(states==state)
                if not ids.any():continue
                for k,h in enumerate(meta['horizons']):
                    rows.append({'cutoff_us':c['cutoff_us'],'stratum':stratum,'state':int(state),'horizon':h,'n':int(ids.sum()),'brier':float(np.mean((p[ids,k]-y[ids,k])**2)),'predicted_fraction':float(p[ids,k].mean()),'observed_fraction':float(y[ids,k].mean())})
                    if stratum=='all':
                        bins=np.minimum((p[:,k]*10).astype(int),9)
                        for b in range(10):
                            selected=ids&(bins==b)
                            if selected.any():reliability.append({'cutoff_us':c['cutoff_us'],'state':int(state),'horizon':h,'bin':b,'n':int(selected.sum()),'mean_prediction':float(p[selected,k].mean()),'fraction_positive':float(y[selected,k].mean())})
    out=ROOT/'reports'/f'final_diagnostics_{a.dataset}';out.mkdir(exist_ok=False)
    for name,values in [('marginal_scores',rows),('reliability',reliability),('cohorts',cohorts)]:pd.DataFrame(values).to_csv(out/f'{name}.csv',index=False)
    report={'dataset':a.dataset,'n_origins':len(cm['cohorts']),'frozen_protocol_sha256':protocol_hash,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'status':'complete','scope':'Strata use cutoff-known features. This evaluates marginal robustness; it does not imply conditional joint calibration.'}
    (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
