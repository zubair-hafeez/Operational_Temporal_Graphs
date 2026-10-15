"""Apply the frozen temporal-token contract to final observed cohorts."""
import argparse,hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from otg.history import last_event_indices
from otg.protocol import verify_final_protocol
ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['borg','bpi'],required=True);a=ap.parse_args();_,protocol_hash=verify_final_protocol(ROOT);start=time.monotonic()
    source=ROOT/'data/processed'/f'{a.dataset}_final_cohorts_v1';sm=json.loads((source/'metadata.json').read_text());prefix=ROOT/sm['prefix']
    old=ROOT/'data/processed'/f'{a.dataset}_temporal_histories_k16_v1';hm=json.loads((old/'metadata.json').read_text())
    out=ROOT/'data/processed'/f'{a.dataset}_final_histories_k16_v1';out.mkdir(exist_ok=False)
    cols=['time_us','task','job','machine','event_type','cpu','memory','priority','sched_class'] if a.dataset=='borg' else ['time_us','case','resource','activity','lifecycle']
    e=pd.read_parquet(prefix/'events.parquet',columns=cols);times=e.time_us.to_numpy();np.save(out/'event_times.npy',np.r_[0,times])
    if a.dataset=='borg':
        categorical=np.column_stack([e.event_type.to_numpy()+1,np.zeros(len(e),np.int16)]);raw=e[['cpu','memory','priority','sched_class']].to_numpy(np.float32)
        continuous=np.column_stack([np.nan_to_num(raw,nan=0),np.isnan(raw)]).astype(np.float32);continuous[:,:4]=np.clip(continuous[:,:4],-100,100)/np.array([1,1,12,3])
    else:
        categorical=np.column_stack([e[c].map({v:i+1 for i,v in enumerate(hm['vocab'][c])}).fillna(0).to_numpy(np.int16) for c in ['activity','lifecycle']]);continuous=np.zeros((len(e),1),np.float32)
    np.save(out/'event_categories.npy',np.vstack([np.zeros((1,2),np.int16),categorical]).astype(np.int16));np.save(out/'event_values.npy',np.vstack([np.zeros((1,continuous.shape[1]),np.float32),continuous]))
    tables=[(c['folder'],pd.read_parquet(source/c['folder']/'queries.parquet')) for c in sm['cohorts']];scopes=hm['scopes'];arrays={}
    for name,q in tables:arrays[name]=np.lib.format.open_memmap(out/f'{name}_indices.npy',mode='w+',dtype=np.int32,shape=(len(q),len(scopes),16))
    qt=np.concatenate([q.cutoff_us.to_numpy() for _,q in tables])
    for column,scope in enumerate(scopes):
        qi=np.concatenate([q[scope].to_numpy() for _,q in tables]);indices=last_event_indices(e[scope].to_numpy(),times,qi,qt,16)+1;lo=0
        for name,q in tables:arrays[name][:,column]=indices[lo:lo+len(q)];lo+=len(q);arrays[name].flush()
        print(json.dumps({'scope':scope,'queries':len(qt),'elapsed_s':time.monotonic()-start}),flush=True)
    report={**hm,'period':'final','source_dataset':str(source.relative_to(ROOT)),'events':len(e),'tables':{name:len(q) for name,q in tables},'frozen_protocol_sha256':protocol_hash,
            'frozen_history_metadata_sha256':hashlib.sha256((old/'metadata.json').read_bytes()).hexdigest(),'source_metadata_sha256':hashlib.sha256((source/'metadata.json').read_bytes()).hexdigest(),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'elapsed_s':time.monotonic()-start}
    (out/'metadata.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ['period','events','tables','elapsed_s']}),flush=True)


if __name__=='__main__':main()
