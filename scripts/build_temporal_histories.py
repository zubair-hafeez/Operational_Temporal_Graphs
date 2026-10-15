"""Build strictly past event tokens for temporal neural survival baselines.

No third-party source code is copied. The Mixer baseline will be an explicit
survival-task adaptation of the GraphMixer temporal-link encoder idea.
"""
from __future__ import annotations
import argparse,hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from otg.history import last_event_indices
ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['borg','bpi'],required=True);ap.add_argument('--length',type=int,default=16);a=ap.parse_args()
    version='history_v1' if a.dataset=='borg' else 'horizons_v1'
    source=ROOT/'data/processed'/f'{a.dataset}_development_{version}';meta=json.loads((source/'metadata.json').read_text())
    prefix=ROOT/'data/processed'/('borg_prefix_22d' if a.dataset=='borg' else 'bpi_before_20161101')
    out=ROOT/'data/processed'/f'{a.dataset}_temporal_histories_k{a.length}_v1';out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    columns=['time_us','task','job','machine','event_type','cpu','memory','priority','sched_class'] if a.dataset=='borg' else ['time_us','case','resource','activity','lifecycle']
    e=pd.read_parquet(prefix/'events.parquet',columns=columns);q=pd.read_parquet(source/'queries.parquet')
    tables=[('queries',q)]
    if a.dataset=='borg':
        tables += [(c['folder'],pd.read_parquet(source/c['folder']/'queries.parquet')) for c in meta['cohorts'] if c['elapsed_day']>=20]
    scopes=['task','job','machine'] if a.dataset=='borg' else ['case','resource']
    times=e.time_us.to_numpy();np.save(out/'event_times.npy',np.r_[0,times])
    if a.dataset=='borg':
        categorical=np.column_stack([e.event_type.to_numpy()+1,np.zeros(len(e),np.int16)])
        raw=e[['cpu','memory','priority','sched_class']].to_numpy(np.float32)
        continuous=np.column_stack([np.nan_to_num(raw,nan=0),np.isnan(raw)]).astype(np.float32)
        continuous[:,:4]=np.clip(continuous[:,:4],-100,100)/np.array([1,1,12,3])
        vocab={'event_type':[str(x) for x in range(9)],'lifecycle':[]};vocab_sizes=[10,1]
    else:
        # Vocabulary from the earliest fitting period; later unseen categories
        # map to the reserved unknown value, without exposing future names.
        early=e.time_us.to_numpy()<pd.Timestamp('2016-05-01',tz='UTC').value//1000
        vocab={c:sorted(e.loc[early,c].unique()) for c in ['activity','lifecycle']}
        categorical=np.column_stack([e[c].map({v:i+1 for i,v in enumerate(vocab[c])}).fillna(0).to_numpy(np.int16) for c in vocab])
        continuous=np.zeros((len(e),1),np.float32);vocab_sizes=[len(v)+1 for v in vocab.values()]
    np.save(out/'event_categories.npy',np.vstack([np.zeros((1,2),np.int16),categorical]).astype(np.int16))
    np.save(out/'event_values.npy',np.vstack([np.zeros((1,continuous.shape[1]),np.float32),continuous]))
    arrays={}
    for name,frame in tables:
        arrays[name]=np.lib.format.open_memmap(out/f'{name}_indices.npy',mode='w+',dtype=np.int32,shape=(len(frame),len(scopes),a.length))
    all_ids={scope:np.concatenate([f[scope].to_numpy() for _,f in tables]) for scope in scopes}
    query_times=np.concatenate([f.cutoff_us.to_numpy() for _,f in tables])
    for column,scope in enumerate(scopes):
        indices=last_event_indices(e[scope].to_numpy(),times,all_ids[scope],query_times,a.length)+1
        lo=0
        for name,frame in tables:arrays[name][:,column]=indices[lo:lo+len(frame)];lo+=len(frame);arrays[name].flush()
        print(json.dumps({'scope':scope,'queries':len(query_times),'elapsed_s':time.monotonic()-start}),flush=True)
    report={'config':vars(a),'source_dataset':str(source.relative_to(ROOT)),'scopes':scopes,'vocab':vocab,'vocab_sizes':vocab_sizes,'event_continuous_dim':continuous.shape[1],
      'events':len(e),'tables':{name:len(f) for name,f in tables},'contract':'Last K observed events at or before cutoff for entity and cutoff-observed relation; oldest to newest; zero is padding; unknown relation never retrieves events.',
      'source_metadata_sha256':hashlib.sha256((source/'metadata.json').read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'retrieval_sha256':hashlib.sha256((ROOT/'otg/history.py').read_bytes()).hexdigest(),'elapsed_s':time.monotonic()-start}
    (out/'metadata.json').write_text(json.dumps(report,indent=2)+'\n');(ROOT/'reports'/f'{a.dataset}_temporal_histories_k{a.length}.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
