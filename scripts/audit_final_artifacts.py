"""Read-only consistency audit of the frozen final evaluation and predictions."""
import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from otg.protocol import verify_final_protocol
ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    protocol,digest=verify_final_protocol(ROOT);frozen={r['path']:r['sha256'] for r in protocol['frozen_files']};results={}
    for dataset,expected,spacing in [('borg',28,6*3600_000_000),('bpi',61,86400_000_000)]:
        data=ROOT/'data/processed'/f'{dataset}_joint_selected_v2';directory=data/'final_cohorts';cm=json.loads((directory/'manifest.json').read_text());m=json.loads((data/'metadata.json').read_text());origins=[c['cutoff_us'] for c in cm['cohorts']]
        if len(origins)!=expected or not np.all(np.diff(origins)==spacing):raise ValueError('Wrong final origins')
        if cm['selected_metadata_sha256']!=frozen[str((data/'metadata.json').relative_to(ROOT))]:raise ValueError('Selected metadata changed')
        scal=np.load(data/'scaler.npz');unit='m' if dataset=='borg' else 'd';entity='task' if dataset=='borg' else 'case';n=0
        for c in cm['cohorts']:
            folder=directory/c['folder'];q=pd.read_parquet(folder/'queries.parquet');p=np.load(folder/'cdf.npy');x=np.load(folder/'features.npy');raw_folder=ROOT/'data/processed'/f'{dataset}_final_cohorts_v1'/c['folder'];raw=np.load(raw_folder/'features.npy')
            if sha(folder/'cdf.npy')!=c['cdf_sha256']:raise ValueError('Forecast changed')
            if sha(folder/'queries.parquet')!=sha(raw_folder/'queries.parquet'):raise ValueError('Prediction changed query/label file')
            if q[entity].nunique()!=len(q) or not (q.cutoff_us==c['cutoff_us']).all():raise ValueError('Invalid cohort membership')
            if p.shape!=(len(q),4) or not np.isfinite(p).all() or np.any((p<0)|(p>1)) or np.any(np.diff(p,axis=1)<-1e-7):raise ValueError('Invalid CDF')
            y=q[[f'exit_{h}{unit}' for h in m['horizons']]].to_numpy()
            if not np.isin(y,[0,1]).all() or np.any(np.diff(y,axis=1)<0):raise ValueError('Invalid first-exit label')
            expected_x=np.clip((raw-scal['mean'])/scal['scale'],-10,10).astype(np.float32)
            if not np.array_equal(x,expected_x):raise ValueError('Context features do not match frozen transform')
            n+=len(q)
        score_files=[]
        for family in ['gaussian','commonshock']:
            for seed in range(3):
                run=ROOT/'runs'/f'joint_{dataset}_selected_v2_{family}_seed{seed}'
                for draws in ([512,1024] if seed==0 else [512]):
                    score=run/f'final_cohort_scores_m{draws}.parquet';r=json.loads((run/f'final_cohort_report_m{draws}.json').read_text());d=pd.read_parquet(score)
                    if r['scores_sha256']!=sha(score):raise ValueError('Score file changed')
                    if r['script_sha256']!=frozen['scripts/evaluate_joint_development.py']:raise ValueError('Evaluation code was not frozen version')
                    if r['training_manifest_sha256']!=frozen[str((run/'manifest.json').relative_to(ROOT))]:raise ValueError('Checkpoint provenance changed')
                    if sorted(d.cutoff_us.unique())!=origins:raise ValueError('Missing final origins in scores')
                    if set(d['mode'])!={'independent','global','static','context','static_permuted'}:raise ValueError('Missing comparison')
                    if not np.isfinite(d.select_dtypes('number')).all().all():raise ValueError('Nonfinite score')
                    score_files.append({'path':str(score.relative_to(ROOT)),'sha256':sha(score)})
        results[dataset]={'origins':expected,'entity_origin_queries':n,'cdf_monotonicity':'passed','label_monotonicity':'passed','frozen_feature_transform':'exact','scores':score_files}
    correction=json.loads((ROOT/'reports/final_borg_timestamp_correction.json').read_text())
    if not all(x['all_values_exactly_preserved'] for x in correction['audits']):raise ValueError('Unaudited schema correction')
    report={'status':'passed','protocol_sha256':digest,'frozen_files_verified':len(frozen),'datasets':results,'timestamp_correction_sha256':sha(ROOT/'reports/final_borg_timestamp_correction.json'),'limitations':'Checks artifact consistency and causal input contracts implemented in code; cannot establish real-world calibration, causal validity or absence of every conceptual error.'}
    (ROOT/'reports/final_artifact_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='datasets'}))


if __name__=='__main__':main()
