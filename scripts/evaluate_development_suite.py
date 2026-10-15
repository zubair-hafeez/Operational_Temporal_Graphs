"""Score completed development controls; no training or final-period access."""
import argparse,concurrent.futures,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def evaluate(dataset,tag,family,seed,draws,period,device):
    run=ROOT/'runs'/f'joint_{dataset}_{tag}_{family}_seed{seed}'
    manifest=run/'manifest.json'
    if not manifest.exists() or json.loads(manifest.read_text())['status']!='complete':raise RuntimeError(f'Training not complete: {run}')
    prefix='full_cohort' if period=='development' else 'final_cohort'
    report=run/f'{prefix}_report_m{draws}.json'
    if report.exists():return {'skipped_complete':str(run)}
    log=ROOT/'reports'/f'evaluate_joint_{dataset}_{tag}_{family}_seed{seed}_m{draws}_{period}.log'
    command=[sys.executable,'scripts/evaluate_joint_development.py','--dataset',dataset,'--data-tag',tag,'--family',family,'--seed',str(seed),'--draws',str(draws),'--period',period,'--device',device]
    with log.open('x') as f:subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    return json.loads(report.read_text())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['borg','bpi'],required=True);ap.add_argument('--data-tag',choices=['v1','history_v1','selected_v2'],required=True)
    ap.add_argument('--draws',type=int,default=512);ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--period',choices=['development','final'],default='development');ap.add_argument('--device',default='cpu');ap.add_argument('--seeds',nargs='+',type=int,default=[0,1,2]);a=ap.parse_args()
    os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(evaluate,a.dataset,a.data_tag,family,seed,a.draws,a.period,a.device) for family in ['gaussian','commonshock'] for seed in a.seeds]
        for future in concurrent.futures.as_completed(futures):print(json.dumps(future.result()),flush=True)


if __name__=='__main__':main()
