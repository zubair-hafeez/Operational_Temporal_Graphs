"""Run the fixed three-seed development controls, one worker per GPU.

Completed runs are verified and skipped; partial runs are never overwritten.
This script never opens final-period data.
"""
from __future__ import annotations
import argparse,concurrent.futures,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def complete_temporal(dataset,architecture,seed):
    base=ROOT/'runs'/f'temporal_{dataset}_{architecture}_seed{seed}'
    folds=['oof8_10','oof12_14','oof16_18','development20_22'] if dataset=='borg' else ['oof_june','oof_july','oof_august','development_october']
    return all((base/f/'manifest.json').exists() and json.loads((base/f/'manifest.json').read_text())['status']=='complete' for f in folds)


def worker(dataset,suite,data_tag=None):
    gpu=0 if dataset=='borg' else 1;history=['--history'] if dataset=='borg' else [];tag=data_tag or ('history_v1' if history else 'v1')
    jobs=[]
    if suite=='temporal':
        for architecture in ['mixer','gru','mlp']:
            for seed in range(3):
                name=f'train_temporal_{dataset}_{architecture}_seed{seed}'
                if complete_temporal(dataset,architecture,seed):print(f'Skip completed {name}',flush=True);continue
                jobs.append((name,[sys.executable,'scripts/train_temporal_marginals.py','--dataset',dataset,'--architecture',architecture,'--seed',str(seed),'--device',f'cuda:{gpu}']))
    else:
        for family in ['gaussian','commonshock']:
            for seed in range(3):
                name=f'train_joint_{dataset}_{tag}_{family}_seed{seed}'
                manifest=ROOT/'runs'/f'joint_{dataset}_{tag}_{family}_seed{seed}'/'manifest.json'
                if manifest.exists() and json.loads(manifest.read_text())['status']=='complete':print(f'Skip completed {name}',flush=True);continue
                jobs.append((name,[sys.executable,'scripts/train_joint_development.py','--dataset',dataset,*history,'--data-tag',tag,'--family',family,'--seed',str(seed),'--device',f'cuda:{gpu}']))
    for name,cmd in jobs:
        log=ROOT/'reports'/f'{name}.log'
        # Preserve any failed/incomplete logs instead of silently retrying.
        if log.exists():raise FileExistsError(f'Unfinished run requires inspection: {log}')
        print(json.dumps({'start':name,'gpu':gpu}),flush=True);start=time.monotonic()
        with log.open('x') as f:subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
        print(json.dumps({'complete':name,'elapsed_s':time.monotonic()-start}),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--suite',choices=['temporal','joint'],required=True)
    ap.add_argument('--datasets',nargs='+',choices=['borg','bpi'],default=['borg','bpi']);ap.add_argument('--data-tag',choices=['v1','history_v1','selected_v2']);a=ap.parse_args()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(worker,d,a.suite,a.data_tag) for d in a.datasets]
        for future in futures:future.result()


if __name__=='__main__':main()
