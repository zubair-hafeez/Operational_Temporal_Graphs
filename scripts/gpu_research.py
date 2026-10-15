"""Project-scoped entry point for the user's authorized GPU research.

Use an approval prefix consisting of the Python executable and this absolute
script path. Logging is internal so shell redirection does not turn each set of
experiment arguments into a separate whole-shell approval. Only the named
project experiment programs below can be launched; no shell is invoked.
"""
from __future__ import annotations
import argparse,datetime,hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PROGRAMS={
    'suite':'run_development_suite.py',
    'joint':'train_joint_development.py',
    'temporal':'train_temporal_marginals.py',
    'predict':'predict_selected_cohorts.py',
    'audit':'audit_temporal_inference.py',
    'evaluate':'evaluate_joint_development.py',
    'sampler-audit':'audit_gpu_scenarios.py',
    'evaluate-suite':'evaluate_development_suite.py',
}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('task',choices=['status',*PROGRAMS])
    ap.add_argument('experiment_arguments',nargs=argparse.REMAINDER)
    a=ap.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
    if a.task=='status':
        if a.experiment_arguments:ap.error('status takes no additional arguments')
        import torch
        print(json.dumps({'cuda_available':torch.cuda.is_available(),'gpus':[{'index':i,'name':torch.cuda.get_device_name(i)} for i in range(torch.cuda.device_count())]}))
        return
    script=ROOT/'scripts'/PROGRAMS[a.task]
    if not script.is_file():raise FileNotFoundError(script)
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    log=ROOT/'reports'/f'gpu_runner_{a.task}_{stamp}.log'
    command=[sys.executable,str(script),*a.experiment_arguments]
    metadata={'task':a.task,'command':command,'log':str(log),'program_sha256':hashlib.sha256(script.read_bytes()).hexdigest()}
    print(json.dumps(metadata),flush=True)
    with log.open('x') as output:
        output.write(json.dumps(metadata)+'\n');output.flush()
        result=subprocess.run(command,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,shell=False)
    print(json.dumps({'exit_code':result.returncode,'log':str(log)}),flush=True)
    sys.exit(result.returncode)


if __name__=='__main__':main()
