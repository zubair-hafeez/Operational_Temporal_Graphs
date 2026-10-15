"""Scoped GPU runner for authorized revision tasks; original runner stays frozen."""
import argparse,datetime,hashlib,json,os,subprocess,sys,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PROGRAMS={'collective-risk':'collective_risk.py','fresh-fit':'fit_bpi2012_joint.py','joint-profiles':'evaluate_joint_profiles.py'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('task',choices=list(PROGRAMS));ap.add_argument('arguments',nargs=argparse.REMAINDER);a=ap.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
    script=ROOT/'revision'/PROGRAMS[a.task];stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'_'+uuid.uuid4().hex[:8];log=ROOT/'reports'/f'revision_runner_{a.task}_{stamp}.log';command=[sys.executable,str(script),*a.arguments]
    record={'command':command,'script_sha256':hashlib.sha256(script.read_bytes()).hexdigest(),'log':str(log)};print(json.dumps(record),flush=True)
    with log.open('x') as f:
        f.write(json.dumps(record)+'\n');f.flush();result=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,shell=False)
    print(json.dumps({'returncode':result.returncode,'log':str(log)}),flush=True);sys.exit(result.returncode)


if __name__=='__main__':main()
