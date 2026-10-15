"""Open and prepare the final periods after the immutable protocol check."""
import argparse,concurrent.futures,datetime,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from otg.protocol import verify_final_protocol
ROOT=Path(__file__).resolve().parents[1]


def prepare(dataset):
    prefix='borg_prefix_29d' if dataset=='borg' else 'bpi_before_20170116'
    jobs=[(prefix,['prepare_borg.py','--through-days','29','--frozen-final'] if dataset=='borg' else ['prepare_bpi.py','--before','2017-01-16','--frozen-final']),
          (f'{dataset}_final_cohorts_v1',['build_final_cohorts.py','--dataset',dataset]),
          (f'{dataset}_final_histories_k16_v1',['build_final_histories.py','--dataset',dataset])]
    for name,args in jobs:
        manifest=ROOT/'data/processed'/name/'metadata.json'
        if manifest.exists():print(json.dumps({'skip_completed':name}),flush=True);continue
        log=ROOT/'reports'/f'prepare_final_{name}.log'
        with log.open('x') as f:subprocess.run([sys.executable,str(ROOT/'scripts'/args[0]),*args[1:]],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
        print(json.dumps({'completed':name,'log':str(log)}),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--datasets',nargs='+',choices=['borg','bpi'],default=['borg','bpi']);a=ap.parse_args()
    _,digest=verify_final_protocol(ROOT);state_path=ROOT/'STATE.json';state=json.loads(state_path.read_text())
    if not state['final_test_opened']:
        now=datetime.datetime.now(datetime.timezone.utc).isoformat();state.update(final_test_opened=True,final_opened_utc=now,final_protocol_sha256=digest,phase='frozen_final_data_preparation')
        state_path.write_text(json.dumps(state,indent=2)+'\n')
        with (ROOT/'LEDGER.md').open('a') as f:f.write(f'\n## {now} Final evaluation opened\n\nVerified frozen protocol SHA256 `{digest}`. Final preparation begins; model/architecture/threshold selection is closed. Preserve every outcome and any required bug-fix audit.\n')
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for future in [pool.submit(prepare,d) for d in a.datasets]:future.result()


if __name__=='__main__':main()
