"""Freeze confirmation assets only after every development run is complete."""
import datetime,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from revision.common import ROOT,write_json,digest
from otg.protocol import verify_final_protocol


def main():
    target=ROOT/'revision/FRESH_FINAL_PROTOCOL.json'
    if target.exists():raise FileExistsError('Never replace a frozen protocol')
    if (ROOT/'data/processed/revision_bpi2012_final_v1').exists():raise RuntimeError('Final outcomes already opened')
    verify_final_protocol(ROOT)
    sources=['common.py','calibration.py','counts.py','reconcile.py','joint_profile.py','bpi2012_data.py','prepare_bpi2012.py','predict_bpi2012_final.py','evaluate_bpi2012.py','fresh_protocol.py','fit_bpi2012_marginals.py','fit_bpi2012_joint.py','test_revision.py','freeze_fresh_protocol.py']
    paths=[ROOT/'revision'/s for s in sources]+[ROOT/'revision/FRESH_FINAL_ANALYSIS.md',ROOT/'revision/FRESH_REPLICATION_DESIGN.md']
    for folder in ['data/processed/revision_bpi2012_development_v1','runs/revision_bpi2012_marginals_v1','runs/revision_bpi2012_development_evaluation_v1']+[f'runs/revision_bpi2012_joint_v1_seed{s}' for s in range(3)]:
        directory=ROOT/folder
        manifest=directory/('metadata.json' if folder.startswith('data') else 'manifest.json')
        m=json.loads(manifest.read_text())
        if folder.startswith('runs') and m['status']!='complete':raise RuntimeError('Incomplete: '+folder)
        paths.extend(p for p in directory.rglob('*') if p.is_file())
    paths.extend(ROOT/'otg'/name for name in ['bpi.py','copula.py','scores.py','cohort_scores.py','residuals.py','scenarios.py','torch_scenarios.py','protocol.py'])
    paths.extend([ROOT/'scripts/prepare_joint_development.py',ROOT/'data/raw/revision_bpi2012/metadata.json',ROOT/'data/raw/revision_bpi2012/bpi2012.xes.gz'])
    files=[{'path':str(p.relative_to(ROOT)),'sha256':digest(p),'bytes':p.stat().st_size} for p in sorted(set(paths))]
    write_json(target,{'status':'frozen','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'test_origins':'2012-02-01 12:00 UTC through 2012-02-28 12:00 UTC','observation_end':'2012-03-14 00:00 UTC','analysis':'revision/FRESH_FINAL_ANALYSIS.md','original_protocol_sha256':digest(ROOT/'docs/FINAL_PROTOCOL.json'),'raw_source_sha256':digest(ROOT/'data/raw/revision_bpi2012/bpi2012.xes.gz'),'files':files})
    print(json.dumps({'files':len(files),'protocol_sha256':digest(target)}),flush=True)


if __name__=='__main__':main()
