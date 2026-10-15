"""Package local review artifacts, frozen models and evidence without raw logs."""
import datetime,hashlib,json,subprocess,sys,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from otg.protocol import verify_final_protocol
ROOT=Path(__file__).resolve().parents[1]


def main():
    protocol,digest=verify_final_protocol(ROOT)
    for name in ['paper/main.pdf','paper/generated/evidence_manifest.json','reports/final_artifact_audit.json','REVIEW_NOTES.md']:
        if not (ROOT/name).is_file():raise RuntimeError(f'Review package incomplete: {name}')
    if json.loads((ROOT/'reports/final_artifact_audit.json').read_text())['status']!='passed':raise RuntimeError('Final audit failed')
    files={ROOT/r['path'] for r in protocol['frozen_files']}
    tracked=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    files.update(ROOT/p for p in tracked if p and not p.startswith('.git'))
    for directory in ['paper','reports']:
        files.update(p for p in (ROOT/directory).rglob('*') if p.is_file() and not any(part in ['tools','preview','__pycache__'] for part in p.parts) and p.suffix not in ['.aux','.log','.out','.bbl','.blg'])
    for dataset in ['borg','bpi']:
        for run in (ROOT/'runs').glob(f'joint_{dataset}_selected_v2_*'):
            files.update(run.glob('final_cohort*'))
        empirical=ROOT/'runs'/f'aggregate_residual_{dataset}_selected_v2';files.update(p for p in empirical.iterdir() if p.is_file())
    files.update(ROOT/name for name in ['REPRODUCE.md','REVIEW_NOTES.md','requirements-research.txt'])
    files={p for p in files if p.is_file()};out=ROOT/'releases';out.mkdir(exist_ok=True);archive=out/'operational_cohorts_review.zip'
    if archive.exists():raise FileExistsError('Preserve prior review archive')
    records=[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(files)]
    manifest={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'protocol_sha256':digest,'scope':'Local manuscript, source, frozen model/selection assets and numerical evidence. Raw/provider datasets and full final feature/history arrays are excluded. Not externally published.','files':records}
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(files):z.write(p,str(p.relative_to(ROOT)))
        z.writestr('RELEASE_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(archive) as z:
        bad=z.testzip()
        if bad:raise RuntimeError(f'Corrupt archive member: {bad}')
    manifest.update(archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),archive_bytes=archive.stat().st_size)
    (out/'RELEASE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'archive':str(archive),'files':len(files),'bytes':archive.stat().st_size,'sha256':manifest['archive_sha256']}))


if __name__=='__main__':main()
