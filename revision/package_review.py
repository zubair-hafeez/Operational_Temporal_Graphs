"""Create a separate local revision package, preserving the original release."""
import datetime, json, subprocess, sys, zipfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from revision.common import ROOT, digest, write_json
from revision.fresh_protocol import verify_fresh_protocol
from otg.protocol import verify_final_protocol


def main():
    original, original_hash = verify_final_protocol(ROOT)
    fresh = verify_fresh_protocol()
    assert json.loads((ROOT/'reports/revision_artifact_audit.json').read_text())['status'] == 'passed'
    required = ['paper_revision/main.pdf', 'REVISION_REVIEW_NOTES.md', 'REVISION_REPRODUCE.md']
    assert all((ROOT/p).is_file() for p in required)
    files = {ROOT/r['path'] for r in original['frozen_files']+fresh['files']}
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    files.update(ROOT/p for p in tracked if p)
    # Retain all original review evidence without nesting a redundant ZIP.
    with zipfile.ZipFile(ROOT/'releases/operational_cohorts_review.zip') as z:
        old_manifest = json.loads(z.read('RELEASE_MANIFEST.json'))
        files.update(ROOT/r['path'] for r in old_manifest['files'])
    for directory in ['revision', 'paper_revision', 'reports',
                      'data/processed/revision_bpi2012_final_v1',
                      'data/processed/revision_bpi_calibration_window_v1']:
        files.update(p for p in (ROOT/directory).rglob('*') if p.is_file())
    for directory in (ROOT/'runs').glob('revision_*'):
        files.update(p for p in directory.rglob('*') if p.is_file())
    files.update(ROOT/p for p in required)
    excluded_parts = {'.git', '__pycache__', '.ipynb_checkpoints', 'tools', 'preview'}
    files = {p for p in files if p.is_file()
             and not excluded_parts.intersection(p.relative_to(ROOT).parts)
             and p.suffix not in {'.pyc', '.aux', '.out', '.bbl', '.blg'}
             and p.name not in {'revision_archive_verification.json', 'REVISION_RELEASE_MANIFEST.json'}
             and not str(p.relative_to(ROOT)).startswith('releases/')}
    # TeX logs are build artifacts; experiment logs remain as evidence.
    files = {p for p in files if not (p.suffix == '.log' and p.relative_to(ROOT).parts[0] in {'paper', 'paper_revision'})}
    records = [{'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size, 'sha256': digest(p)} for p in sorted(files)]
    manifest = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip(),
                'original_protocol_sha256': original_hash,
                'fresh_protocol_sha256': digest(ROOT/'revision/FRESH_FINAL_PROTOCOL.json'),
                'scope': 'Local author-review package: manuscript, source, both frozen asset sets, original/revised scores, fresh BPI2012 data and fitted models. No external release. Large original logs/final arrays, unused candidate raw data and environment/TeX binaries excluded.',
                'third_party_terms': 'Included BPI2012 source retains 4TU General Terms of Use; source metadata included. This package does not relicense third-party data/software.',
                'files': records}
    target = ROOT/'releases/operational_cohorts_revision_review.zip'
    with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(files):
            z.write(p, str(p.relative_to(ROOT)))
        z.writestr('REVISION_RELEASE_MANIFEST.json', json.dumps(manifest, indent=2)+'\n')
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
    manifest.update(archive_sha256=digest(target), archive_bytes=target.stat().st_size)
    write_json(ROOT/'releases/REVISION_RELEASE_MANIFEST.json', manifest)
    print(json.dumps({'archive': str(target), 'files': len(records), 'bytes': target.stat().st_size, 'sha256': manifest['archive_sha256']}), flush=True)


if __name__ == '__main__':
    main()
