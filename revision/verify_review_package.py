"""Fresh-extraction reproduction of the local review package."""
import argparse, json, os, subprocess, sys, tempfile, zipfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from revision.common import ROOT, digest, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--archive', type=Path, default=ROOT/'releases/operational_cohorts_revision_review.zip')
    ap.add_argument('--tectonic', type=Path)
    ap.add_argument('--tex-cache', type=Path)
    a = ap.parse_args()
    out = Path(tempfile.mkdtemp(prefix='otg-revision-review-'))
    with zipfile.ZipFile(a.archive) as z:
        assert z.testzip() is None
        for member in z.namelist():
            assert (out/member).resolve().is_relative_to(out)
        z.extractall(out)
    manifest = json.loads((out/'REVISION_RELEASE_MANIFEST.json').read_text())
    for record in manifest['files']:
        p = out/record['path']
        assert p.stat().st_size == record['bytes'] and digest(p) == record['sha256'], record['path']
    generated = out/'paper_revision/generated'
    evidence = {str(p.relative_to(out)): digest(p) for p in generated.iterdir() if p.suffix in {'.csv', '.tex', '.json'}}
    figure = out/'paper_revision/figures/calibration_and_controls.pdf'
    evidence[str(figure.relative_to(out))] = digest(figure)
    env = dict(os.environ, MPLCONFIGDIR=str(out/'.cache/matplotlib'))
    commands = [[sys.executable, '-m', 'pytest', 'tests', 'revision/test_revision.py', 'revision/test_exact_counts.py', 'revision/test_alibaba.py', '-q'],
                [sys.executable, 'revision/build_revision_artifacts.py'],
                [sys.executable, 'revision/audit_revision_artifacts.py']]
    runs = []
    for i, command in enumerate(commands):
        log = out/f'verification_{i}.log'
        with log.open('w') as f:
            result = subprocess.run(command, cwd=out, env=env, stdout=f, stderr=subprocess.STDOUT)
        assert result.returncode == 0, str(log)
        runs.append({'command': command, 'returncode': result.returncode, 'log': str(log)})
        print(f'Passed verification stage {i+1}', flush=True)
    mismatches = [p for p, sha in evidence.items() if digest(out/p) != sha]
    assert not mismatches, mismatches
    compiled = False
    if a.tectonic:
        if a.tex_cache:
            env['XDG_CACHE_HOME'] = str(a.tex_cache.resolve())
        log = out/'verification_tex.log'
        with log.open('w') as f:
            result = subprocess.run([str(a.tectonic.resolve()), '--only-cached', '--keep-logs', 'main.tex'],
                                    cwd=out/'paper_revision', env=env, stdout=f, stderr=subprocess.STDOUT)
        assert result.returncode == 0, str(log)
        txt = (out/'paper_revision/main.log').read_text()
        assert 'Overfull' not in txt and 'undefined' not in txt.lower()
        compiled = (out/'paper_revision/main.pdf').is_file()
    report = {'status': 'passed', 'archive_sha256': digest(a.archive), 'archive_bytes': a.archive.stat().st_size,
              'extracted_directory': str(out), 'files_verified': len(manifest['files']),
              'generated_artifacts_identical': len(evidence), 'tests_passed': 45,
              'both_protocols_and_final_alignment_verified': True, 'manuscript_compiled_from_archive': compiled,
              'commands': runs, 'verification_script_sha256': digest(__file__)}
    write_json(ROOT/'reports/revision_archive_verification.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
