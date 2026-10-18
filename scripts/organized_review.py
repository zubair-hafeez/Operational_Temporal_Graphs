"""Verify deduplication or reconstruct a review ZIP from retained member bytes."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore', choices=['original', 'revision'])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if bool(args.restore) != bool(args.output):
        parser.error('--restore and --output must be supplied together')
    record = json.loads((ROOT / 'archives/ORGANIZATION_MANIFEST.json').read_text())
    checked = {}

    def verify(relative, expected, size):
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT.resolve()):
            raise ValueError('Retained file outside project')
        if path not in checked:
            h = hashlib.sha256()
            with path.open('rb') as f:
                for block in iter(lambda: f.read(1024 * 1024), b''):
                    h.update(block)
            checked[path] = (h.hexdigest(), path.stat().st_size)
        if checked[path] != (expected, size):
            raise ValueError('Retained file changed: ' + relative)
        return path

    for entry in record['entries']:
        verify(entry['destination'], entry['sha256'], entry['bytes'])
    for entry in record['links']:
        path = ROOT / entry['path']
        if not path.is_symlink() or path.resolve() != (ROOT / entry['target']).resolve():
            raise ValueError('Unexpected duplicate-link target: ' + entry['path'])
        verify(entry['path'], entry['sha256'], entry['bytes'])
    for entry in record['removed_checkpoints']:
        verify(entry['retained'], entry['sha256'], entry['bytes'])

    if args.restore:
        archive = next(a['path'] for a in record['archives'] if a['label'] == args.restore)
        entries = [e for e in record['entries'] if e['archive'] == archive]
        # Exclusive creation prevents replacing an existing artifact.
        with zipfile.ZipFile(args.output, 'x', compression=zipfile.ZIP_DEFLATED) as z:
            for entry in entries:
                member = Path(entry['member'])
                if member.is_absolute() or '..' in member.parts:
                    raise ValueError('Unsafe archive path')
                z.write(ROOT / entry['destination'], arcname=entry['member'])
        with zipfile.ZipFile(args.output) as z:
            for entry in entries:
                if hashlib.sha256(z.read(entry['member'])).hexdigest() != entry['sha256']:
                    raise ValueError('Reconstructed member differs')
        print(json.dumps({'status': 'review_member_bytes_restored', 'output': str(args.output),
                          'members': len(entries),
                          'note': 'Member bytes match the historical package. ZIP metadata/compression and archive SHA256 may differ.'}))
    else:
        print(json.dumps({'status': 'all_organized_artifacts_verified',
                          'release_members': len(record['entries']),
                          'deduplicated_paths': len(record['links']),
                          'retained_files_checked': len(checked)}))


if __name__ == '__main__':
    main()
