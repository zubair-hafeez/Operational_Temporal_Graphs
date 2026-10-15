"""Verification gate for the once-frozen final evaluation protocol."""
import hashlib,json
from pathlib import Path


def verify_final_protocol(root):
    root=Path(root);path=root/'docs/FINAL_PROTOCOL.json'
    if not path.exists():raise RuntimeError('Final data remain locked: no frozen evaluation manifest')
    protocol=json.loads(path.read_text())
    if protocol.get('status')!='frozen':raise RuntimeError('Final protocol is not frozen')
    for item in protocol['frozen_files']:
        file=root/item['path']
        if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest()!=item['sha256']:
            raise RuntimeError(f'Frozen artifact changed or missing: {item["path"]}')
    return protocol,hashlib.sha256(path.read_bytes()).hexdigest()
