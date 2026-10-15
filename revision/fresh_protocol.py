"""Immutable gate for the new February 2012 confirmation period."""
import json
from revision.common import ROOT,digest


def verify_fresh_protocol():
    p=ROOT/'revision/FRESH_FINAL_PROTOCOL.json'
    if not p.exists():raise RuntimeError('Fresh final outcomes remain locked: no frozen protocol')
    m=json.loads(p.read_text())
    if m['status']!='frozen':raise RuntimeError('Fresh protocol is not frozen')
    for item in m['files']:
        f=ROOT/item['path']
        if not f.is_file() or digest(f)!=item['sha256']:raise RuntimeError('Fresh frozen asset changed: '+item['path'])
    return m
