"""Fetch public research sources with pinned GCS generations and checksums."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import shutil
import time
from pathlib import Path
import xml.etree.ElementTree as ET
import requests

ROOT = Path(__file__).resolve().parents[1]
BUCKET = 'https://storage.googleapis.com/clusterdata-2011-2'


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def fetch(item):
    dest = ROOT / item['path']
    dest.parent.mkdir(parents=True, exist_ok=True)
    def valid():
        return dest.exists() and (not item.get('bytes') or dest.stat().st_size == item['bytes']) and (not item.get('md5') or digest(dest, 'md5') == item['md5']) and (not item.get('sha256') or digest(dest) == item['sha256'])
    if not valid():
        for attempt in range(3):
            part = dest.with_name(dest.name+'.partial')
            try:
                with requests.get(item['url'], timeout=(20,120), stream=True) as r:
                    r.raise_for_status()
                    with part.open('wb') as f:
                        for block in r.iter_content(1 << 20):
                            f.write(block)
                if item.get('bytes') and part.stat().st_size != item['bytes']:
                    raise ValueError('download size mismatch '+str(dest))
                if item.get('md5') and digest(part,'md5') != item['md5']:
                    raise ValueError('download MD5 mismatch '+str(dest))
                if item.get('sha256') and digest(part) != item['sha256']:
                    raise ValueError('download SHA256 mismatch '+str(dest))
                part.replace(dest)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(attempt+1)
    return {**item,'bytes':dest.stat().st_size,'sha256':digest(dest)}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--borg-parts',type=int,default=500)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if not 0 <= args.borg_parts <= 500:
        raise ValueError('borg-parts must be in [0,500]')
    raw=ROOT/'data/raw';raw.mkdir(parents=True,exist_ok=True)
    # Import previously verified public downloads without altering originals.
    for src,rel in [('/tmp/borg_task_events_00000.csv.gz','borg/task_events/part-00000-of-00500.csv.gz'),('/tmp/borg_task_events_00001.csv.gz','borg/task_events/part-00001-of-00500.csv.gz'),('/tmp/bpi2017.xes.gz','bpi/bpi2017.xes.gz')]:
        dst=raw/rel
        if Path(src).exists() and not dst.exists():
            dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
    r=requests.get(BUCKET+'?prefix=task_events/',timeout=60);r.raise_for_status()
    (raw/'borg/listing.xml').write_bytes(r.content)
    listing=ET.fromstring(r.content)
    if any(x.tag.endswith('IsTruncated') and x.text == 'true' for x in listing):
        raise RuntimeError('Unexpected paginated object listing')
    objects=[]
    for e in listing:
        if e.tag.split('}')[-1]=='Contents':
            x={a.tag.split('}')[-1]:a.text for a in e}
            objects.append(dict(url=BUCKET+'/'+x['Key']+'?generation='+x['Generation'], path='data/raw/borg/'+x['Key'],bytes=int(x['Size']),md5=x['ETag'].strip('"'),generation=x['Generation']))
    objects.sort(key=lambda x:x['path'])
    tasks=objects[:args.borg_parts]
    tasks.extend([
        dict(url=BUCKET+'/machine_events/part-00000-of-00001.csv.gz',path='data/raw/borg/machine_events.csv.gz'),
        dict(url=BUCKET+'/schema.csv',path='data/raw/borg/schema.csv'),
        dict(url='https://api.figshare.com/v2/articles/12696884',path='data/raw/bpi/metadata.json'),
        dict(url='https://ndownloader.figshare.com/files/24044117',path='data/raw/bpi/bpi2017.xes.gz',bytes=29658747,sha256='183c5e5189282779c811c78c33ff936351b3dd201165d612211fc220936f8249'),
    ])
    print(json.dumps({'scheduled_files':len(tasks),'borg_task_bytes':sum(t['bytes'] for t in tasks[:args.borg_parts])}),flush=True)
    results=[]
    with ThreadPoolExecutor(max_workers=min(args.workers,8)) as ex:
        futures={ex.submit(fetch,item):item for item in tasks}
        for i,f in enumerate(as_completed(futures),1):
            results.append(f.result())
            if i%25==0 or i==len(tasks):
                print(json.dumps({'complete':i,'total':len(tasks)}),flush=True)
    results.sort(key=lambda x:x['path'])
    report=dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),files=results,licenses={'borg':'CC-BY-4.0','bpi2017':'4TU General Terms of Use; see source metadata'})
    manifest=ROOT/'reports/source_manifest.json'
    manifest.write_text(json.dumps(report,indent=2)+'\n')
    print(manifest,flush=True)


if __name__=='__main__':
    main()
