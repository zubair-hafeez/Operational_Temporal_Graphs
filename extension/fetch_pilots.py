"""Bounded, resumable acquisition of explicitly selected public pilot assets.

Run with --source NAME. Downloads are immutable once complete. No extraction.
The Azure prefix option intentionally stores only complete decompressed rows;
that sample is for schema/feasibility, never an unbiased cohort benchmark.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import shutil
import time
import urllib.request
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/extension_20261021"
REPORT = ROOT / "reports/extension_20261021"
SOURCES = {
    "google2019": ("google2019_a_instance_000000000000.parquet.gz",
        "https://storage.googleapis.com/clusterdata_2019_a/instance_events-000000000000.parquet.gz", 32_000_000),
    "google_schema": ("google2019_schema.proto",
        "https://raw.githubusercontent.com/google/cluster-data/master/clusterdata_trace_format_v3.proto", 100_000),
    "azure_schema": ("azure_v2_schema.csv",
        "https://github.com/Azure/AzurePublicDataset/releases/download/dataset-v2/schema.csv", 100_000),
    "azure_prefix": ("azure_v2_vmtable_prefix.csv",
        "https://github.com/Azure/AzurePublicDataset/releases/download/dataset-v2/trace_data_vmtable_vmtable.csv.gz", 8_000_000),
    "azure_full": ("azure_v2_vmtable.csv.gz",
        "https://github.com/Azure/AzurePublicDataset/releases/download/dataset-v2/trace_data_vmtable_vmtable.csv.gz", 450_000_000),
    "citibike": ("JC-202401-citibike-tripdata.csv.zip",
        "https://s3.amazonaws.com/tripdata/JC-202401-citibike-tripdata.csv.zip", 4_000_000),
    "airline": ("bts_202401.zip",
        "https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_2024_1.zip", 40_000_000),
    "bpi2013": ("BPI_Challenge_2013_incidents.xes.gz",
        "https://data.4tu.nl/file/0fc5c579-e544-4fab-9143-fab1f5192432/aa51ffbb-25fd-4b5a-b0b8-9aba659b7e8c", 10_000_000),
    "bpi2014": ("BPIC-2014.zip", "https://fluxicon.com/blog/2014/04/bpi-challenge-2014/", 100_000_000),
    "uci_incidents": ("uci_incidents.zip",
        "https://archive.ics.uci.edu/static/public/498/incident+management+process+enriched+event+log.zip", 15_000_000),
}


def request(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={
        "User-Agent": "OTG-research-pilot/1.0", "Accept-Encoding": "identity"}), timeout=30)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(source):
    RAW.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    name, url, limit = SOURCES[source]
    path = RAW / name
    manifest = REPORT / f"source_{source}.json"
    if path.exists():
        old = json.loads(manifest.read_text())
        if old["sha256"] != sha(path):
            raise RuntimeError(f"Existing asset hash differs: {path}")
        print(json.dumps({"source": source, "status": "verified_existing", "bytes": path.stat().st_size}), flush=True)
        return
    used = sum(p.stat().st_size for p in RAW.rglob("*") if p.is_file())
    if used + limit > 2 * 1024**3 or shutil.disk_usage(RAW).free < 20 * 1024**3:
        raise RuntimeError("Pilot disk budget or free-space reserve would be exceeded")
    record = {"source": source, "requested_url": url, "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "script_sha256": sha(Path(__file__)), "limit_bytes": limit}
    start = time.monotonic()
    try:
        if source == "bpi2014":
            with request(url) as response:
                html = response.read(1_000_000).decode()
            links = re.findall(r'href=[\"\x27]([^\"\x27]+\.zip)[\"\x27]', html)
            url = next(u for u in links if "files.fluxicon.com" in u and "2014" in u).replace("http:", "https:")
        partial = path.with_suffix(path.suffix + ".partial")
        if partial.exists():
            raise RuntimeError(f"Partial file retained from prior failure: {partial}")
        with request(url) as response, partial.open("xb") as out:
            record.update(final_url=response.geturl().split("?")[0], content_length=response.headers.get("Content-Length"),
                          etag=response.headers.get("ETag"))
            if source != "azure_prefix" and record["content_length"] and int(record["content_length"]) > limit:
                raise RuntimeError("Published content length exceeds the source-specific byte cap")
            n = 0
            decoder = zlib.decompressobj(16 + zlib.MAX_WBITS) if source == "azure_prefix" else None
            pending = b""
            while n < limit:
                block = response.read(min(1 << 18, limit - n))
                if not block:
                    break
                n += len(block)
                if decoder:
                    pending += decoder.decompress(block)
                    if len(pending) > 8 * 1024**2:
                        raise RuntimeError("Unexpected decompression expansion")
                    end = pending.rfind(b"\n")
                    if end >= 0:
                        out.write(pending[:end + 1]); pending = pending[end + 1:]
                else:
                    out.write(block)
            if not decoder and response.read(1):
                raise RuntimeError("Download exceeded byte cap")
        partial.rename(path)
        record.update(status="complete", downloaded_bytes=n, stored_bytes=path.stat().st_size, sha256=sha(path),
                      sampling="first 8 MB compressed stream; complete rows only; schema audit" if decoder else "complete named asset",
                      elapsed_seconds=time.monotonic() - start)
        manifest.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record), flush=True)
    except Exception as error:
        record.update(status="failed", error=repr(error), elapsed_seconds=time.monotonic() - start)
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        (REPORT / f"source_{source}_failure_{stamp}.json").write_text(json.dumps(record, indent=2) + "\n")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=SOURCES, required=True)
    fetch(parser.parse_args().source)
