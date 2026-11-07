"""Persist sample feasibility evidence, without predictive holdout evaluation."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from extension.prepare_pilots import RAW, REPORT, read_zip, seconds


def main():
    reports={}
    google=pq.read_table(RAW/"google2019_a_instance_000000000000.parquet.gz",columns=["time","type","collection_id","instance_index","machine_id","missing_type"]).to_pandas()
    sizes=google.groupby(["collection_id","instance_index"]).size()
    reports["google2019"]={"rows":len(google),"entities":len(sizes),"singleton_entity_fraction":float((sizes==1).mean()),
        "sentinel_end_rows":int((google.time==np.iinfo(np.int64).max).sum()),"types":{str(k):int(v) for k,v in google.type.value_counts().items()},
        "decision":"schema pilot only: isolated export shard is not complete task or machine histories",
        "next":"requires whole-entity extraction across shards or bounded BigQuery export before valid forecasts",
        "relationships":["collection/job","machine","allocation"],
        "source":"https://github.com/google/cluster-data/blob/master/ClusterData2019.md"}
    change=read_zip("BPIC-2014.zip","Detail Change.csv",sep=";")
    opened=seconds(change["Change record Open Time"],True)
    closed=seconds(change["Change record Close Time"],True)
    reports["bpi2014_changes"]={"rows":len(change),"distinct_changes":int(change["Change ID"].nunique()),
        "snapshot_columns":list(change.columns),"negative_or_missing_closure":int(((closed<opened)|(closed<0)).sum()),
        "decision":"snapshot-schema pilot; planned times, affected CI and risk have no version history; no causal graph claim",
        "alternative":"timestamped incident activities from same release support team relation replay",
        "source":"https://ais.win.tue.nl/bpi/2014/challenge.html"}
    for source in ["azure","citibike","uci","bpi2014","airline"]:
        path=REPORT/f"prepared_{source}.json"
        if not path.exists():continue
        m=json.loads(path.read_text())
        reports[source]={k:v for k,v in m.items() if k not in ["cohorts"]}
        reports[source]["origin_counts"]={part:sum(c["part"]==part for c in m["cohorts"]) for part in ["train","calibration","dependence","validation"]}
    report={"datasets":reports,"raw_bytes":sum(p.stat().st_size for p in RAW.iterdir() if p.is_file()),
            "code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "bpi2013_status":"primary endpoint timed out; failure record retained; UCI and BPI2014 IT logs acquired"}
    (REPORT/"source_feasibility.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"raw_bytes":report["raw_bytes"],"datasets":list(reports)}),flush=True)


if __name__=="__main__":main()
