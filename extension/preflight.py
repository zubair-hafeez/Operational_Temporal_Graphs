"""Audit invariants, immutable prior studies, and saved-model inference before freeze."""
import contextlib,datetime,io,json,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from extension.freeze_confirmation import digest
from extension.confirm import predict


def main():
    report={"utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"frozen_prior_assets":0,"datasets":{},"roundtrip":{}}
    for name,key in [("docs/FINAL_PROTOCOL.json","frozen_files"),("revision/FRESH_FINAL_PROTOCOL.json","files")]:
        for entry in json.loads((ROOT/name).read_text())[key]:
            if digest(ROOT/entry["path"])!=entry["sha256"]:raise ValueError(f"Changed prior asset: {entry['path']}")
            report["frozen_prior_assets"]+=1
    for source in ["azure","azure_full","citibike","airline","uci","bpi2014","borg"]:
        folder=ROOT/"data/processed/extension_20261021"/source
        z=np.load(folder/"cohorts.npz");m=json.loads((folder/"manifest.json").read_text())
        assert all(np.isfinite(z[k]).all() for k in ["x","y","history"])
        assert np.all(np.diff(z["y"],axis=-1)>=0)
        assert set(np.unique(z["y"]))<=set([0,1])
        for c in m["cohorts"]:
            k=["train","calibration","dependence","validation"].index(c["part"])
            assert m["bounds"][k]<=c["origin"] and c["origin"]+max(m["horizons_seconds"])<m["bounds"][k+1]
        report["datasets"][source]={"origins":len(m["cohorts"]),"queries":len(z["y"]),"finite_and_monotone":True,"maturity_embargo_verified":True}
    # Full in-memory reload, with no fitting, checks the actual final inference path.
    with contextlib.redirect_stdout(io.StringIO()):
        predictions,_,_,_=predict("uci",ROOT/"data/processed/extension_20261021/uci","cpu")
    old=[np.load(ROOT/f"runs/extension_20261021/uci_seed{s}_replication_v2/predictions.npz") for s in [0,1,2]]
    for name,p in predictions.items():
        error=float(np.max(np.abs(p-np.mean([z[name] for z in old],axis=0))))
        assert error<2e-6
        report["roundtrip"][name]=error
    command=[sys.executable,"-m","pytest","tests","revision/test_revision.py","revision/test_exact_counts.py","revision/test_alibaba.py","extension/test_extension.py","-q"]
    result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
    report["tests"]={"command":command,"exit_code":result.returncode,"output":result.stdout+result.stderr}
    if result.returncode:raise RuntimeError(report["tests"]["output"])
    report["status"]="passed"
    target=ROOT/"reports/extension_20261021/pre_confirmation_audit.json"
    target.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report),flush=True)


if __name__=="__main__":main()
