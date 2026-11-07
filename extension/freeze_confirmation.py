"""Freeze executable models and analysis choices before new-period scoring."""
import datetime,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCES=["uci","azure_full","bpi2014"]
PROTOCOL=ROOT/"extension/CONFIRMATION_PROTOCOL.json"


def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()


def main():
    if PROTOCOL.exists():raise FileExistsError("Existing confirmation freeze is immutable")
    files=set((ROOT/"extension").glob("*.py"))
    files.update([ROOT/"revision/counts.py",ROOT/"revision/exact_counts.py",ROOT/"revision/joint_profile.py",ROOT/"otg/scores.py"])
    selection={}
    for source in SOURCES:
        if (ROOT/"data/processed/extension_20261021"/f"{source}_final").exists():raise RuntimeError("Final data already prepared")
        joint=ROOT/"runs/extension_20261021"/f"{source}_joint_controls_v2"
        report=json.loads((joint/"report.json").read_text())
        selection[source]={"marginal":report["selected_marginal"],"joint_models":list(report["joint_models"]),
                           "development_best_coherent":min(report["joint_models"],key=lambda k:report["macro"][k]["count_crps"])}
        for seed in [0,1,2]:
            run=ROOT/"runs/extension_20261021"/f"{source}_seed{seed}_replication_v2"
            json.loads((run/"report.json").read_text())
            files.update(p for p in run.iterdir() if p.is_file())
        files.update(p for p in joint.iterdir() if p.is_file())
        files.update((ROOT/"data/processed/extension_20261021"/source).glob("*"))
    for name in ["azure_full","uci_incidents","bpi2014"]:
        f=ROOT/"reports/extension_20261021"/f"source_{name}.json"
        source_record=json.loads(f.read_text());files.add(f)
        raw=ROOT/"data/raw/extension_20261021"/Path(source_record["requested_url"]).name
        if name=="azure_full":raw=ROOT/"data/raw/extension_20261021/azure_v2_vmtable.csv.gz"
        elif name=="uci_incidents":raw=ROOT/"data/raw/extension_20261021/uci_incidents.zip"
        else:raw=ROOT/"data/raw/extension_20261021/BPIC-2014.zip"
        if digest(raw)!=source_record["sha256"]:raise ValueError("Raw source hash mismatch")
        files.add(raw)
    protocol={"created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"sources":SOURCES,"selection":selection,
        "periods":{"uci":"May1--28,2016 at12:00UTC; followup before June1", "azure_full":"days22--28.75 every6h; followup before day30", "bpi2014":"January1--28,2014 at12:00UTC; followup before February1"},
        "primary_comparisons":[["gaussian_typed","gaussian_global"],["gaussian_typed","gaussian_shuffled"],["student_t_typed","student_t_global"],["student_t_typed","direct_dm_exact"],["student_t_typed","direct_bias_dm_exact"]],
        "marginal_comparisons":"three-seed mean CDF for all prespecified architectures, equal-origin integrated Brier; no best-seed selection",
        "scores":["fair count CRPS, equal projection type then group and horizon","coverage and interval width","remaining-cohort reserve ratio9","joint profile energy","fair variogram","peak deviation CRPS"],
        "score_detail":"Pilots and dependence fitting use empirical-ensemble CRPS. Confirmation uses unbiased distinct-pair CRPS; exact DM scores are also reported. Confirmation peak scores use sqrt(1+sum p(1-p)), matching the original manuscript; exploratory pilot peak scaling used sqrt(max(1,sum p(1-p))). No final scoring has occurred under either convention.",
        "draws":512,"sampling_seeds":[0,1,2],"uncertainty":"average seeds per origin before circular paired block bootstrap;1/3/7 days;2000 replicates",
        "adaptive_updates":"none; all predictors and dependence/count parameters fixed",
        "audit_disclosure":"Raw files were acquired; schema, whole-file state totals and timestamp ranges were inspected. These periods have not been prepared as prediction targets, scored, or used for model/threshold selection. This is an internal prediction-holdout freeze, not external preregistration or a claim that raw files were never opened.",
        "promotion_disclosure":"Sources chosen after small development pilots; all negative and deferred pilots remain reported.",
        "files":{str(p.relative_to(ROOT)):digest(p) for p in sorted(files)}}
    PROTOCOL.write_text(json.dumps(protocol,indent=2)+"\n")
    print(json.dumps({"protocol":str(PROTOCOL),"sha256":digest(PROTOCOL),"assets":len(files),"selection":selection}),flush=True)


if __name__=="__main__":main()
