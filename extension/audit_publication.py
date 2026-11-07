"""Read-only scientific checks; write a new post-confirmation audit report."""
import datetime, hashlib, importlib.metadata, json, subprocess, sys, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from extension.freeze_confirmation import digest, PROTOCOL

def main():
    report = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "freezes": {}, "sources": {}, "sensitivity": {}}
    for path, key in [("docs/FINAL_PROTOCOL.json", "frozen_files"), ("revision/FRESH_FINAL_PROTOCOL.json", "files")]:
        entries = json.loads((ROOT/path).read_text())[key]
        for e in entries: assert digest(ROOT/e["path"]) == e["sha256"], e["path"]
        report["freezes"][path] = len(entries)
    protocol = json.loads(PROTOCOL.read_text())
    for name, expected in protocol["files"].items(): assert digest(ROOT/name) == expected, name
    report["freezes"][str(PROTOCOL.relative_to(ROOT))] = len(protocol["files"])
    report["prefix_extension"] = json.loads((ROOT/"reports/extension_20261021/prefix_extension_audit.json").read_text())
    assert report["prefix_extension"]["status"] == "passed"
    end = {"azure_full": 30*86400, "uci": pd.Timestamp("2016-06-01", tz="UTC").timestamp(), "bpi2014": pd.Timestamp("2014-02-01", tz="UTC").timestamp()}
    for source in protocol["sources"]:
        run = ROOT/"runs/extension_20261021"/f"{source}_confirmation"
        result = json.loads((run/"report.json").read_text())
        assert result["status"] == "complete" and result["origins"] == 28
        assert result["protocol_sha256"] == digest(PROTOCOL)
        opened = json.loads((run/"opened.json").read_text())
        assert pd.Timestamp(opened["utc"]) > pd.Timestamp(protocol["created_utc"])
        assert result["selected_marginal"] == protocol["selection"][source]["marginal"]
        folder = ROOT/"data/processed/extension_20261021"/f"{source}_final"
        m = json.loads((folder/"manifest.json").read_text()); z = np.load(folder/"cohorts.npz")
        assert len(z["y"]) == result["queries"] and np.all(np.diff(z["y"], axis=-1)>=0)
        for c in m["cohorts"]:
            assert c["origin"] + max(m["horizons_seconds"]) < end[source]
            assert c["part"] == "final" and c["origin"] > m["bounds"][-1]
        predictions = np.load(run/"predictions.npz")
        for name in predictions.files:
            p = predictions[name]
            assert p.shape == z["y"].shape and np.isfinite(p).all()
            assert p.min() >= 0 and p.max() <= 1+1e-7 and np.all(np.diff(p, axis=-1)>=-1e-8)
        projection = pd.read_csv(run/"projection_scores.csv")
        origin = pd.read_csv(run/"origin_scores.csv")
        numeric = ["count_crps", "coverage90", "width90", "reserve_loss", "joint_profile_energy", "joint_variogram_fair", "peak_slowdown_crps"]
        recomputed = projection.groupby(["model", "origin"])[numeric].mean().sort_index()
        np.testing.assert_allclose(recomputed, origin.set_index(["model", "origin"]).sort_index()[numeric], atol=1e-12, equal_nan=True)
        macro = recomputed.groupby("model").mean()
        for model, d in result["macro"].items():
            for metric in numeric: np.testing.assert_allclose(macro.loc[model, metric], d[metric], atol=1e-12, equal_nan=True)
        error = max(abs(macro.loc[a, "count_crps"]-macro.loc[a+"_exact", "count_crps"]) for a in ["direct_dm", "direct_bias_dm"])
        assert error < .0002
        report["sources"][source] = {"origins": len(m["cohorts"]), "queries": len(z["y"]), "predictors": len(predictions.files), "direct_mc_exact_max_crps_gap": float(error), "maturity_and_cdf_checks": "passed", "freeze_precedes_scoring": True}
        sensitivity = pd.read_csv(ROOT/"reports/extension_20261021"/f"sensitivity_{source}.csv").groupby("model").count_crps.mean()
        seed0 = projection[projection.seed.eq(0)].groupby(["model", "origin"]).count_crps.mean().groupby("model").mean()
        report["sensitivity"][source] = {"scope": "post-confirmation 1024-draw seed0 check; no refitting or reselection", "max_change_vs_seed0_512": float((sensitivity-seed0[sensitivity.index]).abs().max()), "scores1024": sensitivity.to_dict(), "typed_vs_global_order_preserved": bool((sensitivity["gaussian_typed"]<=sensitivity["gaussian_global"]) == (seed0["gaussian_typed"]<=seed0["gaussian_global"]))}
    artifact_paths = sorted((ROOT/"paper_revision/generated").glob("extension_*")) + [ROOT/"paper_revision/figures/extension_effects.pdf"]
    before = {str(p.relative_to(ROOT)): digest(p) for p in artifact_paths}
    rebuilt = subprocess.run([sys.executable, "extension/build_publication.py"], cwd=ROOT, capture_output=True, text=True)
    assert rebuilt.returncode == 0, rebuilt.stderr
    after = {str(p.relative_to(ROOT)): digest(p) for p in artifact_paths}
    assert before == after, "Generated numerical/figure artifacts changed on regeneration"
    report["identical_regeneration"] = before
    cmd = [sys.executable, "-m", "pytest", "tests", "revision/test_revision.py", "revision/test_exact_counts.py", "revision/test_alibaba.py", "extension/test_extension.py", "-q"]
    tests = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    report["tests"] = {"exit_code": tests.returncode, "output": tests.stdout+tests.stderr}
    assert tests.returncode == 0, report["tests"]
    roots = [ROOT/f"data/{kind}/extension_20261021" for kind in ["raw", "processed"]] + [ROOT/"runs/extension_20261021", ROOT/"reports/extension_20261021"]
    report["new_storage_bytes"] = sum(p.stat().st_size for r in roots for p in r.rglob("*") if p.is_file())
    assert report["new_storage_bytes"] < 2*1024**3
    report["versions"] = {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn", "torch", "matplotlib", "pyarrow", "pytest"]}
    log = (ROOT/"paper_revision/main.log").read_text()
    assert "Overfull" not in log and "undefined" not in log.lower()
    assert "Warning--" not in (ROOT/"paper_revision/main.blg").read_text()
    bbox = subprocess.run(["pdftotext", "-bbox", str(ROOT/"paper_revision/main.pdf"), "-"], capture_output=True, text=True, check=True)
    pages = ET.fromstring(bbox.stdout).findall(".//{*}page")
    for page in pages:
        for word in page.findall(".//{*}word"):
            assert float(word.attrib["xMin"])>=30 and float(word.attrib["xMax"])<=float(page.attrib["width"])-30
            assert float(word.attrib["yMax"])<=float(page.attrib["height"])-20
    report["manuscript"] = {"path":"paper_revision/main.pdf", "sha256":digest(ROOT/"paper_revision/main.pdf"), "pages":len(pages), "no_overfull_or_unresolved_references":True, "word_bounds_within_page":True, "venue":"neutral, user confirmed"}
    report["status"] = "passed"
    target = ROOT/"reports/extension_20261021/publication_audit.json"
    target.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))

if __name__ == "__main__": main()
