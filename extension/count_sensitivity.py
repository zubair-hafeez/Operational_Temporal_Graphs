"""Post-freeze numerical sensitivity only: no training, tuning, or reselection."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from extension.freeze_confirmation import PROTOCOL, digest
from extension.run_pilots import scenario, projections
from otg.scores import ensemble_crps

def main(source):
    protocol = json.loads(PROTOCOL.read_text())
    for name, expected in protocol["files"].items():
        assert digest(ROOT / name) == expected, name
    run = ROOT / "runs/extension_20261021" / f"{source}_confirmation"
    fit = json.loads((ROOT / "runs/extension_20261021" / f"{source}_joint_controls_v2/report.json").read_text())
    folder = ROOT / "data/processed/extension_20261021" / f"{source}_final"
    z = np.load(folder / "cohorts.npz")
    manifest = json.loads((folder / "manifest.json").read_text())
    p = np.load(run / "predictions.npz")[fit["selected_marginal"]]
    names = list(dict.fromkeys(["independent", "gaussian_global", "gaussian_typed", "student_t_global", "student_t_typed", protocol["selection"][source]["development_best_coherent"]]))
    rows = []
    for c in manifest["cohorts"]:
        ix = slice(c["offset"], c["offset"] + c["size"])
        groups = projections(z["relations"][ix])
        for name in names:
            w, df, shuffled = fit["joint_models"][name]
            samples = scenario(p[ix], z["relations"][ix], w, 1024, 1400, df, shuffled)
            scores = {}
            for kind, members in groups:
                pred = samples[:, members].sum(1) / len(members)
                truth = z["y"][ix][members].mean(0)
                scores.setdefault(kind, []).append(float(ensemble_crps(pred, truth).mean()))
            rows.append({"source": source, "origin": c["origin"], "model": name, "draws": 1024, "seed": 0,
                         "count_crps": float(np.mean([np.mean(v) for v in scores.values()]))})
        print(json.dumps({"source": source, "sensitivity_origin": c["origin"]}), flush=True)
    target = ROOT / "reports/extension_20261021" / f"sensitivity_{source}.csv"
    pd.DataFrame(rows).to_csv(target, index=False)
    print(pd.DataFrame(rows).groupby("model").count_crps.mean().to_string(), flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--source", required=True)
    main(ap.parse_args().source)
