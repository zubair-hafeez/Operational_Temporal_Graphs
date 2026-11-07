"""Generate current-manuscript extension tables/figures from immutable run outputs."""
import hashlib, json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
os.environ.setdefault("MPLCONFIGDIR", "/tmp/otg-extension-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from extension.joint_controls import bootstrap_difference
RUNS = ROOT / "runs/extension_20261021"
OUT = ROOT / "paper_revision/generated"
SOURCES = {"uci": "UCI incidents", "azure_full": "Azure V2", "bpi2014": "BPI 2014"}
MODELS = {"independent": "Independent", "gaussian_global": "Gaussian global", "gaussian_typed": "Gaussian typed", "gaussian_shuffled": "Shuffled typed", "student_t_global": "Student-$t$ global", "student_t_typed": "Student-$t$ typed", "direct_dm_exact": "Direct DM (exact)", "direct_bias_dm_exact": "Bias-DM (exact)", "direct_residual": "Residual"}
MARGINALS = {"hgb_personal": "HGB personal", "hgb_relational": "HGB relations", "hgb_temporal": "HGB temporal", "mlp": "Node MLP", "set": "Global set", "graph": "Typed graph", "graph_shuffled": "Shuffled graph", "graph_relation0": "Relation 1 only", "graph_relation1": "Relation 2 only", "temporal_attention": "Temporal attention"}

def table(path, caption, label, columns, header, rows, wide=False):
    env = "table*" if wide else "table"
    body = [f"\\begin{{{env}}}[t]", "\\centering\\small", f"\\caption{{{caption}}}\\label{{{label}}}", f"\\begin{{tabular}}{{{columns}}}", "\\toprule", " & ".join(header) + r" \\", "\\midrule"]
    body += [" & ".join(row) + r" \\" for row in rows]
    body += ["\\bottomrule", "\\end{tabular}", f"\\end{{{env}}}"]
    path.write_text("\n".join(body) + "\n")

def main():
    OUT.mkdir(exist_ok=True)
    reports = {s: json.loads((RUNS / f"{s}_confirmation/report.json").read_text()) for s in SOURCES}
    protocol = json.loads((ROOT / "extension/CONFIRMATION_PROTOCOL.json").read_text())
    frames, marginal_frames, ci = [], [], []
    for s, report in reports.items():
        frame = pd.DataFrame.from_dict(report["macro"], orient="index").rename_axis("model").reset_index()
        frame["source"] = s; frames.append(frame)
        marginal = pd.read_csv(RUNS / f"{s}_confirmation/marginal_scores.csv").groupby("model").brier.mean().reset_index()
        marginal["source"] = s; marginal_frames.append(marginal)
        ci.extend({"source": s, **r} for r in report["paired_intervals"])
    scores = pd.concat(frames, ignore_index=True)
    scores.to_csv(OUT / "extension_macro.csv", index=False)
    marg = pd.concat(marginal_frames, ignore_index=True)
    marg.to_csv(OUT / "extension_marginals.csv", index=False)
    pd.DataFrame(ci).to_csv(OUT / "extension_intervals.csv", index=False)
    secondary = []
    for s in SOURCES:
        origin = pd.read_csv(RUNS / f"{s}_confirmation/origin_scores.csv")
        selected = protocol["selection"][s]["development_best_coherent"]
        for reference in ["gaussian_global", "direct_dm_exact", "direct_bias_dm_exact"]:
            for block in [1, 3, 7]:
                secondary.append({"source": s, "candidate": selected, "reference": reference,
                                  "status": "post-confirmation descriptive interval; model fixed before confirmation",
                                  **bootstrap_difference(origin, reference, selected, block)})
    (OUT / "extension_secondary_intervals.json").write_text(json.dumps(secondary, indent=2)+"\n")
    rows = []
    for model, label in MODELS.items():
        row = [label]
        for s in SOURCES:
            d = reports[s]["macro"][model]
            row += [f"{1000*d['count_crps']:.2f}", f"{1000*d['reserve_loss']:.2f}"]
        rows.append(row)
    row = ["Dev.-selected coherent"]
    for s in SOURCES:
        d = reports[s]["macro"][protocol["selection"][s]["development_best_coherent"]]
        row += [f"{1000*d['count_crps']:.2f}", f"{1000*d['reserve_loss']:.2f}"]
    rows.append(row)
    table(OUT / "extension_counts.tex", "Frozen new-period results: count-fraction CRPS and ratio-9 reserve loss, both multiplied by 1,000 (lower is better). Each source has 28 origins. All joint arms share its development-selected three-seed marginal predictor. Direct rows use exact count distributions. Last row: typed Student-$t$ (UCI), subscription-only Gaussian (Azure), team-only Gaussian (BPI14), selected on development, never on confirmation.", "tab:extension-counts", "lrrrrrr", ["Model", "UCI CRPS", "Loss", "Azure CRPS", "Loss", "BPI14 CRPS", "Loss"], rows, True)
    rows = []
    for model, label in MARGINALS.items():
        row = [label]
        for s in SOURCES:
            v = marg[(marg.source == s) & (marg.model == model)]
            row.append("---" if v.empty else f"{1000*v.brier.iloc[0]:.2f}")
        rows.append(row)
    table(OUT / "extension_marginals.tex", "New-period equal-origin integrated Brier $\\times 1,000$ (lower is better), using three-seed mean CDFs. Selected beforehand: typed graph (UCI), temporal HGB (Azure), relational HGB (BPI14). Single-relation order: handler/team for UCI; deployment/subscription for Azure. BPI14 has one relation.", "tab:extension-marginals", "lrrr", ["Marginal model", "UCI", "Azure", "BPI14"], rows)
    rows = []
    for s in SOURCES:
        for model in ["independent", "gaussian_global", "gaussian_typed", "student_t_global", "student_t_typed", "direct_dm", "direct_bias_dm"]:
            d = reports[s]["macro"][model]
            label = MODELS.get(model, {"direct_dm":"Direct DM", "direct_bias_dm":"Bias-DM"}.get(model))
            rows.append([SOURCES[s], label, f"{d['coverage90']:.3f}", f"{d['width90']:.3f}", f"{d['joint_profile_energy']:.5f}", f"{d['joint_variogram_fair']:.5f}", f"{d['peak_slowdown_crps']:.3f}"])
    table(OUT / "extension_joint.tex", "All-source confirmation distribution diagnostics. Coverage targets .90; width is in cohort fractions. Energy, fair variogram and peak-deviation CRPS are lower-is-better. Direct DM joint profiles are separately generated group trajectories, not coherent entity-level realizations.", "tab:extension-joint", "llrrrrr", ["Source", "Model", "Cov.", "Width", "Energy", "Variogram", "Peak"], rows, True)
    rows = []
    for s in SOURCES:
        fit = json.loads((RUNS / f"{s}_joint_controls_v2/report.json").read_text())
        for model, (weights, df, shuffled) in fit["joint_models"].items():
            if model in ["student_t", "independent"] or shuffled: continue
            d = reports[s]["macro"][model]
            rows.append([SOURCES[s], model.replace("gaussian_", "G: ").replace("student_t_", "$t$: ").replace("_", " "), "/".join(f"{w:g}" for w in weights), "---" if df is None else str(df), f"{1000*d['count_crps']:.2f}", f"{1000*d['reserve_loss']:.2f}"])
    table(OUT / "extension_relations.tex", "Frozen relation controls. Variance weights are global/handler/team (UCI), global/deployment/subscription (Azure), and global/team (BPI14). Only degrees of freedom change between matched Gaussian and Student-$t$ arms; Student-$t$ uses a shared scale. Scores are $\\times 1,000$.", "tab:extension-relations", "lllrrr", ["Source", "Coupling", "Weights", "$\\nu$", "CRPS", "Loss"], rows, True)
    # Primary comparison CIs, expressed on a common relative-to-global scale.
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    for ax, family in zip(axes, ["gaussian", "student_t"]):
        for j, s in enumerate(SOURCES):
            base = reports[s]["macro"][f"{family}_global"]["count_crps"]
            for block, offset, color in [(1, -.14, "#80b1d3"), (3, 0, "#1f77b4"), (7, .14, "#08306b")]:
                item = next(v for v in ci if v["source"] == s and v["candidate"] == f"{family}_typed" and v["reference"] == f"{family}_global" and v["block_days"] == block)
                point = -100*item["mean_difference"]/base
                lo, hi = sorted(-100*np.array(item["interval95"])/base)
                ax.errorbar(point, j+offset, xerr=[[max(0,point-lo)], [max(0,hi-point)]], fmt="o", markersize=3, color=color, label=f"{block}-day blocks" if j == 0 else None, capsize=2)
        ax.axvline(0, color="gray", linewidth=.8); ax.grid(axis="x", alpha=.2)
        ax.set_title("Gaussian" if family == "gaussian" else "Student-t")
        ax.set_xlabel("Typed improvement over global (%)")
        ax.set_yticks(range(3), SOURCES.values())
    axes[0].invert_yaxis()
    axes[1].legend(fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(ROOT / "paper_revision/figures/extension_effects.pdf", metadata={"CreationDate": None, "ModDate": None}); plt.close(fig)
    inputs = [ROOT / "extension/CONFIRMATION_PROTOCOL.json"]
    inputs += [RUNS / f"{s}_confirmation" / f for s in SOURCES for f in ["report.json", "macro_scores.csv", "marginal_scores.csv", "origin_scores.csv"]]
    artifact = {"generated_by": str(Path(__file__).relative_to(ROOT)), "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "total_new_origins": sum(r["origins"] for r in reports.values()), "total_new_queries": sum(r["queries"] for r in reports.values()), "selected": protocol["selection"], "inputs": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}}
    (OUT / "extension_manifest.json").write_text(json.dumps(artifact, indent=2)+"\n")
    print(json.dumps({k:v for k,v in artifact.items() if k != "inputs"}, indent=2))

if __name__ == "__main__": main()
