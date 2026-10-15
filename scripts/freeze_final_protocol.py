"""Freeze finished development choices and exact final evaluation artifacts."""
import datetime,hashlib,json,sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]


def main():
    target=ROOT/'docs/FINAL_PROTOCOL.json'
    if target.exists():raise FileExistsError('A final protocol already exists; never silently replace it')
    state=json.loads((ROOT/'STATE.json').read_text())
    if state['final_test_opened']:raise RuntimeError('Cannot claim a pre-final freeze after final access')
    choices={};primary={};files=set()
    for dataset in ['borg','bpi']:
        data=ROOT/'data/processed'/f'{dataset}_joint_selected_v2';m=json.loads((data/'metadata.json').read_text());choices[dataset]=m['selected_marginals']
        json.loads((data/'cohorts/manifest.json').read_text())
        for name in ['metadata.json','scaler.npz','cdf.npy','queries.parquet']:files.add(data/name)
        summary=ROOT/'reports'/f'summary_{dataset}_selected_v2_development_m512';report=json.loads((summary/'manifest.json').read_text())
        if report['status']!='complete':raise RuntimeError('Development evaluation incomplete')
        table=pd.read_csv(summary/'macro_metrics.csv');eligible=table[table['mode']!='static_permuted'];best=eligible.loc[eligible.fraction_crps.idxmin()]
        primary[dataset]={'family':str(best.family),'mode':str(best['mode']),'selection_metric':'Development macro fraction CRPS averaged over three seeds','selected_score':float(best.fraction_crps)}
        files.add(summary/'macro_metrics.csv');files.add(summary/'manifest.json')
        residual=ROOT/'runs'/f'aggregate_residual_{dataset}_selected_v2'
        for name in ['templates.npz','fit_manifest.json','development_report_m512.json']:
            if not (residual/name).is_file():raise RuntimeError('Complete the historical residual comparator before freeze')
            files.add(residual/name)
        for family in ['gaussian','commonshock']:
            for seed in range(3):
                run=ROOT/'runs'/f'joint_{dataset}_selected_v2_{family}_seed{seed}';manifest=json.loads((run/'manifest.json').read_text())
                if manifest['status']!='complete':raise RuntimeError(f'Incomplete coupling: {run}')
                for name in ['manifest.json','global.pt','static.pt','context.pt']:files.add(run/name)
        # Keep every marginal baseline available for final diagnostic tables;
        # the selected common CDF remains fixed for all joint comparisons.
        for pattern in [f'{dataset}_marginals_v1_*',f'{dataset}_marginals_history_v1_*',f'temporal_{dataset}_*']:
            for run in (ROOT/'runs').glob(pattern):
                for folder in run.iterdir():
                    if folder.is_dir() and folder.name.startswith('development'):
                        for file in folder.iterdir():
                            if file.suffix in ['.pt','.joblib'] or file.name in ['manifest.json','scaler.npz','calibration_audit.json']:files.add(file)
    files.update((ROOT/'otg').glob('*.py'));files.update((ROOT/'scripts').glob('*.py'));files.add(ROOT/'docs/SCOPE_REVISION.md')
    frozen=[{'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(files)]
    protocol={'status':'frozen','frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Conditional discrete-horizon first-exit/resolution trajectories for observed current cohorts. No full event-log, new-arrival, or causal deployment claim.',
      'development_informed':True,'selected_marginals':choices,'primary_selected_comparison':primary,
      'final_origins':{'borg':{'elapsed_days':[22,29],'every_hours':6,'first_day':22,'last_day':28.75,'horizons_minutes':[1,5,15,60]},
                       'bpi':{'first':'2016-11-01T12:00:00Z','last':'2016-12-31T12:00:00Z','every_days':1,'horizons_days':[1,3,7,14],'followup_before':'2017-01-16'}},
      'model_update_policy':'Use frozen development-selected individual and coupling checkpoints. No final retraining, recalibration, architecture/threshold selection, or best-seed selection.',
      'additional_baseline':'Historical aggregate residual bootstrap, fitted on development origins. Standardized four-horizon errors, shared resampled origin, nearest log2 group-size bin, clipped/rounded/monotonized counts. Does not preserve individual CDFs or entity-level coherence; compare separately. Development evaluation leaves each origin out.',
      'comparisons':{'families':['gaussian','commonshock'],'modes':['independent','global','static','context','static_permuted'],'seeds':[0,1,2],'main_draws':512,'monte_carlo_sensitivity_draws':1024},
      'projections':{'population':'Current-state groups','relations':'Known job/machine or handler groups with at least five current entities','normalization':'Each group score divided by its current size; equal group, horizon, projection and origin weights'},
      'metrics':['fair count-fraction CRPS','90% coverage/width/interval score','group multihorizon energy','fraction of groups with >=20% exits CRPS','current-cohort reserve loss with under/over ratios3 and9'],
      'inference':{'unit':'Forecast origin after averaging seeds','paired_circular_block_origins':{'borg':[4,8],'bpi':[14,28]},'repeats':10000,'bootstrap_seed':20261014,'report_insufficient_span':True,'interpretation':'Descriptive paired intervals; no IID task-count inference or unqualified cross-site generalization'},
      'sensitivity_plan':['Report per-horizon and per-projection scores, not only macro averages','Check Monte Carlo stability with1024 draws for seed0','Report cutoff-wise marginal bias and calibration','Report known-imputation and unknown-initial-age marginal-score strata using cutoff features; no outcome-defined filtering','Report end-to-end scenario sampling and projection-scoring time and process memory; no isolated kernel-speed claim'],
      'bug_policy':'If a correctness bug is discovered, preserve original artifacts, record exact correction and which conclusions may change, and rerun all affected comparisons. Do not silently tune after final outcomes.',
      'frozen_files':frozen}
    target.write_text(json.dumps(protocol,indent=2)+'\n');print(json.dumps({'path':str(target),'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'frozen_files':len(frozen),'primary':primary}),flush=True)


if __name__=='__main__':main()
