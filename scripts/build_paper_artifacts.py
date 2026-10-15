"""Generate paper tables and vector figures solely from saved final evidence."""
import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from otg.protocol import verify_final_protocol
from otg.uncertainty import paired_moving_block_interval
ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT/'paper';OUT=PAPER/'generated';FIG=PAPER/'figures'
LABELS={'independent':'Independent','global':'Global','static':'Typed static','context':'Typed context','static_permuted':'Permuted incidence','historical_residual':'Historical residual'}
SCORES=['fraction_crps','coverage_90','buffer_cost_ratio9','interval_score_90_fraction','trajectory_energy']


def table(rows,columns,caption,label,wide=True):
    kind='table*' if wide else 'table';spec='l'+'r'*(len(columns)-1)
    lines=[r'\begin{'+kind+r'}[t]',r'\centering\small',r'\caption{'+caption+r'}\label{'+label+'}',r'\begin{tabular}{'+spec+'}',r'\toprule',' & '.join(columns)+r' \\',r'\midrule']
    lines+=[' & '.join(map(str,row))+r' \\' for row in rows]
    return '\n'.join(lines+[r'\bottomrule',r'\end{tabular}',r'\end{'+kind+'}'])+'\n'


def main():
    protocol,digest=verify_final_protocol(ROOT);OUT.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True);inputs=[];all_data={};macros={};primary={};intervals=[]
    for dataset in ['borg','bpi']:
        path=ROOT/'reports'/f'summary_{dataset}_selected_v2_final_m512/all_scores.parquet';d=pd.read_parquet(path);inputs.append(path)
        rpath=ROOT/'runs'/f'aggregate_residual_{dataset}_selected_v2/final_scores_m512.parquet';r=pd.read_parquet(rpath);inputs.append(rpath)
        d=pd.concat([d,r],ignore_index=True);all_data[dataset]=d
        origin=d.groupby(['family','mode','cutoff_us'])[SCORES].mean().reset_index();macro=origin.groupby(['family','mode'])[SCORES].mean();macros[dataset]=macro
        macro.to_csv(OUT/f'{dataset}_all_macro_metrics.csv')
        choice=protocol['primary_selected_comparison'][dataset];primary[dataset]=choice
        pairs=[(choice['family'],choice['mode'],choice['family'],'independent'),('gaussian','static','gaussian','global'),('gaussian','context','gaussian','static'),('gaussian','static','gaussian','static_permuted'),(choice['family'],choice['mode'],'empirical','historical_residual')]
        for family,mode,ref_family,ref_mode in pairs:
            for metric in ['fraction_crps','buffer_cost_ratio9']:
                left=origin[(origin.family==family)&(origin['mode']==mode)].set_index('cutoff_us')[metric]
                right=origin[(origin.family==ref_family)&(origin['mode']==ref_mode)].set_index('cutoff_us')[metric]
                diff=(left-right).sort_index().to_numpy()
                for block in ([4,8] if dataset=='borg' else [14,28]):
                    intervals.append({'dataset':dataset,'family':family,'mode':mode,'reference_family':ref_family,'reference_mode':ref_mode,'metric':metric,'block':block,**paired_moving_block_interval(diff,block,10000,20261014)})
    pd.DataFrame(intervals).to_csv(OUT/'primary_paired_intervals.csv',index=False)
    models=[(f,m) for f in ['gaussian','commonshock'] for m in ['independent','global','static','context','static_permuted']]+[('empirical','historical_residual')]
    rows=[]
    for f,m in models:
        label=({'gaussian':'G: ','commonshock':'S: ','empirical':''}[f])+LABELS[m]
        values=[label]
        for dataset in ['borg','bpi']:
            v=macros[dataset].loc[(f,m)];values += [f'{v.fraction_crps:.5f}',f'{v.coverage_90:.3f}',f'{v.buffer_cost_ratio9:.4f}']
        rows.append(values)
    (OUT/'main_table.tex').write_text(table(rows,['Model','Borg CRPS','Cov.','Loss$_9$','BPI CRPS','Cov.','Loss$_9$'],
      'Frozen final evaluation, averaged over three seeds and forecast origins. CRPS and reserve loss are normalized by group size (lower is better); coverage targets 0.9. G: Gaussian; S: common shock. Independent draws are shown for each family to expose Monte Carlo variation. The historical-residual comparator changes aggregate means and lacks entity-level coherence.','tab:main'))
    # All marginal selection candidates are development-only and are labeled so.
    candidates={}
    for dataset in ['borg','bpi']:
        path=ROOT/'reports'/f'{dataset}_selected_marginals_v2.json';inputs.append(path);candidates[dataset]=json.loads(path.read_text())['candidate_marginals']
    arm_names=[('hgb','v1','personal'),('hgb','v1','global'),('hgb','v1','relational'),('hgb','history_v1','personal'),('hgb','history_v1','global'),('hgb','history_v1','relational'),('temporal_ensemble',None,'mixer'),('temporal_ensemble',None,'gru'),('temporal_ensemble',None,'mlp')]
    rows=[]
    for kind,tag,arm in arm_names:
        label=('HGB '+('history ' if tag=='history_v1' else '')+arm) if kind=='hgb' else arm.upper()+' ensemble'
        row=[label]
        for dataset,state in [('borg',1),('borg',2),('bpi',1)]:
            eligible=[c for c in candidates[dataset] if c['state']==state and c['kind']==kind and (c.get('tag')==tag and c.get('arm')==arm if kind=='hgb' else c['architecture']==arm)]
            row.append(f"{min(c['development_integrated_brier'] for c in eligible):.5f}" if eligible else '--')
        rows.append(row)
    (OUT/'marginal_table.tex').write_text(table(rows,['Individual predictor','Borg pending','Borg running','BPI'],
      'Development integrated Brier used for model selection. Each row selects its calibration variant on development; neural rows average three seeds before scoring. These are selection scores, not final estimates. HGB history is the additional recent-event-count feature arm.','tab:marginal'))
    supplemental=[]
    for dataset in ['borg','bpi']:
        d=all_data[dataset];metrics=d.groupby(['family','mode','cutoff_us','projection'])[SCORES].mean().groupby(['family','mode','projection']).mean();metrics.to_csv(OUT/f'{dataset}_projection_metrics.csv');rows=[]
        for projection in metrics.index.get_level_values('projection').unique():
            for f,m in [('gaussian','independent'),('gaussian','global'),('gaussian','static'),('gaussian','context'),('gaussian','static_permuted'),('commonshock','static'),('empirical','historical_residual')]:
                v=metrics.loc[(f,m,projection)];name=projection.replace('population_by_state','population').replace('resource','handler')+': '+('S ' if f=='commonshock' else '')+LABELS[m]
                rows.append([name,f'{v.fraction_crps:.5f}',f'{v.coverage_90:.3f}',f'{v.interval_score_90_fraction:.4f}',f'{v.buffer_cost_ratio9:.4f}'])
        supplemental.append(table(rows,['Projection/model','CRPS','Cov.','Interval score','Loss$_9$'],f'{dataset.upper()} final projection diagnostics. Equal horizon and origin weights; the full CSV also includes every family/mode and seed.','tab:'+dataset+'projection'))
    (OUT/'supplementary_tables.tex').write_text('\n'.join(supplemental))
    # Main figure: same model order across applications and distinct score/loss axes.
    shown=[('gaussian','independent'),('gaussian','global'),('gaussian','static'),('gaussian','context'),('commonshock','static'),('empirical','historical_residual')]
    colors=['#757575','#4C78A8','#007C78','#72B7B2','#C47A27','#8F63A3'];names=['Independent','Gaussian global','Gaussian typed','Gaussian context','Shock typed','Historical errors']
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(9.2,5.3),layout='constrained')
    for j,dataset in enumerate(['borg','bpi']):
        for i,(metric,title) in enumerate([('fraction_crps','Count-fraction CRPS'),('buffer_cost_ratio9','Reserve loss (ratio 9)')]):
            values=[macros[dataset].loc[pair,metric] for pair in shown];ax=axes[i,j];ax.barh(np.arange(len(shown)),values,color=colors);ax.set_yticks(np.arange(len(shown)),names if j==0 else []);ax.invert_yaxis();ax.set_title(dataset.upper()+' — '+title);ax.set_xlabel('Lower is better');ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    fig.savefig(FIG/'joint_scores.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(9.5,2.9),layout='constrained')
    diagnostic_summary={}
    for ax,(dataset,state,title) in zip(axes,[('borg',1,'Borg pending: 60 min'),('borg',2,'Borg running: 60 min'),('bpi',1,'BPI: 14 days')]):
        path=ROOT/'reports'/f'final_diagnostics_{dataset}/marginal_scores.csv';inputs.append(path);d=pd.read_csv(path);d=d[(d.stratum=='all')&(d.state==state)&(d.horizon==d.horizon.max())].sort_values('cutoff_us')
        x=np.arange(len(d));ax.plot(x,d.observed_fraction,label='Observed',color='#222222');ax.plot(x,d.predicted_fraction,label='Predicted',color='#007C78');ax.set_title(title);ax.set_xlabel('Chronological final origin');ax.set_ylabel('Current-cohort exit fraction');ax.grid(alpha=.2);diagnostic_summary[dataset+str(state)]=d[['brier','predicted_fraction','observed_fraction']].mean().to_dict()
    axes[0].legend(frameon=False);fig.savefig(FIG/'marginal_bias.pdf');plt.close(fig)
    mc=[];runtimes=[]
    for dataset in ['borg','bpi']:
        for family in ['gaussian','commonshock']:
            run=ROOT/'runs'/f'joint_{dataset}_selected_v2_{family}_seed0'
            p=run/'final_cohort_scores_m1024.parquet';inputs.append(p);larger=pd.read_parquet(p).groupby('mode')[SCORES].mean()
            smaller=pd.read_parquet(run/'final_cohort_scores_m512.parquet').groupby('mode')[SCORES].mean()
            for mode in smaller.index:mc.append({'dataset':dataset,'family':family,'mode':mode,'crps_512':smaller.loc[mode,'fraction_crps'],'crps_1024':larger.loc[mode,'fraction_crps'],'difference':larger.loc[mode,'fraction_crps']-smaller.loc[mode,'fraction_crps']})
            for seed in range(3):
                p=ROOT/'runs'/f'joint_{dataset}_selected_v2_{family}_seed{seed}'/'final_cohort_report_m512.json';inputs.append(p);r=json.loads(p.read_text());runtimes.append({'dataset':dataset,'family':family,'seed':seed,'elapsed_s':r['elapsed_s'],'n_origins':r['n_cohorts'],'peak_host_rss_kib':r['peak_host_rss_kib'],'peak_torch_allocated_bytes':r['peak_torch_allocated_bytes'],'gpu_name':r['gpu_name'],'sampling_scoring_s':sum(t['sampling_and_scoring_s'] for t in r['timings'])})
    pd.DataFrame(mc).to_csv(OUT/'monte_carlo_sensitivity.csv',index=False);pd.DataFrame(runtimes).to_csv(OUT/'runtimes.csv',index=False)
    report={'status':'complete','protocol_sha256':digest,'primary':primary,'macro':{d:{f+'_'+m:macros[d].loc[(f,m)].to_dict() for f,m in models} for d in macros},'final_longest_horizon_marginals':diagnostic_summary,'paired_intervals':intervals,'monte_carlo_max_abs_crps_change':max(abs(r['difference']) for r in mc),
       'inputs':[{'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(set(inputs))],'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'evidence_manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ['status','primary','final_longest_horizon_marginals','monte_carlo_max_abs_crps_change']}))


if __name__=='__main__':main()
