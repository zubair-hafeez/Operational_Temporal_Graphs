"""Generate manuscript evidence from immutable original/fresh and revision runs."""
import json,sys,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from revision.common import ROOT,write_json,digest
from revision.fresh_protocol import verify_fresh_protocol
from otg.protocol import verify_final_protocol
from otg.uncertainty import paired_moving_block_interval

OUT=ROOT/'paper_revision/generated'
METRICS=['fraction_crps','coverage_90','buffer_cost_ratio9']
SOURCES=[]


def read(path):
    p=ROOT/path;SOURCES.append({'path':str(p.relative_to(ROOT)),'sha256':digest(p)})
    return pd.read_parquet(p) if p.suffix=='.parquet' else pd.read_csv(p)


def read_json(path):
    p=ROOT/path;SOURCES.append({'path':str(p.relative_to(ROOT)),'sha256':digest(p)})
    return json.loads(p.read_text())


def main():
    verify_final_protocol(ROOT);verify_fresh_protocol();OUT.mkdir(exist_ok=True)
    allframes=[];jointframes=[]
    for ds in ['borg','bpi']:
        old=read(f'reports/summary_{ds}_selected_v2_final_m512/all_scores.parquet');old=old[(old.family=='gaussian')&old['mode'].isin(['independent','global','static'])].copy();old['method']=old['mode'];old['dataset']=ds;allframes.append(old)
        residual=read(f'runs/aggregate_residual_{ds}_selected_v2/final_scores_m512.parquet');residual['method']='historical_residual';residual['dataset']=ds;allframes.append(residual)
        base=ROOT/'runs'/f'revision_{ds}_direct_counts_v2';chosen=read_json(f'runs/revision_{ds}_direct_counts_v2/manifest.json')['selected_on_development'];dm=read(str((base/'scores.parquet').relative_to(ROOT)));dm=dm[(dm.period=='retrospective_final')&(dm['mode']==chosen)].copy();dm['method']='early_dm';dm['dataset']=ds;allframes.append(dm)
        dm=read(f'runs/revision_{ds}_development_fitted_dm_v1/scores.parquet');dm=dm[dm.period=='retrospective_final'].copy();dm['method']='recent_dm';dm['dataset']=ds;allframes.append(dm)
        if ds=='bpi':
            re=read('runs/revision_bpi_updated_dm_profiles_v1/count_scores.parquet');re=re[re['mode'].isin(['development_bottom_up_dm','development_conditioned_dm'])].copy();re['method']=re['mode'].map({'development_bottom_up_dm':'bottom_up','development_conditioned_dm':'conditioned'});re['dataset']=ds;allframes.append(re)
        j=read(f'runs/revision_{ds}_joint_profiles_v1/scores.parquet');j=j[j.period=='retrospective_final'].copy();j['dataset']=ds;j['method']=j['mode'].replace({'direct_selected_dm':'early_dm'});jointframes.append(j)
        j=read(f'runs/revision_{ds}_updated_dm_profiles_v1/scores.parquet');j['dataset']=ds;j['method']=j['mode'].replace({'development_fitted_dm':'recent_dm','development_bottom_up_dm':'bottom_up','development_conditioned_dm':'conditioned'});jointframes.append(j)
    f=read('runs/revision_bpi2012_final_evaluation_v1/scores.parquet');f=f[f.calibration=='identity'].copy();f['method']=f['mode'].replace({'bias_dm':'early_dm','bias_dm_bottom_up':'bottom_up','bias_dm_conditioned':'conditioned'});f['dataset']='bpi2012';allframes.append(f)
    j=read('runs/revision_bpi2012_final_evaluation_v1/joint_profiles.parquet');j['method']=np.where(j.calibration=='identity',j['mode'],j.calibration.replace({'rolling60':'rolling'})+'_calibration_'+j['mode']);j['method']=j.method.replace({'bias_dm':'early_dm','bias_dm_bottom_up':'bottom_up','bias_dm_conditioned':'conditioned'});j['dataset']='bpi2012';jointframes.append(j)
    frames=pd.concat(allframes,ignore_index=True);origin=frames.groupby(['dataset','method','cutoff_us'])[METRICS].mean().reset_index();macro=origin.groupby(['dataset','method'])[METRICS].mean().reset_index();macro.to_csv(OUT/'count_macro.csv',index=False);origin.to_csv(OUT/'count_origin_scores.csv',index=False)
    jm=['joint_profile_energy','joint_variogram_fair','peak_slowdown_crps'];jf=pd.concat(jointframes,ignore_index=True);jo=jf.groupby(['dataset','method','cutoff_us'])[jm].mean().reset_index();jmacro=jo.groupby(['dataset','method'])[jm].mean().reset_index();jmacro.to_csv(OUT/'joint_macro.csv',index=False);jo.to_csv(OUT/'joint_origin_scores.csv',index=False)
    projection=jf.groupby(['dataset','method','projection'])[jm].mean().reset_index();projection.to_csv(OUT/'joint_projection_scores.csv',index=False)
    labels={'independent':'Independent','global':'Global factor','static':'Typed factors','early_dm':'DM, earlier fit','recent_dm':'DM, development fit','bottom_up':'Bottom-up DM','conditioned':'Conditioned DM','historical_residual':'Historical residuals'}
    lines=[r'\begin{table*}[t]\centering\small',r'\caption{Count forecasts on original final periods and fresh BPI 2012 confirmation. Lower CRPS and loss are better; nominal coverage is .90. The three factor rows retain the original individual CDFs. Revised controls on Borg/BPI 2017 are retrospective. DM development-fit scores are exact; sampled methods use 512 draws and three seeds. A dash denotes an unimplemented comparison.}\label{tab:counts}',r'\begin{tabular}{lrrr rrr rrr}\toprule',r'&\multicolumn{3}{c}{Borg}&\multicolumn{3}{c}{BPI 2017}&\multicolumn{3}{c}{BPI 2012 (fresh)}\\',r'Method&CRPS&Cov.&Loss$_9$&CRPS&Cov.&Loss$_9$&CRPS&Cov.&Loss$_9$\\\midrule']
    for method,label in labels.items():
        values=[]
        for ds in ['borg','bpi','bpi2012']:
            z=macro[(macro.dataset==ds)&(macro.method==method)]
            values.extend(['---']*3 if not len(z) else [f'{z.iloc[0][m]:.4f}' for m in METRICS])
        lines.append(label+'&'+'&'.join(values)+r'\\')
    lines.extend([r'\bottomrule\end{tabular}\end{table*}']);(OUT/'count_table.tex').write_text('\n'.join(lines)+'\n')
    lines=[r'\begin{table*}[t]\centering\small',r'\caption{Joint group-profile scores. ES is RMS-normalized energy score; Peak is CRPS of the maximum standardized slow-progress deviation. Scales differ across datasets. Original-period rows are retrospective; all fresh endpoints were frozen before opening February.}\label{tab:joint}',r'\begin{tabular}{lrr rr rr}\toprule',r'&\multicolumn{2}{c}{Borg}&\multicolumn{2}{c}{BPI 2017}&\multicolumn{2}{c}{BPI 2012 (fresh)}\\',r'Method&ES&Peak&ES&Peak&ES&Peak\\\midrule']
    for method in ['independent','global','static','early_dm','recent_dm','bottom_up','conditioned','rolling_calibration_global']:
        vals=[]
        for ds in ['borg','bpi','bpi2012']:
            z=jmacro[(jmacro.dataset==ds)&(jmacro.method==method)];vals.extend(['---']*2 if not len(z) else [f'{z.iloc[0][k]:.4f}' for k in ['joint_profile_energy','peak_slowdown_crps']])
        lines.append(labels.get(method,'Rolling correction + global')+'&'+'&'.join(vals)+r'\\')
    lines.extend([r'\bottomrule\end{tabular}\end{table*}']);(OUT/'joint_table.tex').write_text('\n'.join(lines)+'\n')
    cal=read('runs/revision_bpi_calibration_same_version_v2/scores.parquet');cs=cal.groupby(['period','calibration','mode'])[METRICS].mean().reset_index();cs.to_csv(OUT/'calibration_macro.csv',index=False)
    lines=[r'\begin{table*}[t]\centering\small',r'\caption{BPI 2017 controlled same-version correction. Original Gaussian weights are fixed. These are retrospective diagnostics; improved mean calibration does not guarantee a better reserve quantile.}\label{tab:calibration}',r'\begin{tabular}{llrr rr}\toprule',r'& &\multicolumn{2}{c}{October development}&\multicolumn{2}{c}{November--December}\\',r'Correction&Dependence&CRPS&Loss$_9$&CRPS&Loss$_9$\\\midrule']
    for correction in ['identity','static','rolling60']:
        for mode in ['independent','global','static']:
            vals=[]
            for period in ['development','retrospective_final']:
                v=cs[(cs.period==period)&(cs.calibration==correction)&(cs['mode']==mode)].iloc[0];vals.extend([f'{v.fraction_crps:.4f}',f'{v.buffer_cost_ratio9:.4f}'])
            lines.append(correction.replace('rolling60','Rolling 60d').title()+'&'+labels[mode]+'&'+'&'.join(vals)+r'\\')
    lines.extend([r'\bottomrule\end{tabular}\end{table*}']);(OUT/'calibration_table.tex').write_text('\n'.join(lines)+'\n')
    cf=cal[cal.period=='retrospective_final'].copy();cf['dataset']='bpi';cf['method']=cf.calibration+'_correction_'+cf['mode'];co=cf.groupby(['dataset','method','cutoff_us'])[METRICS].mean().reset_index();combined=pd.concat([origin,co],ignore_index=True)
    intervals=[]
    comparisons={'borg':[('static','global'),('static','recent_dm'),('static','independent')],'bpi':[('global','independent'),('recent_dm','global'),('rolling60_correction_global','identity_correction_global'),('rolling60_correction_global','recent_dm')],'bpi2012':[('global','independent'),('early_dm','global'),('static','global')]}
    for ds,pairs in comparisons.items():
        for model,reference in pairs:
            for metric in ['fraction_crps','buffer_cost_ratio9']:
                pivot=combined[combined.dataset==ds].pivot(index='cutoff_us',columns='method',values=metric).sort_index();delta=(pivot[model]-pivot[reference]).to_numpy()
                for block in {'borg':[4,8],'bpi':[14,28],'bpi2012':[7,14]}[ds]:intervals.append({'dataset':ds,'model':model,'reference':reference,'metric':metric,**paired_moving_block_interval(delta,block,repeats=10000,seed=20261015)})
        for model,reference in [('global','independent'),('early_dm','global')]+([('recent_dm','global')] if ds!='bpi2012' else []):
            for metric in ['joint_profile_energy','peak_slowdown_crps']:
                pivot=jo[jo.dataset==ds].pivot(index='cutoff_us',columns='method',values=metric).sort_index();delta=(pivot[model]-pivot[reference]).to_numpy()
                for block in {'borg':[4,8],'bpi':[14,28],'bpi2012':[7,14]}[ds]:intervals.append({'dataset':ds,'model':model,'reference':reference,'metric':metric,**paired_moving_block_interval(delta,block,repeats=10000,seed=20261015)})
    pd.DataFrame(intervals).to_csv(OUT/'paired_intervals.csv',index=False)
    groups=read('runs/revision_borg_collective_risk_v1/group_scores.parquet');groups=groups[groups.period=='retrospective_final'];g=groups.groupby(['mode','projection','cutoff_us','seed']).crps.mean().groupby(['mode','projection']).mean().unstack();g.to_csv(OUT/'graph_ablations.csv')
    strat=groups.groupby(['mode','projection','size_bin','cutoff_us']).crps.mean().groupby(['mode','projection','size_bin']).agg(['mean','count']).reset_index();strat.to_csv(OUT/'graph_size_strata.csv',index=False)
    marginal=read('runs/revision_bpi_calibration_same_version_v2/marginals.csv');z=marginal[(marginal.period=='retrospective_final')&(marginal.horizon==14)].copy();z['date']=pd.to_datetime(z.cutoff_us,unit='us',utc=True)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False});fig,axes=plt.subplots(1,2,figsize=(10,3.1),layout='constrained')
    for name,label,color in [('identity','Original forecast','#4878a8'),('rolling60','Delayed correction','#e08028')]:
        v=z[z.calibration==name];axes[0].plot(v.date,v.predicted,label=label,color=color)
    v=z[z.calibration=='identity'];axes[0].plot(v.date,v.observed,label='Observed',color='#333333',linewidth=1.3);axes[0].set_ylabel('14-day resolution fraction');axes[0].set_title('BPI 2017: same-version mean correction');axes[0].legend(frameon=False,fontsize=8);axes[0].tick_params(axis='x',rotation=25)
    ix=np.arange(3);width=.23
    for shift,method,label,color in [(-1,'independent','Independent','#888888'),(0,'global','Global','#4878a8'),(1,'static','Typed','#e08028')]:axes[1].bar(ix+shift*width,[float(macro[(macro.dataset==ds)&(macro.method==method)].fraction_crps.iloc[0]) for ds in ['borg','bpi','bpi2012']],width,label=label,color=color)
    axes[1].set_xticks(ix,['Borg','BPI 2017','BPI 2012']);axes[1].set_ylabel('Count fraction CRPS');axes[1].set_title('Shared dependence versus typed incidence');axes[1].legend(frameon=False,fontsize=8);fig.savefig(ROOT/'paper_revision/figures/calibration_and_controls.pdf',metadata={'CreationDate':None,'ModDate':None});plt.close(fig)
    write_json(OUT/'manifest.json',{'status':'complete','sources':list({v['path']:v for v in SOURCES}.values()),'script_sha256':digest(__file__),'original_frozen_assets_verified':264,'fresh_frozen_assets_verified':82,'interpretation':'Original holdout scores, explicitly retrospective follow-ups, and separately frozen fresh confirmation are distinguished in manuscript.'})
    print(macro.to_string(index=False));print('Generated manuscript evidence',flush=True)


if __name__=='__main__':main()
