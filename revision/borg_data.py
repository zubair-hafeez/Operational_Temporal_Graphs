import json
import numpy as np
import pandas as pd
from revision.common import ROOT


def borg_origins():
    result=[]
    for period,directory in [('train',ROOT/'data/processed/revision_borg_oof_full_v1'),('development',ROOT/'data/processed/borg_joint_selected_v2/cohorts'),('retrospective_final',ROOT/'data/processed/borg_joint_selected_v2/final_cohorts')]:
        m=json.loads((directory/'manifest.json').read_text())
        for c in m['cohorts']:
            f=directory/c['folder'];q=pd.read_parquet(f/'queries.parquet');p=np.load(f/'cdf.npy')
            result.append({'t':int(c['cutoff_us']),'period':period,'q':q,'p':p,'y':q[['exit_1m','exit_5m','exit_15m','exit_60m']].to_numpy(np.int8)})
    return result
