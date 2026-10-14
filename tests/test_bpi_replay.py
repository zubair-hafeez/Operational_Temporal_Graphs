import numpy as np
import pandas as pd
from otg.bpi import BPIReplay,DAY_US


def events(rows):
    return pd.DataFrame(rows,columns=['case','time_us','activity','lifecycle','resource','offer','offer_id'])


def test_future_resolution_does_not_change_prefix_and_last_handler_is_not_schedule():
    e=events([(0,DAY_US,'A_Create Application','complete',0,0,None),
              (0,2*DAY_US,'W_Complete application','start',1,0,None),
              (0,3*DAY_US,'W_Complete application','schedule',0,0,None),
              (1,3*DAY_US,'A_Create Application','complete',0,0,None),
              (0,5*DAY_US,'A_Pending','complete',0,0,None)])
    vocab=sorted(e.activity.unique());a=BPIReplay(2,2,vocab);b=BPIReplay(2,2,vocab)
    a.apply(e.iloc[:4])
    for i in range(4):b.apply(e.iloc[i:i+1])
    x,_=a.features([0,1],4*DAY_US);xx,_=b.features([0,1],4*DAY_US)
    np.testing.assert_allclose(x,xx)
    assert a.resource[0]==1 and a.resource[1]==2 and not a.resolved[0]
    b.apply(e.iloc[4:]);assert b.resolved[0] and not a.resolved[0]


def test_unknown_resource_has_no_shared_context_and_offers_are_prefix_only():
    e=events([(0,DAY_US,'A_Create Application','complete',0,0,None),
              (1,DAY_US,'A_Create Application','complete',0,0,None),
              (0,2*DAY_US,'O_Created','complete',0,1,'offer1'),
              (0,3*DAY_US,'O_Accepted','complete',0,1,'offer1')])
    a=BPIReplay(2,1,sorted(e.activity.unique()));a.apply(e.iloc[:3])
    x,g=a.features([0,1],2*DAY_US)
    assert np.isclose(x[0,3],np.log(2)) and np.isclose(x[0,4],np.log(2))
    assert np.all(x[:,-5:]==0)
    a.apply(e.iloc[3:]);y,_=a.features([0],3*DAY_US)
    assert y[0,4]==0 and np.isclose(y[0,3],np.log(2))
