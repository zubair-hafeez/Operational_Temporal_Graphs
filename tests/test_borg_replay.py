import numpy as np
import pandas as pd
import pytest
from otg.borg import BorgReplay, ORIGIN_US, PERSONAL, GLOBAL, RELATIONAL, assign_interval_labels


def events():
    rows=[
        # t, task, job, machine, type, cpu
        (0,0,0,0,0,np.nan),(0,0,0,0,1,.2),
        (ORIGIN_US+1_000_000,1,0,2,0,.3),
        (ORIGIN_US+2_000_000,1,0,0,1,.3),
        (ORIGIN_US+3_000_000,0,0,0,2,np.nan),
        (ORIGIN_US+4_000_000,0,0,2,0,np.nan),
        (ORIGIN_US+5_000_000,0,0,1,1,.4),
    ]
    df=pd.DataFrame(rows,columns=['time_us','task','job','machine','event_type','cpu'])
    df['source_row']=np.arange(len(df));df['missing']=-1
    df['priority']=2;df['sched_class']=1.;df['memory']=.1
    df['task']=df.task.astype('int32')
    return df


def replay():return BorgReplay(3,2,3,2)


def test_batching_preserves_state_and_features():
    df=events();a=replay();b=replay()
    a.apply(df)
    for i in range(len(df)):b.apply(df.iloc[i:i+1])
    np.testing.assert_allclose(a.features([0,1],ORIGIN_US+6_000_000),b.features([0,1],ORIGIN_US+6_000_000))
    assert a.evictions[0]==1
    assert a.cpu[0]==pytest.approx(.4)
    assert a.unknown_age[0]==False


def test_initial_state_age_remains_censored():
    a=replay();a.apply(events().iloc[:2])
    x=a.features([0],ORIGIN_US+10_000_000)[0]
    assert x[PERSONAL.index('unknown_state_age')]==1
    assert x[PERSONAL.index('log_state_age_s')]==pytest.approx(np.log1p(10))


def test_pending_machine_field_does_not_expose_assignment():
    df=events().iloc[:3].copy();a=replay();a.apply(df)
    df.loc[df.task==1,'machine']=0;b=replay();b.apply(df)
    np.testing.assert_allclose(a.features([1],ORIGIN_US+1_500_000),b.features([1],ORIGIN_US+1_500_000))


def test_shared_resource_context_uses_current_assignment():
    a=replay();a.apply(events().iloc[:4]);x=a.features([1],ORIGIN_US+2_500_000)
    names=PERSONAL+GLOBAL+RELATIONAL
    assert x[0,names.index('log_machine_running')]==pytest.approx(np.log1p(2))
    a.apply(events().iloc[4:]);x=a.features([1],ORIGIN_US+6_000_000)
    assert x[0,names.index('log_machine_running')]==pytest.approx(np.log1p(1))


def test_missing_request_does_not_erase_known_value():
    a=replay();a.apply(events().iloc[:6]);assert a.cpu[0]==pytest.approx(.2)


def test_pre_cutoff_features_are_independent_of_later_events():
    df=events();cutoff=ORIGIN_US+2_500_000
    a=replay();a.apply(df[df.time_us<=cutoff]);xa=a.features([0,1],cutoff)
    df.loc[df.time_us>cutoff,['cpu','priority','machine']]=[999,11,1]
    b=replay();b.apply(df[df.time_us<=cutoff]);xb=b.features([0,1],cutoff)
    np.testing.assert_array_equal(xa,xb)


def test_first_exit_labels_ties_and_censoring():
    intervals=pd.DataFrame({'task':np.array([0,0,0,1],np.int32),'time_us':[0,10,20,0],'end_us':[10,20,100,100],'end_type':[1,4,-1,-1],'state':[1,2,0,2]})
    q=pd.DataFrame({'task':np.array([0,0,1],np.int32),'cutoff_us':[5,10,95],'query_id':[0,1,2]})
    y=assign_interval_labels(intervals,q,10,100)
    assert y.exit.tolist()==[1,1,0]
    assert y.exit_type.tolist()==[1,4,-1]
    assert y.observed.tolist()==[True,True,False]


def test_cannot_query_inactive_task_or_past():
    a=replay()
    with pytest.raises(ValueError):a.features([0],ORIGIN_US)
    a.apply(events())
    with pytest.raises(ValueError):a.features([0],ORIGIN_US)
