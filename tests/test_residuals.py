import numpy as np
from otg.residuals import aggregate_inputs,residual_counts


def test_residual_forecast_uses_only_earlier_errors_and_valid_counts():
    p=np.tile([.1,.3,.6,.9],(20,1));states=np.ones(20);rel=np.repeat([1,2],10)[:,None]
    _,x=aggregate_inputs(p,states,rel,['job'],np.zeros_like(p))
    templates=[{name:{'size':v['size'],'residual':np.tile([-100,100,-100,100],(len(v['size']),1))} for name,v in x.items()}]
    a=residual_counts(x,templates,32,7)
    # Changing final labels cannot change any generated count.
    for v in x.values():v['truth'][:]=999
    b=residual_counts(x,templates,32,7)
    for name in a:
        np.testing.assert_array_equal(a[name],b[name])
        assert (a[name]>=0).all() and (a[name]<=x[name]['size'][None,:,None]).all()
        assert (np.diff(a[name],axis=2)>=0).all()
