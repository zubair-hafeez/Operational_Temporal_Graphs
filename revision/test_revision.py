import itertools
import numpy as np
from scipy.special import gammaln
from revision.calibration import eligible_history,correct,DAY_US
from revision.counts import sample_dm,dm_logpmf
from revision.reconcile import reconcile_trajectories
from revision.joint_profile import profile_scores
from revision.counts import fit_dm


def test_delayed_history_excludes_recent_and_future_labels():
    origins=[{'t':d*DAY_US} for d in [1,2,3,4,60,80,90,100,110]]
    kept=eligible_history(origins,100*DAY_US)
    assert [o['t']//DAY_US for o in kept]==[1,2,3,4,60,80]  # expanding fallback
    for o in kept:assert o['t']+14*DAY_US<=100*DAY_US


def test_correction_preserves_order_and_identity():
    p=np.array([[.01,.1,.4,.8],[.01,.01,.01,.99]])
    for b in [-2,0,2]:assert np.all(np.diff(correct(p,b),axis=1)>=0)
    np.testing.assert_allclose(correct(p,0),p,atol=1e-7)


def test_dirichlet_multinomial_moments_and_support():
    alpha=np.array([[1.,2.,3.,4.,5.]]);n=np.array([20]);s=sample_dm(n,alpha,100000,np.random.default_rng(7))
    assert np.all(s.sum(-1)==20) and np.all(s>=0)
    mu=20*alpha[0]/15;var=20*(alpha[0]/15)*(1-alpha[0]/15)*(20+15)/(1+15)
    np.testing.assert_allclose(s[:,0].mean(0),mu,atol=.04)
    np.testing.assert_allclose(s[:,0].var(0),var,rtol=.025)


def test_reconciliation_matches_enumeration():
    # Two Bernoulli bottom variables; beta-binomial upper forecast.
    p=np.array([.2,.7]);alpha=np.array([3.,1.]);states=np.array(list(itertools.product([0,1],repeat=2)))
    prior=np.prod(np.where(states,p,1-p),axis=1);total=states.sum(1)
    upper=np.exp(dm_logpmf(np.column_stack([total,2-total]),np.full(4,2),alpha))
    posterior=prior*upper;posterior/=posterior.sum();expected=posterior@states
    rng=np.random.default_rng(13);b=rng.binomial(1,p,size=(150000,2));cat=np.stack([b,1-b],axis=-1)
    reconc,diag=reconcile_trajectories(cat,alpha,100000,rng)
    np.testing.assert_allclose(reconc[:,:,0].mean(0),expected,atol=.006)
    assert diag['ess_fraction']>.1


def test_joint_profile_distinguishes_same_marginals_different_dependence():
    rng=np.random.default_rng(20);z=rng.binomial(1,.5,size=4000)
    matching=np.repeat(np.column_stack([z,z])[:,:,None],4,axis=2)
    opposite=np.repeat(np.column_stack([z,1-z])[:,:,None],4,axis=2)
    scores=[]
    for sample in [matching,opposite]:
        values=[]
        for truth in [np.zeros((2,4)),np.ones((2,4))]:
            values.append(profile_scores(sample,truth,np.ones(2),np.full((2,4),.5),np.ones((2,4)))[0])
        scores.append({k:np.mean([v[k] for v in values]) for k in ['joint_profile_energy','joint_variogram_fair']})
    assert scores[0]['joint_profile_energy']<scores[1]['joint_profile_energy']
    assert scores[0]['joint_variogram_fair']<scores[1]['joint_variogram_fair']


def test_count_fit_recovers_known_dispersion():
    alpha=np.array([[1.,2.,3.,4.,5.]]);p=alpha/15;n=np.full(3000,30)
    y=sample_dm(n,np.repeat(alpha,len(n),axis=0),1,np.random.default_rng(18))[0]
    records=[{'n':n[k:k+300],'prob':np.repeat(p,300,axis=0),'target':y[k:k+300]} for k in range(0,len(n),300)]
    fit=fit_dm(records,False)
    assert abs(fit['theta'][0]-np.log(15))<.1
