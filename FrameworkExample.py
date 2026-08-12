import numpyro as npo
npo.set_host_device_count(4)
import inspect
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax import vmap,random
from jax import jit
import numpy as np
import os
from numpyro import handlers
from jax.nn import sigmoid
from numpyro.infer import init_to_median,NUTS,MCMC
from BayesianFramework import BayesianFramework
from pathlib import Path

@jit
def CalcLogTumorBurdenOnTreatment(tDrugStart,t,g,s,r,n0):
    return jnp.where(r<1e-8,n0+(g-s)*(t-tDrugStart),n0+g*(t-tDrugStart)-(s/r)*(1-jnp.exp(-r*(t-tDrugStart))))
@jit
def CalcLogTumorBurden(t0,tsPred,g,s,r,tDrugStart,tDrugEnd):
    nStart=g*(tDrugStart-t0)
    preTs=tsPred<tDrugStart
    onTs=(tsPred>=tDrugStart)&(tsPred<=tDrugEnd)

    preNs=g*(tsPred-t0)
    onNs=CalcLogTumorBurdenOnTreatment(tDrugStart,tsPred,g,s,r,nStart)
    nEnd=CalcLogTumorBurdenOnTreatment(tDrugStart,tDrugEnd,g,s,r,nStart)
    postNs=nEnd+g*(tsPred-tDrugEnd)

    return jnp.where(preTs,preNs,jnp.where(onTs,onNs,postNs))

@jit
def GetLogTumorBurden(ts,tDrugStart,tDrugEnd,tDisp,g,s,r):
    tDrugStart+=tDisp
    tDrugEnd+=tDisp
    t0=ts[0]
    tsPred=ts[1:]
    ns=CalcLogTumorBurden(t0,tsPred,g,s,r,tDrugStart,tDrugEnd)
    return jnp.concatenate([jnp.array([0.0]),ns])


GetLogTumorBurdenBatch=vmap(GetLogTumorBurden,in_axes=(0,0,0,0,0,0,0))

def RunModelBatch(globalParams,localParams,modelData):
    tumorBurden=GetLogTumorBurdenBatch(modelData['dates'],modelData['start'],modelData['stop'],60*sigmoid(localParams['tDisp']),jnp.exp(localParams['growth']),jnp.exp(localParams['sens']),jnp.exp(localParams['res']))
    logPsa0=jnp.log(modelData['psas'][:,0])+localParams['psa0Disp']*globalParams['errorStd']
    modelOut=tumorBurden+logPsa0[:,None]
    return npo.sample("obs",dist.Normal(modelOut[:,1:],globalParams['errorStd']).mask(modelData['masks'][:,1:]),obs=jnp.log(modelData['psas'][:,1:]))

def GenGlobals():
    return {'errorStd':npo.sample('errorStd',dist.HalfNormal(0.5))}

if __name__ == "__main__":

    localParams={'growth':(-4.0,2.0,1.0),'sens':(-4.0,2.0,1.0),'res':(-4.0,2.0,1.0),'tDisp':(-2.0,2.0,1.0),'psa0Disp':(0.0,0.0,0.0)}
    data=np.load(Path(__file__).parent/"Bulkl32.npz")
    ptData={"dates":data['dates'],"start":data['start'],"stop":data['stop'],"psas":data['psas'],"masks":data['masks']}
    bf=BayesianFramework(ptData,RunModelBatch,localParams,GenGlobals)
    bf.SubsetData(np.arange(50))
    bf.RunMCMC(numWarmup=1000,numSamples=1000)