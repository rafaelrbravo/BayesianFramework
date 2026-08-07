import numpyro as npo
import jax.numpy as jnp
import numpyro.distributions as dist
from jax import vmap,random
from numpyro.infer import init_to_median,NUTS,MCMC
from jax import jit

class BayesianFramework():
    def __init__(self,globalParams:dict[jnp.array],localParams:dict[jnp.array],modelDataFull:dict[jnp.array],CovariateFn=None,ModelFn=None,ErrorFn=None,choleskyConcentration=2.0):
        self.modelDataFull=modelDataFull
        self.modelData=None
        self.globalParams=globalParams
        self.localParams=localParams
        self.CovariateFn=CovariateFn
        self.ModelFn=ModelFn
        self.ErrorFn=ErrorFn
        self.choleskyConcentration=choleskyConcentration

#param entries local: priorMeanLocation, priorMeanStd, priorStd. if priorMeanStd=0, then priorMean=priorMeanLocation
#param entries global: priorMean, priorStd if priorStd=0, then halfnormal will be sampled with priorStd=priorMean
    def GenPriorMeanLocal(self,name):
        param=self.localParams[name]
        if param[1]>0: return npo.sample(f"{name} mean",dist.Normal(param[0],param[1]))
        else: return param[0]
    def GetPriorStdLocal(self,name):
        param=self.localParams[name]
        return npo.sample(f"{name} std",dist.HalfNormal(param[2]))

    def _GenPriorsLocal(self):
        return jnp.array([self.GenPriorMeanLocal(k) for k in self.localParams]),jnp.array([self.GetPriorStdLocal(k) for k in self.localParams])

    def GenPriorGlobal(self,k):
        param=self.globalParams[k]
        if param[1]>0: return npo.sample(f"{k}",dist.Normal(param[0],param[1]))
        else: return npo.sample(f"{k}",dist.HalfNormal(param[0]))


    def GenPriorsGlobal(self):
        return jnp.array([self.GenPriorGlobal(k) for k in self.globalParams])

    def _GenAllPriors(self,dataSize,choleskyConcentration=2.0):
        uncorrelatedUnitPriors=npo.sample("locals",dist.Normal(0.0,1.0).expand((dataSize,len(self.localParams))))
        cholesky=npo.sample("cholesky",dist.LKJCholesky(len(self.localParams),concentration=choleskyConcentration))
        localMeans,localStds=self._GenPriorsLocal()
        correlatedUnitPriors=uncorrelatedUnitPriors @ cholesky.T
        localPriors=localMeans + correlatedUnitPriors * localStds
        localPriorsDict=dict(zip(self.localParams.keys(),localPriors.T))

        globalPriors=self.GenPriorsGlobal()
        globalPriorsDict=dict(zip(self.globalParams.keys(),globalPriors))

        return localPriorsDict,globalPriorsDict

    def _RunModel(self):
        if self.modelData==None:
            self.modelData=self.modelDataFull
        nIndices=len(self.modelData[next(iter(self.modelData))])
        localParams,globalParams=self._GenAllPriors(nIndices,choleskyConcentration=self.choleskyConcentration)
        if self.CovariateFn is not None: 
            covariateEffects=self.CovariateFn(globalParams,localParams,self.modelData)
            for k,v in covariateEffects.items(): localParams[k]=localParams[k]+v
        modelOut=self.ModelFn(globalParams,localParams,self.modelData)
        self.ErrorFn(globalParams,localParams,self.modelData,modelOut)

    def SubsetData(self,indices):
        out={}
        for k in self.modelDataFull.keys(): out[k]=self.modelDataFull[k][indices]
        self.modelData=out

    def RunMCMC(self,numWarmup=2000,numSamples=2000,acceptProb=0.95,dense_mass=False,medianSamples=50,rngKey=random.PRNGKey(0),**kwargs):
        kernel=NUTS(self._RunModel,init_strategy=init_to_median(num_samples=medianSamples),target_accept_prob=acceptProb,dense_mass=dense_mass)
        mcmc=MCMC(kernel,num_warmup=numWarmup,num_samples=numSamples,num_chains=4,chain_method="parallel")
        mcmc.run(rngKey,**kwargs)
        mcmc.print_summary()
        return mcmc
