import numpyro as npo
import jax.numpy as jnp
import numpyro.distributions as dist
from jax import random
from numpyro.infer import init_to_median,NUTS,MCMC
from .InputValidator import _ValidateInit

class BayesianFramework():
    def __init__(self,modelDataFull,ModelFn,ErrorFn,localParams=None,GlobalFn=None,CovariateFn=None,choleskyConcentration=2.0,fixedLocalPriors=None):
        _ValidateInit(modelDataFull,ModelFn,ErrorFn,localParams,GlobalFn,CovariateFn,choleskyConcentration,fixedLocalPriors)
        self._modelDataFull=modelDataFull
        self._modelData=None
        self._localParams={} if localParams is None else localParams
        self._CovariateFn=CovariateFn
        self._ModelFn=ModelFn
        self._ErrorFn=ErrorFn
        self._GlobalFn=GlobalFn
        self._choleskyConcentration=choleskyConcentration
        self._fixedLocalPriors=fixedLocalPriors

        if self._localParams and self._fixedLocalPriors is None:
            params = jnp.array(list(self._localParams.values()))
            self._localMeanLocations = params[:, 0]
            self._localMeanStds = params[:, 1]
            self._localPriorStds = params[:, 2]
            self._sampleMeanInds = jnp.where(params[:, 1] > 0)[0]
            self._sampleStdInds = jnp.where(params[:, 2] > 0)[0]

        elif self._fixedLocalPriors is not None: 
            self._fixedLocalPriors = { "means": jnp.asarray(fixedLocalPriors["means"]), "stds": jnp.asarray(fixedLocalPriors["stds"]), }
            if len(self._localParams) > 1: self._fixedLocalPriors["cholesky"] = jnp.asarray( fixedLocalPriors["cholesky"])

    def _GenLocalMeansStds(self):
        localMeans = self._localMeanLocations
        if len(self._sampleMeanInds) > 0:
            sampledMeans = npo.sample( "means", dist.Normal( self._localMeanLocations[self._sampleMeanInds], self._localMeanStds[self._sampleMeanInds]))
            localMeans = localMeans.at[self._sampleMeanInds].set(sampledMeans)
        localStds = jnp.ones(len(self._localParams))
        if len(self._sampleStdInds) > 0:
            sampledStds = npo.sample( "stds", dist.HalfNormal( self._localPriorStds[self._sampleStdInds]))
            localStds = localStds.at[self._sampleStdInds].set(sampledStds)
        return localMeans, localStds

    def _GenLocalParams(self,dataSize):
        nLocal = len(self._localParams)
        uncorrelatedUnitParams = npo.sample( "locals", dist.Normal(0.0,1.0).expand((dataSize,nLocal)))
        if self._fixedLocalPriors is not None: 
            localMeans=self._fixedLocalPriors['means']
            localStds=self._fixedLocalPriors['stds']
            if nLocal > 1: cholesky=self._fixedLocalPriors['cholesky']
        else:
            localMeans,localStds = self._GenLocalMeansStds()
            if nLocal > 1: cholesky = npo.sample( "cholesky", dist.LKJCholesky( nLocal, concentration=self._choleskyConcentration))

        if nLocal == 1: correlatedUnitParams = uncorrelatedUnitParams
        else: correlatedUnitParams = uncorrelatedUnitParams @ cholesky.T
        localParams = localMeans + correlatedUnitParams * localStds
        return dict(zip(self._localParams.keys(),localParams.T))

    def _RunModel(self,**modelData):
        dataLen=len(modelData[next(iter(modelData))])
        if self._localParams: localParams=self._GenLocalParams(dataLen)
        else: localParams={}
        if self._GlobalFn is None: globalParams={}
        else: globalParams=self._GlobalFn()
        if self._CovariateFn is not None: 
            covariateEffects=self._CovariateFn(globalParams,localParams,modelData)
            for k,v in covariateEffects.items(): localParams[k]=localParams[k]+v
        modelOut=self._ModelFn(globalParams,localParams,modelData)
        self._ErrorFn(globalParams,localParams,modelData,modelOut)

    def SubsetData(self,indices): self._modelData = { k: v[indices] for k, v in self._modelDataFull.items() }

    def RunMCMC(self,numWarmup=2000,numSamples=2000,num_chains=4,acceptProb=0.95,dense_mass=False,medianSamples=50,rngKey=random.PRNGKey(0)):
        if self._modelData is None: self._modelData=self._modelDataFull
        kernel=NUTS(self._RunModel,init_strategy=init_to_median(num_samples=medianSamples),target_accept_prob=acceptProb,dense_mass=dense_mass)
        mcmc=MCMC(kernel,num_warmup=numWarmup,num_samples=numSamples,num_chains=num_chains,chain_method="parallel")
        mcmc.run(rngKey,extra_fields=("num_steps", "accept_prob"),**self._modelData)
        mcmc.print_summary()
        extra = mcmc.get_extra_fields(group_by_chain=True)
        print("Mean NUTS steps:", extra["num_steps"].mean())
        print("Median NUTS steps:", jnp.median(extra["num_steps"]))
        print("Max NUTS steps:", extra["num_steps"].max())
        print("Mean acceptance:", extra["accept_prob"].mean())
        print("Divergences:", extra["diverging"].sum())
        return mcmc
