import numpyro as npo
import numpyro.distributions as dist
from jax import random, vmap
from numpyro.infer import init_to_median,NUTS,MCMC
from .InputValidator import (
    _ValidateInit,
    _ValidateTrain,
    _ValidateTest,
    _ValidateSetTrainIndices,
    _ValidateSetTestIndices,
    _ValidateScoreTrain,
    _ValidateScoreTest,
    _ValidatePrintTrainSummary,
    _ValidatePrintTestSummary,)
from .OutputUtils import _PrintSummary
import jax.numpy as jnp


class BayesianFramework():
    def __init__(self,modelDataFull,ModelFn,TrainLikelihoodFn,localParamNames=None,GlobalParamFn=None,choleskyConcentration=2.0):
        _ValidateInit(modelDataFull,ModelFn,TrainLikelihoodFn,localParamNames,GlobalParamFn,choleskyConcentration)
        self._modelDataFull=modelDataFull
        self._localParamNames=[] if localParamNames is None else localParamNames
        self._ModelFn=ModelFn
        self._TrainLikelihoodFn=TrainLikelihoodFn
        self._GlobalFn=GlobalParamFn
        self._choleskyConcentration=choleskyConcentration
        self._trainMCMC=None
        self._trainData=None
        self._trainIndices=None
        self._testMCMC=None
        self._testNoMCMC=None
        self._testIndices=None
        self._testData=None
        self._globalNames=None
        self._localNames=self._localParamNames
        if self._GlobalFn is not None: self._dataShapes={key:value.shape[1:] for key,value in self._modelDataFull.items()}

    def Train(self,numWarmup=2000,numSamples=2000,num_chains=4,acceptProb=0.95,dense_mass=False,medianSamples=50,rngKey=random.PRNGKey(0)):
        _ValidateTrain(self,numWarmup,numSamples,num_chains,acceptProb,dense_mass,medianSamples,rngKey)
        kernel=NUTS( self._RunMCMCTrain, init_strategy=init_to_median(num_samples=medianSamples), target_accept_prob=acceptProb, dense_mass=dense_mass)
        mcmc=MCMC( kernel, num_warmup=numWarmup, num_samples=numSamples, num_chains=num_chains, chain_method="parallel")
        mcmc.run(rngKey,extra_fields=("num_steps","accept_prob"),**self._trainData)
        self._trainMCMC=mcmc
        return self._trainMCMC

    def Test(self,TestLikelihoodFn=None,nPosteriorSamples=None,numWarmup=2000,numSamples=2000,num_chains=4,acceptProb=0.95,dense_mass=False,medianSamples=50,rngKey=random.PRNGKey(0),nTrajectorySamples=None):
        _ValidateTest(self,TestLikelihoodFn,nPosteriorSamples,numWarmup,numSamples,num_chains,acceptProb,dense_mass,medianSamples,rngKey,nTrajectorySamples)
        if TestLikelihoodFn is None: return self._TestNoMCMC(nTrajectorySamples,rngKey)
        return self._TestMCMC(TestLikelihoodFn,nPosteriorSamples,numWarmup,numSamples,num_chains,acceptProb,dense_mass,medianSamples,rngKey)

    def SetTrainIndices(self,indices):
        _ValidateSetTrainIndices(self,indices)
        self._trainIndices=indices
        self._trainData={k:v[indices] for k,v in self._modelDataFull.items()}

    def SetTestIndices(self,indices):
        _ValidateSetTestIndices(self,indices)
        self._testIndices=indices
        self._testData={k:v[indices] for k,v in self._modelDataFull.items()}

    def ScoreTrain(self,ScoreFn,nSamples,RNGkey):
        _ValidateScoreTrain(self,ScoreFn,nSamples,RNGkey)
        return self._ScoreTrajectories( ScoreFn,self._trainMCMC, self._trainData, nSamples, RNGkey, "__trainModelOutput__")

    def ScoreTest(self,ScoreFn,nSamples,RNGkey): 
        _ValidateScoreTest(self,ScoreFn,nSamples,RNGkey)
        if self._testMCMC is not None: return jnp.array([self._ScoreTrajectories( ScoreFn,test, self._testData, nSamples, RNGkey, "__testModelOutput__") for test in self._testMCMC])
        else: return self._ScoreTrajectoriesArray( ScoreFn,self._testNoMCMC, self._testData, nSamples, RNGkey)

    def PrintTrainSummary(self): 
        _ValidatePrintTrainSummary(self)
        _PrintSummary(self._trainMCMC,self._localNames,self._globalNames)

    def PrintTestSummary(self): 
        _ValidatePrintTestSummary(self)
        for mcmc in self._testMCMC: _PrintSummary(mcmc,self._localNames,self._globalNames)

    def GetTrainModelOutput(self):
        if self._trainMCMC is None: raise RuntimeError("Train must be run before getting training model output.")
        return self._trainMCMC.get_samples( group_by_chain=False)["__trainModelOutput__"]

    def GetTestModelOutput(self):
        if self._testMCMC is not None: return [ mcmc.get_samples( group_by_chain=False)["__testModelOutput__"] for mcmc in self._testMCMC]
        if self._testNoMCMC is not None: return self._testNoMCMC
        raise RuntimeError("Test must be run before getting testing model output.")

    def GetTrainData(self): return self._trainData
    def GetTestData(self): return self._testData
    def GetTrainIndices(self): return self._trainIndices
    def GetTestIndices(self): return self._testIndices
    def GetTrainMCMC(self): return self._trainMCMC
    def GetTestMCMC(self): return self._testMCMC

    def _TestMCMC(self,TestLikelihoodFn,nPosteriorSamples=None,numWarmup=2000,numSamples=2000,num_chains=4, acceptProb=0.95,dense_mass=False,medianSamples=50,rngKey=random.PRNGKey(0)):
        self._testNoMCMC=None
        posteriorKey,rngKey=random.split(rngKey)
        posteriors=self._SampleTrainingPosterior(nPosteriorSamples,posteriorKey)
        kernel=NUTS( self._RunMCMCTest, init_strategy=init_to_median(num_samples=medianSamples), target_accept_prob=acceptProb, dense_mass=dense_mass)
        if nPosteriorSamples is None: 
            mcmc=MCMC(kernel,num_warmup=numWarmup,num_samples=numSamples, num_chains=num_chains,chain_method="parallel")
            mcmc.run(rngKey,TestLikelihoodFn,posteriors, extra_fields=("num_steps","accept_prob"), **self._testData)
            self._testMCMC=[mcmc]
            return self._testMCMC
        self._testMCMC=[]
        for i in range(nPosteriorSamples):
            rngKey,mcmcKey=random.split(rngKey)
            posterior={name:value[i] for name,value in posteriors.items()}
            mcmc=MCMC(kernel,num_warmup=numWarmup,num_samples=numSamples, num_chains=num_chains,chain_method="parallel")
            mcmc.run(mcmcKey,TestLikelihoodFn,posterior, extra_fields=("num_steps","accept_prob"), **self._testData)
            self._testMCMC.append(mcmc)
        return self._testMCMC

    def _TestNoMCMC(self,nSamples,RNGkey):
        self._testMCMC=None
        posteriorKey,localKey=random.split(RNGkey)
        posterior=self._SampleTrainingPosterior(nSamples,posteriorKey)
        nTest=len(self._testData[next(iter(self._testData))])
        nLocal=len(self._localParamNames)
        if nLocal>0:
            localSamples=random.normal(localKey,(nSamples,nTest,nLocal))
            if nLocal>1:
                localSamples=jnp.einsum("spi,sji->spj",localSamples,posterior["__localCholesky__"])
            localParams={name:localSamples[:,:,i] for i,name in enumerate(self._localParamNames)}
        else: localParams={}
        globalParams={name:posterior[name] for name in self._globalNames}
        modelOut=vmap(vmap(self._ModelFn,in_axes=(None,0,0)),in_axes=(0,0,None))(globalParams,localParams,self._testData)
        self._testNoMCMC=modelOut
        return modelOut

    def _GetRandomIndices(self,nSamples,length,RNGkey):
        return random.choice(RNGkey,length,shape=(nSamples,),replace=nSamples>length)

    def _GenLocalParamsTraining(self,dataSize):
        nLocal=len(self._localParamNames)
        localSamples=npo.sample("__localSamples__",dist.Normal(0.0,1.0).expand((dataSize,nLocal)))
        if nLocal>1:
            cholesky=npo.sample("__localCholesky__",dist.LKJCholesky(nLocal,concentration=self._choleskyConcentration))
            localSamples=localSamples@cholesky.T
        return dict(zip(self._localParamNames,localSamples.T))

    def _RunMCMCTrain(self,**modelData):
        dataLen=len(modelData[next(iter(modelData))])
        localParams=self._GenLocalParamsTraining(dataLen) if self._localParamNames else {}
        globalParams=self._GlobalFn(self._dataShapes) if self._GlobalFn is not None else {}
        self._globalNames=globalParams.keys()
        modelOut=vmap(self._ModelFn,in_axes=(None,0,0))(globalParams,localParams,modelData)
        npo.deterministic("__trainModelOutput__",modelOut)
        self._TrainLikelihoodFn(globalParams,localParams,modelData,modelOut)

    def _ScoreTrajectoriesArray(self,ScoreFn,samples,data,nSamples,RNGkey):
        indices=self._GetRandomIndices(nSamples,len(samples),RNGkey)
        subset=samples[indices]
        return vmap(vmap(ScoreFn,in_axes=(0,0)),in_axes=(0,None))(subset,data)

    def _ScoreTrajectories(self,ScoreFn,mcmc,data,nSamples,RNGkey,name):
        samples=mcmc.get_samples(group_by_chain=False)[name]
        return self._ScoreTrajectoriesArray(ScoreFn,samples,data,nSamples,RNGkey)

    def _SampleTrainingPosterior(self,nSamples,RNGkey):
        samples=self._trainMCMC.get_samples(group_by_chain=False)
        posterior={}
        if nSamples is None:
            if len(self._localParamNames)>1:
                posterior["__localCholesky__"]=jnp.median(samples["__localCholesky__"],axis=0)
            for name in self._globalNames: posterior[name]=jnp.median(samples[name],axis=0)
        else:
            idxs=self._GetRandomIndices(nSamples,next(iter(samples.values())).shape[0],RNGkey)
            if len(self._localParamNames)>1: posterior["__localCholesky__"]=samples["__localCholesky__"][idxs]
            for name in self._globalNames: posterior[name]=samples[name][idxs]
        return posterior

    def _GenLocalParamsTesting(self,posterior,dataSize):
        nLocal=len(self._localParamNames)
        localSamples=npo.sample( "__localSamples__", dist.Normal(0.0,1.0).expand((dataSize,nLocal)))
        if nLocal>1: localSamples=localSamples @ posterior["__localCholesky__"].T
        return dict(zip(self._localParamNames,localSamples.T))

    def _RunMCMCTest(self,TestLikelihoodFn,posterior,**modelData):
        dataLen=len(modelData[next(iter(modelData))])
        localParams=self._GenLocalParamsTesting(posterior,dataLen) if self._localParamNames else {}
        globalParams={name:posterior[name] for name in self._globalNames}
        modelOut=vmap(self._ModelFn,in_axes=(None,0,0))(globalParams,localParams,modelData)
        npo.deterministic("__testModelOutput__",modelOut)
        TestLikelihoodFn(globalParams,localParams,modelData,modelOut)