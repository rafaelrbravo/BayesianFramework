import numpyro as npo
import numpyro.distributions as dist
from jax import random, vmap
from numpyro.infer import init_to_median,NUTS,MCMC
from .InputValidator import _ValidateInit, _ValidateTrain, _ValidateTest, _ValidateScoreTrain, _ValidateScoreTest, _ValidateSetRng
from .OutputUtils import _PrintSummary,_PrintTestSummary,_GetRecord
import jax.numpy as jnp
import cloudpickle
from pprint import pprint
import jax
import gc

class BayesianFramework():
    def __init__(self,modelDataFull,ModelFn,localParamNames=None,GlobalParamFn=None,choleskyConcentration=2.0,rngSeed=None):
        _ValidateInit(modelDataFull,ModelFn,localParamNames,GlobalParamFn,choleskyConcentration,rngSeed)
        self._modelDataFull=modelDataFull
        self._localParamNames=[] if localParamNames is None else localParamNames
        self._ModelFn=ModelFn
        self._GlobalFn=GlobalParamFn
        self._choleskyConcentration=choleskyConcentration
        self._modelFnName=self._ModelFn.__name__
        self._globalFnName=None if self._GlobalFn is None else self._GlobalFn.__name__
        self._trainPost=None
        self._testPost=None
        self._trainIndices=None
        self._testNoMCMC=None
        self._testIndices=None
        self._globalNames=None
        self._trainParams = None
        self._testParams = None
        self._key=None if rngSeed is None else random.PRNGKey(rngSeed)

    def Train(self,trainIndices,TrainLikelihoodFn,numWarmup=2000,numSamples=2000,num_chains=4,acceptProb=0.95,dense_mass=False,medianSamples=50,printSummary=False,rngSeed=None):
        _ValidateTrain(self,trainIndices,TrainLikelihoodFn,numWarmup,numSamples,num_chains,acceptProb,dense_mass,medianSamples,printSummary,rngSeed)
        self._trainParams = { "numWarmup": numWarmup, "numSamples": numSamples, "numChains": num_chains, "acceptProb": acceptProb, "denseMass": dense_mass, "medianSamples": medianSamples, "trainLikelihoodName": TrainLikelihoodFn.__name__}
        self._trainIndices=trainIndices
        self._globalNames=None
        dataShapes={key:value.shape[1:] for key,value in self._modelDataFull.items()}
        kernel=NUTS( self._RunMCMCTrain, init_strategy=init_to_median(num_samples=medianSamples), target_accept_prob=acceptProb, dense_mass=dense_mass)
        mcmc=MCMC( kernel, num_warmup=numWarmup, num_samples=numSamples, num_chains=num_chains, chain_method="parallel")
        mcmc.run(self._NextKey(rngSeed),TrainLikelihoodFn,dataShapes,extra_fields=("num_steps","accept_prob"),**self._GetTrainData())
        self._trainPost=mcmc.get_samples(group_by_chain=False)
        self._testPost=None
        self._testNoMCMC=None
        self._testParams=None
        if printSummary: self._PrintTrainSummary(mcmc)
        self._ClearJAX()
        return mcmc

    def Test(self,testIndices,TestLikelihoodFn=None,nPosteriorSamples=None,numWarmup=2000,numSamples=2000,num_chains=4,acceptProb=0.95,dense_mass=False,medianSamples=50,nTrajectorySamples=None,printSummary=False,rngSeed=None):
        _ValidateTest(self,testIndices,TestLikelihoodFn,nPosteriorSamples,numWarmup,numSamples,num_chains,acceptProb,dense_mass,medianSamples,nTrajectorySamples,printSummary,rngSeed)
        self._testParams = { "mode":"NoMCMC" if TestLikelihoodFn is None else "MCMC","nPosteriorSamples": nPosteriorSamples, "numWarmup": numWarmup, "numSamples": numSamples, 
                            "numChains": num_chains, "acceptProb": acceptProb, "denseMass": dense_mass, "medianSamples": medianSamples, "nTrajectorySamples": nTrajectorySamples, "testLikelihoodName": None if TestLikelihoodFn is None else TestLikelihoodFn.__name__}
        self._testIndices=testIndices
        if TestLikelihoodFn is None: 
            result=self._TestNoMCMC(nTrajectorySamples,rngSeed)
            if printSummary: self._PrintTestSummary(None)
        else: 
            result=self._TestMCMC(TestLikelihoodFn,nPosteriorSamples,numWarmup,numSamples,num_chains,acceptProb,dense_mass,medianSamples,rngSeed)
            if printSummary: self._PrintTestSummary(result)
        return result

    def ScoreTrain(self,ScoreFn):
        _ValidateScoreTrain(self,ScoreFn)
        return self._ScoreModelOutput( ScoreFn,self.GetTrainModelOutput(), self._GetTrainData())

    def ScoreTest(self,ScoreFn): 
        _ValidateScoreTest(self,ScoreFn)
        return self._ScoreModelOutput( ScoreFn,self.GetTestModelOutput(), self._GetTestData())

    def GetTrainModelOutput(self):
        if self._trainPost is None: raise RuntimeError("Train must be run before getting training model output.")
        return self._GetModelOutput(self._trainPost,"__trainModelOutput__")

    def GetTestModelOutput(self):
        if self._testPost is not None: 
            outputs=[ self._GetModelOutput(post,"__testModelOutput__") for post in self._testPost]
            return {key:jnp.concatenate([output[key] for output in outputs],axis=0) for key in outputs[0]}
        if self._testNoMCMC is not None: return self._testNoMCMC
        raise RuntimeError("Test must be run before getting testing model output.")

    def SetRng(self,rngSeed): 
        _ValidateSetRng(rngSeed)
        self._key=random.PRNGKey(rngSeed)

    def GetRecord(self,print=False): 
        record=_GetRecord(self)
        if print: pprint(record)
        return record

    def Save(self,fileName):
        with open(fileName,"wb") as file:
            cloudpickle.dump(self,file)

    @staticmethod
    def Load(fileName):
        with open(fileName,"rb") as file:
            return cloudpickle.load(file)

    def _PrintTrainSummary(self,mcmc): 
        _PrintSummary(mcmc,self._localParamNames,self._globalNames,"--= Training posterior summary =--")

    def _PrintTestSummary(self,mcmc): 
        _PrintTestSummary(mcmc,self._testNoMCMC,self._localParamNames,self._globalNames)

    def _TestMCMC(self,TestLikelihoodFn,nPosteriorSamples=None,numWarmup=2000,numSamples=2000,num_chains=4, acceptProb=0.95,dense_mass=False,medianSamples=50,rngSeed=None):
        postKey,runKey=random.split(self._NextKey(rngSeed),2)
        posteriors=self._SampleTrainingPosterior(nPosteriorSamples,postKey)
        kernel=NUTS( self._RunMCMCTest, init_strategy=init_to_median(num_samples=medianSamples), target_accept_prob=acceptProb, dense_mass=dense_mass)
        testData=self._GetTestData()
        if nPosteriorSamples is None: 
            mcmc=MCMC(kernel,num_warmup=numWarmup,num_samples=numSamples, num_chains=num_chains,chain_method="parallel")
            mcmc.run(runKey,TestLikelihoodFn,posteriors, extra_fields=("num_steps","accept_prob"), **testData)
            self._testPost=[mcmc.get_samples(group_by_chain=False)]
            self._ClearJAX()
            return [mcmc]
        mcmcs=[]
        runKeys = random.split(runKey, nPosteriorSamples)
        for i in range(nPosteriorSamples):
            posterior={name:value[i] for name,value in posteriors.items()}
            mcmc=MCMC(kernel,num_warmup=numWarmup,num_samples=numSamples, num_chains=num_chains,chain_method="parallel")
            mcmc.run(runKeys[i],TestLikelihoodFn,posterior, extra_fields=("num_steps","accept_prob"), **testData)
            mcmcs.append(mcmc)
        self._testPost=[m.get_samples(group_by_chain=False) for m in mcmcs]
        self._ClearJAX()
        return mcmcs

    def _TestNoMCMC(self,nSamples,rngSeed=None):
        self._testPost=None
        postKey,sampleKey=random.split(self._NextKey(rngSeed),2)
        posterior=self._SampleTrainingPosterior(nSamples,postKey)
        testData=self._GetTestData()
        nTest=len(testData[next(iter(testData))])
        nLocal=len(self._localParamNames)
        if nLocal>0:
            localSamples=random.normal(sampleKey,(nSamples,nTest,nLocal))
            if nLocal>1:
                localSamples=jnp.einsum("spi,sji->spj",localSamples,posterior["__localCholesky__"])
            localParams={name:localSamples[:,:,i] for i,name in enumerate(self._localParamNames)}
        else: localParams={}
        globalParams={name:posterior[name] for name in self._globalNames}
        modelOut=vmap(vmap(self._ModelFn,in_axes=(None,0,0)),in_axes=(0,0,None))(globalParams,localParams,testData)
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

    def _RunMCMCTrain(self,TrainLikelihoodFn,dataShapes,**modelData):
        dataLen=len(modelData[next(iter(modelData))])
        localParams=self._GenLocalParamsTraining(dataLen) if self._localParamNames else {}
        globalParams=self._GlobalFn(dataShapes) if self._GlobalFn is not None else {}
        if self._globalNames is None: self._globalNames=list(globalParams)
        for name,value in globalParams.items(): npo.deterministic(f"__globalParam__{name}",value)
        modelOut=vmap(self._ModelFn,in_axes=(None,0,0))(globalParams,localParams,modelData)
        for name,value in modelOut.items(): npo.deterministic(f"__trainModelOutput__{name}",value)
        TrainLikelihoodFn(globalParams,localParams,modelData,modelOut)

    def _ScoreModelOutput(self,ScoreFn,modelOut,data):
        return vmap(vmap(ScoreFn,in_axes=(0,0)),in_axes=(0,None))(modelOut,data)

    def _SampleTrainingPosterior(self,nSamples,RNGkey):
        posterior={}
        if nSamples is None:
            if len(self._localParamNames)>1:
                posterior["__localCholesky__"]=jnp.median(self._trainPost["__localCholesky__"],axis=0)
            for name in self._globalNames: posterior[name]=jnp.median(self._trainPost[f"__globalParam__{name}"],axis=0)
        else:
            idxs=self._GetRandomIndices(nSamples,next(iter(self._trainPost.values())).shape[0],RNGkey)
            if len(self._localParamNames)>1: posterior["__localCholesky__"]=self._trainPost["__localCholesky__"][idxs]
            for name in self._globalNames:  posterior[name]=self._trainPost[f"__globalParam__{name}"][idxs]
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
        for name,value in modelOut.items(): npo.deterministic(f"__testModelOutput__{name}",value)
        TestLikelihoodFn(globalParams,localParams,modelData,modelOut)

    def _GetModelOutput(self,post,title):
        outputNames = [name for name in post if name.startswith(title)]
        return {name.removeprefix(title):post[name] for name in outputNames}

    def _NextKey(self,seed=None):
        if seed is not None: return random.PRNGKey(seed)
        self._key,key=random.split(self._key)
        return key

    def _GetTrainData(self):
        return { k: v[self._trainIndices] for k, v in self._modelDataFull.items() }

    def _GetTestData(self):
        return { k: v[self._testIndices] for k, v in self._modelDataFull.items() }

    def _ClearJAX(self):
        jax.clear_caches()
        gc.collect()