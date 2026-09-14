import numpyro as npo
import numpyro.distributions as dist
from jax import random, vmap
from numpyro.infer import init_to_median,NUTS,MCMC
from .InputValidator import _ValidateInit, _ValidateTrain, _ValidateTest, _ValidateScoreTrain, _ValidateScoreTest, _ValidateSetRng
from .OutputUtils import _PrintSummary,_PrintTestSummary,_Save,_Load
import jax.numpy as jnp
from pprint import pprint
import jax
import gc
import functools
import numpy as np

class BayesianFramework():
    def __init__(self,modelDataFull,ModelFn,localParamNames=None,GlobalParamFn=None,choleskyConcentration=2.0,rngSeed=None):
        _ValidateInit(modelDataFull,ModelFn,localParamNames,GlobalParamFn,choleskyConcentration,rngSeed)
        self._modelDataFull=modelDataFull
        self._localParamNames=[] if localParamNames is None else localParamNames
        self._ModelFn=ModelFn
        self._GlobalFn=GlobalParamFn
        self._choleskyConcentration=choleskyConcentration
        self._modelFnName=self._GetFnName(self._ModelFn)
        self._globalFnName=self._GetFnName(self._GlobalFn)
        self._trainGlobals={}
        self._testGlobals={}
        self._trainLocals={}
        self._testLocals={}
        self._trainCholesky=None
        self._testCholesky=None
        self._trainModelOut=None
        self._testModelOut=None
        self._trainIndices=None
        self._testIndices=None
        self._globalNames=None
        self._trainParams = None
        self._testParams = None
        self._key=None if rngSeed is None else random.PRNGKey(rngSeed)

    def Train(self,trainIndices,TrainLikelihoodFn,numWarmup=2000,numSamples=2000,num_chains=4,acceptProb=0.95,dense_mass=False,medianSamples=50,printSummary=False,rngSeed=None,saveLocals=True,saveModelOut=False):
        _ValidateTrain(self,trainIndices,TrainLikelihoodFn,numWarmup,numSamples,num_chains,acceptProb,dense_mass,medianSamples,printSummary,rngSeed,saveLocals,saveModelOut)
        self._trainParams = { "numWarmup": numWarmup, "numSamples": numSamples, "numChains": num_chains, "acceptProb": acceptProb, "denseMass": dense_mass, "medianSamples": medianSamples, "trainLikelihoodName": self._GetFnName(TrainLikelihoodFn)}
        self._trainIndices=trainIndices
        self._globalNames=None
        self._trainGlobals={}
        self._trainLocals={}
        self._trainCholesky=None
        self._trainModelOut=None
        dataShapes={key:value.shape[1:] for key,value in self._modelDataFull.items()}
        kernel=NUTS( self._RunMCMCTrain, init_strategy=init_to_median(num_samples=medianSamples), target_accept_prob=acceptProb, dense_mass=dense_mass)
        mcmc=MCMC( kernel, num_warmup=numWarmup, num_samples=numSamples, num_chains=num_chains, chain_method="parallel")
        mcmc.run(self._NextKey(rngSeed),TrainLikelihoodFn,dataShapes,saveModelOut,extra_fields=("num_steps","accept_prob"),**self._GetTrainData())
        post=mcmc.get_samples(group_by_chain=False)
        self._trainGlobals=self._GetPost(post,"__globalParam__")
        self._trainCholesky=post.get("__cholesky__")
        if saveLocals: self._trainLocals=self._GetPost(post,"__localParam__")
        if saveModelOut: self._trainModelOut=self._GetPost(post,"__modelOutput__")
        self._testGlobals={}
        self._testLocals={}
        self._testCholesky=None
        self._testModelOut=None
        self._testParams=None
        self._testIndices=None
        if printSummary: self._PrintTrainSummary(mcmc)
        self._ClearJAX()
        return mcmc

    def Test(self,testIndices,TestLikelihoodFn=None,nGlobalSamples=None,numSamples=2000,numWarmup=2000,num_chains=4,acceptProb=0.95,dense_mass=False,medianSamples=50,printSummary=False,rngSeed=None,saveModelOut=False):
        _ValidateTest(self,testIndices,TestLikelihoodFn,nGlobalSamples,numWarmup,numSamples,num_chains,acceptProb,dense_mass,medianSamples,printSummary,rngSeed,saveModelOut)
        self._testParams={ "mode":"NoMCMC" if TestLikelihoodFn is None else "MCMC", "nGlobalSamples":nGlobalSamples, "numWarmup":numWarmup, "numSamples":numSamples, "numChains":num_chains, "acceptProb":acceptProb, "denseMass":dense_mass, "medianSamples":medianSamples, "testLikelihoodName":self._GetFnName(TestLikelihoodFn) }
        self._testIndices=testIndices
        self._testGlobals={}
        self._testLocals={}
        self._testCholesky=None
        self._testModelOut=None
        if TestLikelihoodFn is None: 
            result=self._TestNoMCMC(nGlobalSamples,numSamples,rngSeed,saveModelOut)
        else: 
            result=self._TestMCMC(TestLikelihoodFn,nGlobalSamples,numWarmup,numSamples,num_chains,acceptProb,dense_mass,medianSamples,rngSeed,saveModelOut)
            if printSummary: self._PrintTestSummary(result)
        return result

    def ScoreTrain(self,ScoreFn,nSamples=None,rngSeed=None):
        _ValidateScoreTrain(self,ScoreFn,nSamples,rngSeed)
        nPost=self._GetNTrainSamples()
        if nSamples is None: idxs=jnp.arange(nPost)
        else: idxs=self._GetRandomIndices( nSamples, nPost, self._NextKey(rngSeed))
        data=self._GetTrainData()
        if self._trainModelOut is not None: modelOut={ name:value[idxs] for name,value in self._trainModelOut.items() }
        else:
            globalParams={ name:value[idxs] for name,value in self._trainGlobals.items() }
            if self._localParamNames:
                localParams={ name:value[idxs] for name,value in self._trainLocals.items() }
                modelOut=vmap( vmap(self._ModelFn,in_axes=(None,0,0)), in_axes=(0,0,None))(globalParams,localParams,data)
            else: modelOut=vmap( vmap(self._ModelFn,in_axes=(None,None,0)), in_axes=(0,None,None))(globalParams,{},data)
        return vmap( vmap(ScoreFn,in_axes=(0,0)), in_axes=(0,None))(modelOut,data)

    def ScoreTest(self,ScoreFn,nSamples=None,rngSeed=None):
        _ValidateScoreTest(self,ScoreFn,nSamples,rngSeed)
        if self._testModelOut is not None: nLocal=next(iter(self._testModelOut.values())).shape[1]
        else: nLocal=next(iter(self._testLocals.values())).shape[1]
        if nSamples is None: idxs=jnp.arange(nLocal)
        else: idxs=self._GetRandomIndices( nSamples, nLocal, self._NextKey(rngSeed))
        data=self._GetTestData()
        if self._testModelOut is not None: modelOut={ name:value[:,idxs] for name,value in self._testModelOut.items() }
        else:
            localParams={ name:value[:,idxs] for name,value in self._testLocals.items() }
            modelOut=self._RunModelSamples( self._testGlobals, localParams, data)
        return vmap( vmap( vmap(ScoreFn,in_axes=(0,0)), in_axes=(0,None)), in_axes=(0,None))(modelOut,data)

    def Save( self, fileName, saveTrainLocals=True, saveTestLocals=True, saveTrainModelOut=False, saveTestModelOut=False, saveAllData=False):
        record=_Save( self, saveTrainLocals, saveTestLocals, saveTrainModelOut, saveTestModelOut, saveAllData)
        np.savez_compressed(fileName,**record)

    @classmethod
    def Load( cls, fileName, ModelFn, GlobalParamFn=None, modelDataFull=None):
        return _Load( cls, fileName, ModelFn, GlobalParamFn, modelDataFull)

    def _PrintTrainSummary(self,mcmc): 
        _PrintSummary(mcmc,self._localParamNames,self._globalNames,"--= Training posterior summary =--")

    def _PrintTestSummary(self,mcmc): 
        _PrintTestSummary(mcmc,self._testGlobals,self._localParamNames)

    def _TestMCMC(self,TestLikelihoodFn,nGlobalSamples=None,numWarmup=2000,numSamples=2000,num_chains=4, acceptProb=0.95,dense_mass=False,medianSamples=50,rngSeed=None,saveModelOut=False):
        postKey,runKey=random.split(self._NextKey(rngSeed),2)
        self._SetTestGlobals(nGlobalSamples,postKey)
        kernel=NUTS( self._RunMCMCTest, init_strategy=init_to_median(num_samples=medianSamples), target_accept_prob=acceptProb, dense_mass=dense_mass)
        testData=self._GetTestData()
        nGlobalSamples=1 if nGlobalSamples is None else nGlobalSamples
        runKeys = random.split(runKey, nGlobalSamples)
        mcmcs=[]
        localPosts=[]
        modelOutPosts=[]
        for i in range(nGlobalSamples):
            globs={name:value[i] for name,value in self._testGlobals.items()}
            cholesky=None if self._testCholesky is None else self._testCholesky[i]
            mcmc=MCMC(kernel,num_warmup=numWarmup,num_samples=numSamples, num_chains=num_chains,chain_method="parallel")
            mcmc.run(runKeys[i],TestLikelihoodFn,globs,cholesky,saveModelOut, extra_fields=("num_steps","accept_prob"), **testData)
            post=mcmc.get_samples(group_by_chain=False)
            localPosts.append(self._GetPost(post,"__localParam__"))
            if saveModelOut: modelOutPosts.append(self._GetPost(post,"__modelOutput__"))
            mcmcs.append(mcmc)
        self._testLocals={name:jnp.stack([p[name] for p in localPosts]) for name in localPosts[0]}
        if saveModelOut: self._testModelOut={name:jnp.stack([p[name] for p in modelOutPosts]) for name in modelOutPosts[0]}
        self._ClearJAX()
        return mcmcs

    def _TestNoMCMC(self,nGlobalSamples,nLocalSamples,rngSeed=None,saveModelOut=False):
        postKey,sampleKey=random.split(self._NextKey(rngSeed),2)
        self._SetTestGlobals(nGlobalSamples,postKey)
        nGlobalSamples=1 if nGlobalSamples is None else nGlobalSamples
        testData=self._GetTestData()
        nTest=len(testData[next(iter(testData))])
        nLocal=len(self._localParamNames)
        if nLocal>0:
            localSamples=random.normal(sampleKey,(nGlobalSamples,nLocalSamples,nTest,nLocal))
            if nLocal>1:
                localSamples=jnp.einsum("gspi,gji->gspj",localSamples,self._testCholesky)
            localParams={name:localSamples[:,:,:,i] for i,name in enumerate(self._localParamNames)}
        else: localParams={}
        self._testLocals=localParams
        if saveModelOut:
            self._testModelOut=self._RunModelSamples(self._testGlobals,self._testLocals,testData)

    def _RunModelSamples(self,globalParams,localParams,data):
        return vmap( vmap( vmap(self._ModelFn,in_axes=(None,0,0)),in_axes=(None,0,None)),in_axes=(0,0,None))(globalParams,localParams,data)

    def _GetRandomIndices(self,nSamples,length,RNGkey):
        return random.choice(RNGkey,length,shape=(nSamples,),replace=nSamples>length)

    def _GenLocalParamsTraining(self,dataSize):
        nLocal=len(self._localParamNames)
        localSamples=npo.sample("__localSample__",dist.Normal(0.0,1.0).expand((dataSize,nLocal)))
        if nLocal>1:
            cholesky=npo.sample("__cholesky__",dist.LKJCholesky(nLocal,concentration=self._choleskyConcentration))
            localSamples=localSamples@cholesky.T
        return dict(zip(self._localParamNames,localSamples.T))

    def _RunMCMCTrain(self,TrainLikelihoodFn,dataShapes,saveOutput,**modelData):
        dataLen=len(modelData[next(iter(modelData))])
        localParams=self._GenLocalParamsTraining(dataLen) if self._localParamNames else {}
        for name, value in localParams.items(): npo.deterministic(f"__localParam__{name}", value)
        globalParams=self._GlobalFn(dataShapes) if self._GlobalFn is not None else {}
        if self._globalNames is None: self._globalNames=list(globalParams)
        for name,value in globalParams.items(): npo.deterministic(f"__globalParam__{name}",value)
        modelOut=vmap(self._ModelFn,in_axes=(None,0,0))(globalParams,localParams,modelData)
        if saveOutput:
            for name,value in modelOut.items(): npo.deterministic(f"__modelOutput__{name}",value)
        TrainLikelihoodFn(globalParams,localParams,modelData,modelOut)

    def _SetTestGlobals(self,nGlobalSamples,RNGkey):
        if nGlobalSamples is None:
            self._testGlobals={name:jnp.expand_dims(jnp.median(value,axis=0),axis=0) for name,value in self._trainGlobals.items()}
            self._testCholesky=None if self._trainCholesky is None else jnp.expand_dims(jnp.median(self._trainCholesky,axis=0), axis=0)
        else:
            nTrainSamples=self._GetNTrainSamples()
            idxs=self._GetRandomIndices(nGlobalSamples,nTrainSamples,RNGkey)
            self._testGlobals={ name:value[idxs] for name,value in self._trainGlobals.items() }
            self._testCholesky=None if self._trainCholesky is None else self._trainCholesky[idxs]

    def _GenLocalParamsTesting(self,cholesky,dataSize):
        nLocal=len(self._localParamNames)
        localSamples=npo.sample("__localSamples__",dist.Normal(0.0,1.0).expand((dataSize,nLocal)))
        if nLocal>1:
            localSamples=localSamples @ cholesky.T
        return dict(zip(self._localParamNames,localSamples.T))

    def _RunMCMCTest(self,TestLikelihoodFn,globalParams,cholesky,saveOutput,**modelData):
        dataLen=len(modelData[next(iter(modelData))])
        localParams=self._GenLocalParamsTesting(cholesky,dataLen) if self._localParamNames else {}
        for name,value in localParams.items(): npo.deterministic(f"__localParam__{name}",value)
        modelOut=vmap(self._ModelFn,in_axes=(None,0,0))(globalParams,localParams,modelData)
        if saveOutput:
            for name,value in modelOut.items(): npo.deterministic(f"__modelOutput__{name}",value)
        TestLikelihoodFn(globalParams,localParams,modelData,modelOut)

    def _NextKey(self,rngSeed=None):
        if rngSeed is not None:
            self._key=random.PRNGKey(rngSeed)
        self._key,key=random.split(self._key)
        return key

    def _GetTrainData(self):
        return { k: v[self._trainIndices] for k, v in self._modelDataFull.items() }

    def _GetTestData(self):
        return { k: v[self._testIndices] for k, v in self._modelDataFull.items() }

    def _GetNTrainSamples(self):
        return self._trainParams["numSamples"] * self._trainParams["numChains"]

    def _ClearJAX(self):
        jax.clear_caches()
        gc.collect()

    def _GetFnName(self,Fn):
        if Fn is None: return None
        if isinstance(Fn, functools.partial): return Fn.func.__name__
        return getattr(Fn, "__name__", type(Fn).__name__)

    def _GetPost(self,post, prefix): 
        ret = { name.removeprefix(prefix): value for name, value in post.items() if name.startswith(prefix) }
        if not ret: return {}
        return ret