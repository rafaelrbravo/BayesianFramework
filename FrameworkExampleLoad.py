import numpyro as npo
npo.set_host_device_count(4)

import jax.numpy as jnp
import numpyro.distributions as dist

import sys
from pathlib import Path
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from BayesianFramework import BayesianFramework


# ---------------------------------------------------------------------
# Testing likelihoods
# ---------------------------------------------------------------------

def TestLikelihoodFewFn(globalParams, localParams, data, modelOut):
    nFit = 3
    tumorSize = modelOut["tumorSize"]
    mask = jnp.arange(tumorSize.shape[-1]) < nFit
    npo.sample("obsTest", dist.Normal(tumorSize, globalParams["sigma"]).mask(mask), obs=data["tumorSize"])


def TestLikelihoodMoreFn(globalParams, localParams, data, modelOut):
    nFit = 7
    tumorSize = modelOut["tumorSize"]
    mask = jnp.arange(tumorSize.shape[-1]) < nFit
    npo.sample("obsTest", dist.Normal(tumorSize, globalParams["sigma"]).mask(mask), obs=data["tumorSize"])


# ---------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------

def AbsoluteErrorFn(modelOut, data):
    return jnp.mean(jnp.abs(modelOut["tumorSize"] - data["tumorSize"]))


def SquaredErrorFn(modelOut, data):
    return jnp.mean((modelOut["tumorSize"] - data["tumorSize"])**2)


# ---------------------------------------------------------------------
# Load trained framework
# ---------------------------------------------------------------------

bf = BayesianFramework.Load("SimpleTumorGrowth.bf")


# ---------------------------------------------------------------------
# Score training
# ---------------------------------------------------------------------

trainAbsoluteError = bf.ScoreTrain(ScoreFn=AbsoluteErrorFn)
trainSquaredError = bf.ScoreTrain(ScoreFn=SquaredErrorFn)


# ---------------------------------------------------------------------
# Test without MCMC
# ---------------------------------------------------------------------

bf.Test( testIndices=jnp.arange(5, 10), TestLikelihoodFn=None, nTrajectorySamples=500, printSummary=True)

noMCMCAbsoluteError = bf.ScoreTest(ScoreFn=AbsoluteErrorFn)
noMCMCSquaredError = bf.ScoreTest(ScoreFn=SquaredErrorFn)


# ---------------------------------------------------------------------
# Test with 3 observations
# ---------------------------------------------------------------------

bf.Test( testIndices=jnp.arange(5, 10), TestLikelihoodFn=TestLikelihoodFewFn, nPosteriorSamples=None, numWarmup=500, numSamples=500, num_chains=4, printSummary=True)

fewAbsoluteError = bf.ScoreTest(ScoreFn=AbsoluteErrorFn)
fewSquaredError = bf.ScoreTest(ScoreFn=SquaredErrorFn)


# ---------------------------------------------------------------------
# Test with 7 observations
# ---------------------------------------------------------------------

bf.Test( testIndices=jnp.arange(5, 10), TestLikelihoodFn=TestLikelihoodMoreFn, nPosteriorSamples=None, numWarmup=500, numSamples=500, num_chains=4, printSummary=True)

moreAbsoluteError = bf.ScoreTest(ScoreFn=AbsoluteErrorFn)
moreSquaredError = bf.ScoreTest(ScoreFn=SquaredErrorFn)


# ---------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------

print()
print("Mean absolute error")
print("-------------------")
print("Training:            ", jnp.mean(trainAbsoluteError))
print("Test, no observations:", jnp.mean(noMCMCAbsoluteError))
print("Test, 3 observations: ", jnp.mean(fewAbsoluteError))
print("Test, 7 observations: ", jnp.mean(moreAbsoluteError))

print()
print("Mean squared error")
print("------------------")
print("Training:            ", jnp.mean(trainSquaredError))
print("Test, no observations:", jnp.mean(noMCMCSquaredError))
print("Test, 3 observations: ", jnp.mean(fewSquaredError))
print("Test, 7 observations: ", jnp.mean(moreSquaredError))