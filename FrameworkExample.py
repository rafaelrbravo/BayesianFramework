# SimpleTumorGrowthExample.py

import numpyro as npo
npo.set_host_device_count(4)

import jax.numpy as jnp
import numpyro.distributions as dist
from jax import random
from jax.experimental.ode import odeint

import sys
from pathlib import Path
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from BayesianFramework import BayesianFramework


# ---------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------

# Exponential tumor growth:
#     dV/dt = gV
def ODE(V, t, g):
    return g * V


# localParams["g"] is the standardized patient-specific deviation.
# Convert it to the physical growth rate using the population mean/std.
def ModelFn(globalParams, localParams, data):
    g = globalParams["gMean"] + globalParams["gStd"] * localParams["g"]
    return odeint(ODE, data["V0"], data["times"], g)


# Population parameters and observation noise.
def GlobalFn(dataShapes):
    return {
        "gMean": npo.sample("gMean", dist.Normal(0.15, 0.1)),
        "gStd":  npo.sample("gStd", dist.HalfNormal(0.1)),
        "sigma": npo.sample("sigma", dist.HalfNormal(0.1)),
    }


# Training uses every tumor-size measurement.
def TrainLikelihoodFn(globalParams, localParams, data, modelOut):
    npo.sample("obs", dist.Normal(modelOut, globalParams["sigma"]), obs=data["tumorSize"])


# Test MCMC only uses the first nFit observations.
def TestLikelihoodFn(globalParams, localParams, data, modelOut):
    nFit = 3
    mask = jnp.arange(modelOut.shape[-1]) < nFit
    npo.sample("obsTest", dist.Normal(modelOut, globalParams["sigma"]).mask(mask), obs=data["tumorSize"])


# Score one trajectory against all available observations.
def ScoreFn(modelOut, data):
    return jnp.mean(jnp.abs(modelOut - data["tumorSize"]))


# ---------------------------------------------------------------------
# Generate 10 synthetic patient trajectories
# ---------------------------------------------------------------------

keyG, keyNoise = random.split(random.PRNGKey(0))
nPatients = 10
nTimes = 8
times = jnp.linspace(0, 10, nTimes)

# True population:
#     mean growth rate = 0.15
#     std growth rate  = 0.03
trueG = 0.15 + 0.03 * random.normal(keyG, (nPatients,))

# Give patients slightly different initial tumor sizes.
V0 = jnp.linspace(0.8, 1.2, nPatients)
trueTumorSize = V0[:, None] * jnp.exp(trueG[:, None] * times)
tumorSize = ( trueTumorSize + 0.05 * random.normal(keyNoise, trueTumorSize.shape))
data = { "times": jnp.tile(times, (nPatients, 1)), "V0": V0, "tumorSize": tumorSize, }


# ---------------------------------------------------------------------
# Construct framework
# ---------------------------------------------------------------------

bf = BayesianFramework(modelDataFull=data, ModelFn=ModelFn, TrainLikelihoodFn=TrainLikelihoodFn, localParamNames=["g"], GlobalParamFn=GlobalFn)

# Patients 0-4 train the population model.
# Patients 5-9 are completely held out.
bf.SetTrainIndices(jnp.arange(5))
bf.SetTestIndices(jnp.arange(5, 10))


# ---------------------------------------------------------------------
# 1. TRAINING
#
# Infer:
#
#     population mean of g
#     population std of g
#     g for each training patient
#     observation noise sigma
#
# using ALL observations from the five training patients.
# ---------------------------------------------------------------------

bf.Train(numWarmup=500, numSamples=500, num_chains=4, rngKey=random.PRNGKey(2))
trainScores = bf.ScoreTrain(ScoreFn=ScoreFn, nSamples=500, RNGkey=random.PRNGKey(3))
bf.PrintTrainSummary()


# ---------------------------------------------------------------------
# 2. TESTING WITHOUT MCMC
#
# Do not look at the test-patient observations.
#
# Sample population parameters from the training posterior, sample
# unseen-patient local parameters from the learned population
# distribution, and generate trajectories.
# ---------------------------------------------------------------------

bf.Test(TestLikelihoodFn=None, nTrajectorySamples=500, rngKey=random.PRNGKey(4))
noMCMCScores = bf.ScoreTest(ScoreFn=ScoreFn, nSamples=500, RNGkey=random.PRNGKey(5))


# ---------------------------------------------------------------------
# 3. TESTING WITH MCMC
#
# Use the learned training population distribution as the prior.
#
# TestLikelihoodFn exposes only the first nFit=3 tumor measurements.
# Each test patient's g is therefore updated using their early
# trajectory.
#
# ScoreFn then compares the posterior trajectories against all
# available measurements.
# ---------------------------------------------------------------------

bf.Test(TestLikelihoodFn=TestLikelihoodFn, nPosteriorSamples=None, numWarmup=500, numSamples=500, num_chains=4, rngKey=random.PRNGKey(6))
testMCMCScores = bf.ScoreTest(ScoreFn=ScoreFn, nSamples=500, RNGkey=random.PRNGKey(7))
bf.PrintTestSummary()


# ---------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------

print("True population mean g:", jnp.mean(trueG))
print("True population std g: ", jnp.std(trueG))

print()
print("Training score:      ", jnp.mean(trainScores))
print("Test score, no MCMC: ", jnp.mean(noMCMCScores))
print("Test score, MCMC:    ", jnp.mean(testMCMCScores))