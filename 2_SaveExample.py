import numpyro as npo
npo.set_host_device_count(4)

import jax.numpy as jnp
import numpyro.distributions as dist
from jax import random
from jax.experimental.ode import odeint

# Workaround to run this example directly from the repository without installing BayesianFramework.
# For normal use, install from the repository folder with `python -m pip install -e .` and import with `from BayesianFramework import BayesianFramework`.
import sys
from pathlib import Path
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from BayesianFramework import BayesianFramework


# ---------------------------------------------------------------------
# Generate 10 synthetic patient trajectories
# ---------------------------------------------------------------------

keyG, keyD, keySize, keyNoise = random.split(random.PRNGKey(2), 4)

nPatients = 10
nTimes = 11
times = jnp.linspace(0, 10, nTimes)
drugStart = 4.0

rho = 0.8
zG = random.normal(keyG, (nPatients,))
zDIndependent = random.normal(keyD, (nPatients,))
zD = rho * zG + jnp.sqrt(1.0 - rho**2) * zDIndependent

trueG = 0.15 + 0.03 * zG
trueD = 0.25 + 0.05 * zD

v0 = random.uniform(keySize, (nPatients,), minval=0.8, maxval=1.2)

timeOnDrug = jnp.maximum(times - drugStart, 0.0)
trueTumorSize = v0[:, None] * jnp.exp(trueG[:, None] * times - trueD[:, None] * timeOnDrug)
tumorSize = trueTumorSize + 0.05 * random.normal(keyNoise, trueTumorSize.shape)

data = {
    "times": jnp.tile(times, (nPatients, 1)),
    "drugStart": jnp.full(nPatients, drugStart),
    "v0": v0,
    "tumorSize": tumorSize,
}


# ---------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------

def ODE(V, t, g, d, drugStart):
    drugEffect = jnp.where(t >= drugStart, d, 0.0)
    return (g - drugEffect) * V


def ModelFn(globalParams, localParams, data):
    g = globalParams["gMean"] + globalParams["gStd"] * localParams["g"]
    d = globalParams["dMean"] + globalParams["dStd"] * localParams["d"]
    tumorSize = odeint(ODE, data["v0"], data["times"], g, d, data["drugStart"])
    return {"tumorSize": tumorSize}


def GlobalFn(dataShapes):
    return {
        "gMean": npo.sample("gMean", dist.Normal(0.15, 0.1)),
        "gStd": npo.sample("gStd", dist.HalfNormal(0.1)),
        "dMean": npo.sample("dMean", dist.Normal(0.25, 0.1)),
        "dStd": npo.sample("dStd", dist.HalfNormal(0.1)),
        "sigma": npo.sample("sigma", dist.HalfNormal(0.1)),
    }


def TrainLikelihoodFn(globalParams, localParams, data, modelOut):
    npo.sample("obs", dist.Normal(modelOut["tumorSize"], globalParams["sigma"]), obs=data["tumorSize"])


# ---------------------------------------------------------------------
# Train and save
# ---------------------------------------------------------------------

bf = BayesianFramework( modelDataFull=data, ModelFn=ModelFn, localParamNames=["g", "d"], GlobalParamFn=GlobalFn, rngSeed=0)

bf.Train( trainIndices=jnp.arange(5), TrainLikelihoodFn=TrainLikelihoodFn, numWarmup=500, numSamples=500, num_chains=4, printSummary=True)

bf.Save("SimpleTumorGrowth.bf")

# ---------------------------------------------------------------------
# Print a framework state summary
# ---------------------------------------------------------------------

bf.GetRecord(print=True)