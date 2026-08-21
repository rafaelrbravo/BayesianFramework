# SimpleFrameworkExample.py

import numpyro as npo                              # Bayesian sampling framework
npo.set_host_device_count(4)                      # Allow 4 MCMC chains to run in parallel

import numpyro.distributions as dist              # Probability distributions
import jax.numpy as jnp                           # JAX-compatible NumPy
from jax import vmap                              # Vectorize operations across patients
from jax.experimental.ode import odeint            # JAX ODE solver

from pathlib import Path
import sys
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from BayesianFramework import BayesianFramework   # Bayesian fitting framework


# Model

def ODE(y, t, k):
    return -k * y                                 # Exponential decay: dy/dt = -k*y


def SolveODE(y0, k, times):
    return odeint(ODE, y0, times, k)              # Solve one patient's trajectory


def ModelFn(globalParams, localParams, modelData):
    k = localParams["k"]                          # Patient-specific decay rates
    y0 = modelData["y0"]                          # Patient-specific initial values
    times = modelData["times"][0]                 # Shared observation times
    return vmap(SolveODE, in_axes=(0, 0, None))(  # Solve independently across patients
        y0, k, times
    )


# Global parameters and observation model

def GlobalFn():
    sigma = npo.sample("sigma", dist.HalfNormal(0.5))  # Shared observation-error SD
    return {"sigma": sigma}


def ErrorFn(globalParams, localParams, modelData, modelOut):
    npo.sample(
        "obs",
        dist.Normal(modelOut, globalParams["sigma"]),   # Observations centered on predictions
        obs=modelData["y"]                              # Measured patient trajectories
    )


# Synthetic data

times = jnp.array([0., 1., 2., 3., 4.])           # Five observation times
y0 = jnp.array([1., 1., 1.])                      # Initial value for each patient

y = jnp.array([                                    # Observed trajectories
    [1.03, 0.87, 0.85, 0.69, 0.72],              # Patient 1: slow decay
    [0.96, 0.80, 0.52, 0.47, 0.27],              # Patient 2: moderate decay
    [1.04, 0.51, 0.36, 0.14, 0.13]               # Patient 3: fast decay
])

modelData = {
    "times": jnp.tile(times[None, :], (3, 1)),    # Shape: (patients, times)
    "y0": y0,                                      # Shape: (patients,)
    "y": y                                          # Shape: (patients, times)
}


# Local parameter priors

localParams = {
    "k": (0.3, 0.5, 0.3)                          # (mean location, mean SD, population SD scale)
}


# Fit model

model = BayesianFramework(
    modelDataFull=modelData,                       # Patient data
    ModelFn=ModelFn,                               # Generates model predictions
    ErrorFn=ErrorFn,                               # Defines observation likelihood
    localParams=localParams,                       # Patient-specific parameter priors
    GlobalFn=GlobalFn                              # Shared parameters
)

mcmc = model.RunMCMC(
    numWarmup=1000,                                # Adaptation samples
    numSamples=1000,                               # Posterior samples per chain
    num_chains=4                                   # Independent chains
)

samples = mcmc.get_samples()                       # Combine posterior samples across chains


# Inspect posterior

print("Population mean:", samples["means"].shape)  # Population mean of k
print("Population std:", samples["stds"].shape)    # Population SD of k
print("Local latents:", samples["locals"].shape)   # Patient standard-normal latent values
print("Global sigma:", samples["sigma"].shape)     # Observation-error samples
print("Median sigma:", jnp.median(samples["sigma"]))


# Reconstruct patient-specific parameters

meanK = samples["means"][:, 0]                     # Population mean for each posterior sample
stdK = samples["stds"][:, 0]                       # Population SD for each posterior sample
z = samples["locals"][:, :, 0]                     # Patient latent values

kSamples = meanK[:, None] + z * stdK[:, None]      # k = population mean + z * population SD
medianK = jnp.median(kSamples, axis=0)              # Posterior median k for each patient

print("Median k for each patient:", medianK)