# SimpleFrameworkExample.py

# NumPyro is used to define probability distributions and sample parameters.
import numpyro as npo

# Set the number of host devices so NumPyro can run 4 MCMC chains in parallel.
npo.set_host_device_count(4)

# JAX NumPy works like NumPy but is compatible with JAX transformations.
import jax.numpy as jnp

# JAX's built-in ODE solver.
from jax.experimental.ode import odeint

# vmap lets us solve the same ODE independently for every patient.
from jax import vmap

# NumPyro probability distributions.
import numpyro.distributions as dist

# Allows importing BayesianFramework when running this file directly
# from inside the BayesianFramework package directory.
from pathlib import Path
import sys
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Bayesian fitting framework.
from BayesianFramework import BayesianFramework


# ---------------------------------------------------------------------
# 1. DEFINE THE ODE
# ---------------------------------------------------------------------

def ODE(y, t, k):
    """
    Simple exponential decay model:

        dy/dt = -k*y

    y = current state
    t = current time
    k = patient-specific decay rate
    """

    return -k * y


# ---------------------------------------------------------------------
# 2. DEFINE THE MODEL FUNCTION
# ---------------------------------------------------------------------

def ModelFn(globalParams, localParams, modelData):
    """
    Convert the model parameters into predicted observations.

    The framework calls:

        ModelFn(globalParams, localParams, modelData)

    and expects the function to return the model predictions.
    """

    # Get the patient-specific decay rates.
    #
    # Shape:
    #     k.shape == (nPatients,)
    k = localParams["k"]

    # Get the initial value for each patient.
    #
    # Shape:
    #     y0.shape == (nPatients,)
    y0 = modelData["y0"]

    # In this simple example, every patient is observed at the same times.
    #
    # modelData["times"] has shape:
    #
    #     (nPatients, nTimes)
    #
    # so we can simply take the first patient's time vector.
    #
    # Shape:
    #     times.shape == (nTimes,)
    times = modelData["times"][0]

    # Define how to solve the ODE for one patient.
    def SolveOnePatient(y0_i, k_i):

        # odeint evaluates the ODE solution at every time in `times`.
        return odeint(
            ODE,       # ODE function
            y0_i,      # initial condition
            times,     # times at which to return the solution
            k_i        # parameter passed to ODE
        )

    # Solve the ODE independently for every patient.
    #
    # vmap automatically applies SolveOnePatient over:
    #
    #     y0_i
    #     k_i
    #
    # for all patients.
    #
    # Shape:
    #     predictions.shape == (nPatients, nTimes)
    predictions = vmap(SolveOnePatient)(y0, k)

    # Return the model predictions to the framework.
    return predictions


# ---------------------------------------------------------------------
# 3. DEFINE THE GLOBAL PARAMETERS
# ---------------------------------------------------------------------

def GlobalFn():
    """
    Define parameters that are shared by every patient.

    Here, sigma is the observation error standard deviation.

    Unlike k, which is different for every patient, there is only
    one sigma shared by the entire dataset.
    """

    # sigma must be positive, so we give it a HalfNormal prior.
    sigma = npo.sample(
        "sigma",
        dist.HalfNormal(0.5)
    )

    # GlobalFn returns all global parameters as a dictionary.
    return {
        "sigma": sigma
    }


# ---------------------------------------------------------------------
# 4. DEFINE THE ERROR / OBSERVATION MODEL
# ---------------------------------------------------------------------

def ErrorFn(globalParams, localParams, modelData, modelOut):
    """
    Define how the observed data are distributed around the model output.

    Here we assume:

        observed_y ~ Normal(predicted_y, sigma)

    sigma is not fixed. It is inferred as a global parameter.
    """

    # Get the actual observed data.
    #
    # Shape:
    #     observations.shape == (nPatients, nTimes)
    observations = modelData["y"]

    # Get the global observation error inferred by the model.
    sigma = globalParams["sigma"]

    # Define the likelihood.
    #
    # modelOut contains the ODE predictions.
    #
    # NumPyro compares those predictions to the observed data
    # using the inferred sigma.
    npo.sample(
        "obs",
        dist.Normal(modelOut, sigma),
        obs=observations
    )


# ---------------------------------------------------------------------
# 5. CREATE A SMALL SYNTHETIC DATASET
# ---------------------------------------------------------------------

# Five observation times.
times = jnp.array([
    0.0,
    1.0,
    2.0,
    3.0,
    4.0
])


# Three patients, all starting with y = 1.
y0 = jnp.array([
    1.0,
    1.0,
    1.0
])


# Fake observations for three patients.
#
# Patient 1 decays slowly.
# Patient 2 decays moderately.
# Patient 3 decays quickly.
y = jnp.array([
    [1.03, 0.87, 0.85, 0.69, 0.72],
    [0.96, 0.80, 0.52, 0.47, 0.27],
    [1.04, 0.51, 0.36, 0.14, 0.13]
])

# BayesianFramework expects every entry in modelDataFull to have
# the patient dimension first.
#
# Therefore:
#
#     y.shape     == (3, 5)
#     y0.shape    == (3,)
#
# and we repeat the same time vector for all three patients so:
#
#     times.shape == (3, 5)
modelData = {
    "times": jnp.tile(times[None, :], (3, 1)),
    "y0": y0,
    "y": y
}


# ---------------------------------------------------------------------
# 6. DEFINE THE LOCAL PARAMETER PRIOR
# ---------------------------------------------------------------------

# Local parameters are parameters that vary from patient to patient.
#
# Each entry is:
#
#     (
#         prior mean location,
#         prior std of the population mean,
#         prior scale of the population std
#     )
#
# Therefore:
#
#     population mean of k ~ Normal(0.3, 0.5)
#
# and:
#
#     population std of k ~ HalfNormal(0.3)
#
# The framework then gives each patient their own k drawn from
# this inferred population distribution.
localParams = {
    "k": (0.3, 0.5, 0.3)
}


# ---------------------------------------------------------------------
# 7. CREATE THE BAYESIAN MODEL
# ---------------------------------------------------------------------

model = BayesianFramework(

    # Data used for fitting.
    modelDataFull=modelData,

    # Function that generates ODE predictions.
    ModelFn=ModelFn,

    # Function that defines the observation likelihood.
    ErrorFn=ErrorFn,

    # Definition of patient-specific parameters.
    localParams=localParams,

    # Function that defines shared global parameters.
    GlobalFn=GlobalFn
)


# ---------------------------------------------------------------------
# 8. RUN MCMC
# ---------------------------------------------------------------------

mcmc = model.RunMCMC(

    # Number of warmup / adaptation samples.
    numWarmup=1000,

    # Number of posterior samples per chain.
    numSamples=1000,

    # Number of independent MCMC chains.
    num_chains=4
)


# ---------------------------------------------------------------------
# 9. GET THE POSTERIOR SAMPLES
# ---------------------------------------------------------------------

# get_samples() combines the samples from all chains.
samples = mcmc.get_samples()


# ---------------------------------------------------------------------
# 10. INSPECT THE INFERRED PARAMETERS
# ---------------------------------------------------------------------

# "means" contains posterior samples of the inferred
# population mean of k.
print("Population mean samples:")
print(samples["means"].shape)


# "stds" contains posterior samples of the inferred
# population standard deviation of k.
print("Population std samples:")
print(samples["stds"].shape)


# "locals" contains the standard-normal latent variables
# used to generate each patient's value of k.
print("Local latent samples:")
print(samples["locals"].shape)


# "sigma" is the global observation error inferred by GlobalFn.
print("Global sigma samples:")
print(samples["sigma"].shape)


# Print the median inferred observation error.
print("Median sigma:")
print(jnp.median(samples["sigma"]))


# ---------------------------------------------------------------------
# 11. RECONSTRUCT THE PATIENT-SPECIFIC k VALUES
# ---------------------------------------------------------------------

# Because there is only one local parameter, the population
# mean is the first entry in "means".
#
# Shape:
#     meanK.shape == (nPosteriorSamples,)
meanK = samples["means"][:, 0]


# Likewise, the population standard deviation is the first
# entry in "stds".
#
# Shape:
#     stdK.shape == (nPosteriorSamples,)
stdK = samples["stds"][:, 0]


# "locals" contains the latent standard-normal value for
# every posterior sample and every patient.
#
# Shape:
#     z.shape == (nPosteriorSamples, nPatients)
z = samples["locals"][:, :, 0]


# BayesianFramework constructs each patient parameter as:
#
#     k = populationMean + z * populationStd
#
# Add a new axis to meanK and stdK so they broadcast
# across patients.
#
# Shape:
#     kSamples.shape == (nPosteriorSamples, nPatients)
kSamples = (
    meanK[:, None]
    + z * stdK[:, None]
)


# Compute one representative inferred k for each patient.
medianK = jnp.median(kSamples, axis=0)


print("Median k for each patient:")
print(medianK)