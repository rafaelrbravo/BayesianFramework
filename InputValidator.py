import inspect
import numpy as np
import jax


# ====================================================================
# Public method validators
# ====================================================================

def _ValidateInit(
    modelDataFull,
    ModelFn,
    localParamNames=None,
    GlobalParamFn=None,
    choleskyConcentration=2.0,
    rngSeed=None,
):
    # ================================================================
    # modelDataFull
    # ================================================================

    if not isinstance(modelDataFull, (dict, np.lib.npyio.NpzFile)):
        raise TypeError("modelDataFull must be a dictionary.")

    if len(modelDataFull) == 0:
        raise ValueError("modelDataFull cannot be empty.")

    dataSize = None

    for name, value in modelDataFull.items():
        if not isinstance(name, str):
            raise TypeError(
                f"modelDataFull keys must be strings; got {type(name).__name__}."
            )

        if not name:
            raise ValueError("modelDataFull keys cannot be empty strings.")

        if not isinstance(value, (jax.Array, np.ndarray)):
            raise TypeError(
                f"modelDataFull['{name}'] must be a JAX or NumPy array; "
                f"got {type(value).__name__}."
            )

        if value.ndim == 0:
            raise ValueError(
                f"modelDataFull['{name}'] must have at least one dimension "
                "because its first dimension represents data entries."
            )

        if dataSize is None:
            dataSize = len(value)
        elif len(value) != dataSize:
            raise ValueError(
                "All arrays in modelDataFull must have the same first "
                f"dimension. Expected {dataSize}, but "
                f"modelDataFull['{name}'] has length {len(value)}."
            )

    if dataSize == 0:
        raise ValueError("modelDataFull must contain at least one data entry.")

    # ================================================================
    # Functions
    # ================================================================

    # ModelFn(globalParams, localParams, modelData)
    _CheckCallable(ModelFn, "ModelFn", 3, optional=False)

    # GlobalParamFn(dataShapes)
    _CheckCallable(GlobalParamFn, "GlobalParamFn", 1, optional=True)

    # ================================================================
    # localParamNames
    # ================================================================

    if localParamNames is not None:
        if not isinstance(localParamNames, (list, tuple)):
            raise TypeError(
                "localParamNames must be a list or tuple of strings, or None."
            )

        for i, name in enumerate(localParamNames):
            if not isinstance(name, str):
                raise TypeError(
                    f"localParamNames[{i}] must be a string; "
                    f"got {type(name).__name__}."
                )

            if not name:
                raise ValueError(f"localParamNames[{i}] cannot be empty.")

        if len(set(localParamNames)) != len(localParamNames):
            raise ValueError("localParamNames cannot contain duplicate names.")

    # ================================================================
    # choleskyConcentration
    # ================================================================

    try:
        concentration = np.asarray(choleskyConcentration)
    except Exception as e:
        raise TypeError(
            "choleskyConcentration must be a positive finite real scalar."
        ) from e

    if concentration.ndim != 0:
        raise TypeError("choleskyConcentration must be a scalar.")

    if not np.issubdtype(concentration.dtype, np.number):
        raise TypeError("choleskyConcentration must be numeric.")

    if np.issubdtype(concentration.dtype, np.complexfloating):
        raise TypeError("choleskyConcentration must be real.")

    concentration = concentration.item()

    if not np.isfinite(concentration):
        raise ValueError("choleskyConcentration must be finite.")

    if concentration <= 0:
        raise ValueError("choleskyConcentration must be > 0.")

    if rngSeed is not None:
        _CheckNonnegativeInt(rngSeed, "RNG seed")


def _ValidateTrain(
    framework,
    trainIndices,
    TrainLikelihoodFn,
    numWarmup,
    numSamples,
    num_chains,
    acceptProb,
    dense_mass,
    medianSamples,
    printSummary,
    rngSeed,
):
    _CheckIndices(
        trainIndices,
        len(framework._modelDataFull[next(iter(framework._modelDataFull))]),
        "trainIndices",
    )

    _CheckCallable(TrainLikelihoodFn, "TrainLikelihoodFn", 4, optional=False)
    _CheckNonnegativeInt(numWarmup, "numWarmup")
    _CheckPositiveInt(numSamples, "numSamples")
    _CheckPositiveInt(num_chains, "num_chains")
    _CheckProbability(acceptProb, "acceptProb")
    _CheckBool(dense_mass, "dense_mass")
    _CheckPositiveInt(medianSamples, "medianSamples")
    _CheckBool(printSummary, "printSummary")
    _CheckRngAvailable(framework, rngSeed)


def _ValidateTest(
    framework,
    testIndices,
    TestLikelihoodFn,
    nPosteriorSamples,
    numWarmup,
    numSamples,
    num_chains,
    acceptProb,
    dense_mass,
    medianSamples,
    nTrajectorySamples,
    printSummary,
    rngSeed,
):
    if framework._trainPost is None:
        raise RuntimeError("Train() must be run before Test().")

    _CheckIndices(
        testIndices,
        len(framework._modelDataFull[next(iter(framework._modelDataFull))]),
        "testIndices",
    )

    _CheckBool(printSummary, "printSummary")
    _CheckRngAvailable(framework, rngSeed)

    if TestLikelihoodFn is None:
        if nTrajectorySamples is None:
            raise ValueError(
                "nTrajectorySamples must be provided when "
                "TestLikelihoodFn is None."
            )

        _CheckPositiveInt(nTrajectorySamples, "nTrajectorySamples")

        if nPosteriorSamples is not None:
            raise ValueError(
                "nPosteriorSamples cannot be used when "
                "TestLikelihoodFn is None."
            )

        return

    # ================================================================
    # MCMC testing
    # ================================================================

    _CheckCallable(TestLikelihoodFn, "TestLikelihoodFn", 4, optional=False)

    if nTrajectorySamples is not None:
        raise ValueError(
            "nTrajectorySamples cannot be used when "
            "TestLikelihoodFn is provided."
        )

    if nPosteriorSamples is not None:
        _CheckPositiveInt(nPosteriorSamples, "nPosteriorSamples")

    _CheckNonnegativeInt(numWarmup, "numWarmup")
    _CheckPositiveInt(numSamples, "numSamples")
    _CheckPositiveInt(num_chains, "num_chains")
    _CheckProbability(acceptProb, "acceptProb")
    _CheckBool(dense_mass, "dense_mass")
    _CheckPositiveInt(medianSamples, "medianSamples")


def _ValidateScoreTrain(framework, ScoreFn):
    if framework._trainPost is None:
        raise RuntimeError("Train() must be run before ScoreTrain().")

    _CheckCallable(ScoreFn, "ScoreFn", 2, optional=False)


def _ValidateScoreTest(framework, ScoreFn):
    if framework._testPost is None and framework._testNoMCMC is None:
        raise RuntimeError("Test() must be run before ScoreTest().")

    _CheckCallable(ScoreFn, "ScoreFn", 2, optional=False)

def _ValidateSetRng(rngSeed):
    _CheckNonnegativeInt(rngSeed, "RNG seed")



# ====================================================================
# Shared checks
# ====================================================================

def _CheckCallable(fn, name, nArgs, optional=False):
    if fn is None:
        if optional:
            return
        raise TypeError(f"{name} cannot be None.")

    if not callable(fn):
        raise TypeError(
            f"{name} must be callable; got {type(fn).__name__}."
        )

    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return

    try:
        signature.bind(*([None] * nArgs))
    except TypeError as e:
        raise TypeError(
            f"{name} must accept {nArgs} positional argument"
            f"{'s' if nArgs != 1 else ''}. "
            f"Framework call will be incompatible with signature "
            f"{signature}."
        ) from e


def _CheckPositiveInt(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise TypeError(
            f"{name} must be an integer; got {type(value).__name__}."
        )

    if value <= 0:
        raise ValueError(f"{name} must be > 0.")


def _CheckNonnegativeInt(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise TypeError(
            f"{name} must be an integer; got {type(value).__name__}."
        )

    if value < 0:
        raise ValueError(f"{name} must be >= 0.")


def _CheckProbability(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError(
            f"{name} must be a real number; got {type(value).__name__}."
        )

    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")

    if not 0 < value < 1:
        raise ValueError(f"{name} must be between 0 and 1.")


def _CheckBool(value, name):
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(
            f"{name} must be bool; got {type(value).__name__}."
        )


def _CheckIndices(indices, dataSize, name):
    if not isinstance(indices, (list, tuple, np.ndarray, jax.Array)):
        raise TypeError(
            f"{name} must be a list, tuple, NumPy array, or JAX array."
        )

    arr = np.asarray(indices)

    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")

    if len(arr) == 0:
        raise ValueError(f"{name} cannot be empty.")

    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"{name} must contain integers.")

    if np.any(arr < 0):
        raise ValueError(f"{name} cannot contain negative indices.")

    if np.any(arr >= dataSize):
        raise IndexError(
            f"{name} contains an index outside modelDataFull, "
            f"which contains {dataSize} entries."
        )

    if len(np.unique(arr)) != len(arr):
        raise ValueError(f"{name} cannot contain duplicate indices.")


def _CheckRngAvailable(framework, rngSeed):
    if rngSeed is not None:
        _CheckNonnegativeInt(rngSeed, "RNG seed")
        return

    if framework._key is None:
        raise ValueError(
            "An RNG seed must be provided either when constructing the "
            "framework, via SetRng(), or for this call."
        )
