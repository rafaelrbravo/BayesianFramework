from __future__ import annotations
from numpyro.diagnostics import summary
import numpy as np
import jax.numpy as jnp
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from BayesianFramework import BayesianFramework

def _GetStat(stats, key, inds=()):
    x = np.asarray(stats[key])
    return x[inds] if inds else x.item()


def _PrintRow(label, stats, inds=()):
    print(
        f"{label:<20} "
        f"{_GetStat(stats, 'mean', inds):>10.3f} "
        f"{_GetStat(stats, 'std', inds):>10.3f} "
        f"{_GetStat(stats, 'median', inds):>10.3f} "
        f"{_GetStat(stats, '2.5%', inds):>10.3f} "
        f"{_GetStat(stats, '97.5%', inds):>10.3f} "
        f"{_GetStat(stats, 'n_eff', inds):>10.1f} "
        f"{_GetStat(stats, 'r_hat', inds):>8.2f}"
    )


def _PrintSummary(mcmc, localNames, globalNames, title=None):
    stats=summary(mcmc.get_samples(group_by_chain=True),prob=0.95)

    localNames=list(localNames)
    globalNames=[] if globalNames is None else list(globalNames)

    if title is not None:
        print(title)

    print(
        f"{'parameter':<20} {'mean':>10} {'std':>10} {'median':>10} "
        f"{'2.5%':>10} {'97.5%':>10} {'n_eff':>10} {'r_hat':>8}"
    )

    for name in globalNames:
        statName=f"__globalParam__{name}"

        if statName not in stats:
            continue

        shape=np.shape(stats[statName]["mean"])

        if not shape:
            _PrintRow(name,stats[statName])
        else:
            for inds in np.ndindex(shape):
                label=f"{name}[{','.join(map(str,inds))}]"
                _PrintRow(label,stats[statName],inds)

    for name in localNames:
        statName=f"__localParam__{name}"

        if statName not in stats:
            continue

        shape=np.shape(stats[statName]["mean"])

        if not shape:
            _PrintRow(name,stats[statName])
        else:
            for inds in np.ndindex(shape):
                label=f"{name}[{','.join(map(str,inds))}]"
                _PrintRow(label,stats[statName],inds)


def _PrintTestSummary(testMCMC,testGlobals,localParamNames):
    for i,mcmc in enumerate(testMCMC):
        title=(
            "--= Testing posterior summary =--"
            if len(testMCMC)==1
            else f"--= Testing posterior summary {i+1} =--"
        )

        print(title)

        if testGlobals:
            print("Global parameter values:")
            for name,value in testGlobals.items():
                print(f"  {name}: {np.asarray(value[i])}")

        _PrintSummary(
            mcmc,
            localParamNames,
            [],
        )

def _AddInfo(record,prefix,info):
    if info is None:
        return

    for name,value in info.items():
        if not isinstance(name,str):
            raise TypeError(
                f"Saved dictionary keys must be strings; got {type(name).__name__}."
            )

        if not name:
            raise ValueError("Saved dictionary keys cannot be empty.")

        if "/" in name:
            raise ValueError(
                f"Saved dictionary key '{name}' cannot contain '/'."
            )

        if value is None:
            record[f"{prefix}/{name}/__none__"]=np.asarray(True)
        else:
            record[f"{prefix}/{name}"]=np.asarray(value)


def _GetInfo(record,prefix):
    info={}
    start=f"{prefix}/"
    noneSuffix="/__none__"

    keys=record.files if hasattr(record,"files") else record.keys()

    for key in keys:
        if not key.startswith(start):
            continue

        name=key.removeprefix(start)

        if name.endswith(noneSuffix):
            name=name.removesuffix(noneSuffix)

            if not name or "/" in name:
                raise ValueError(f"Invalid saved key '{key}'.")

            info[name]=None

        else:
            if not name or "/" in name:
                raise ValueError(f"Invalid saved key '{key}'.")

            value=record[key]
            info[name]=value.item() if value.ndim==0 else value

    return info

def _Save(
    framework,
    saveTrainLocals=True,
    saveTestLocals=True,
    saveTrainModelOut=False,
    saveTestModelOut=False,
    saveAllData=False,
):
    record={}

    # Metadata
    _AddInfo(record,"meta",{
        "localParamNames":np.asarray(framework._localParamNames,dtype=str),
        "globalNames":(
            None
            if framework._globalNames is None
            else np.asarray(framework._globalNames,dtype=str)
        ),
        "choleskyConcentration":framework._choleskyConcentration,
        "modelFnName":framework._modelFnName,
        "globalFnName":framework._globalFnName,
    })

    # General framework state
    _AddInfo(record,"state",{
        "trainIndices":framework._trainIndices,
        "testIndices":framework._testIndices,
        "rngKey":framework._key,
    })

    # Run settings
    _AddInfo(record,"trainParams",framework._trainParams)
    _AddInfo(record,"testParams",framework._testParams)

    # Training globals are always saved
    _AddInfo(record,"trainGlobals",framework._trainGlobals)

    if framework._trainCholesky is not None:
        record["trainCholesky"]=np.asarray(framework._trainCholesky)

    # Optional training locals
    if saveTrainLocals:
        _AddInfo(record,"trainLocals",framework._trainLocals)

    # Test globals/cholesky are saved as context for test locals
    if saveTestLocals and framework._testLocals:
        _AddInfo(record,"testGlobals",framework._testGlobals)
        _AddInfo(record,"testLocals",framework._testLocals)

        if framework._testCholesky is not None:
            record["testCholesky"]=np.asarray(framework._testCholesky)

    # Optional cached model outputs
    if saveTrainModelOut and framework._trainModelOut is not None:
        _AddInfo(record,"trainModelOut",framework._trainModelOut)

    if saveTestModelOut and framework._testModelOut is not None:
        _AddInfo(record,"testModelOut",framework._testModelOut)

    # Optional original model data
    if saveAllData:
        _AddInfo(record,"modelDataFull",framework._modelDataFull)

    return record

def _Load(
    FrameworkClass,
    fileName,
    ModelFn,
    GlobalParamFn=None,
    modelDataFull=None,
):
    with np.load(fileName,allow_pickle=False) as record:
        meta=_GetInfo(record,"meta")
        state=_GetInfo(record,"state")

        savedData=_GetInfo(record,"modelDataFull")

        if modelDataFull is None:
            if not savedData:
                raise ValueError(
                    "modelDataFull must be provided because the saved "
                    "framework does not contain the original model data."
                )
            modelDataFull=savedData

        localParamNames=meta.get("localParamNames")
        if localParamNames is None:
            localParamNames=[]
        else:
            localParamNames=list(localParamNames)

        framework=FrameworkClass(
            modelDataFull,
            ModelFn,
            localParamNames=localParamNames,
            GlobalParamFn=GlobalParamFn,
            choleskyConcentration=meta["choleskyConcentration"],
        )

        globalNames=meta.get("globalNames")
        framework._globalNames=(
            None
            if globalNames is None
            else list(globalNames)
        )

        # Run settings
        framework._trainParams=_GetInfo(record,"trainParams") or None
        framework._testParams=_GetInfo(record,"testParams") or None

        # Training posterior state
        framework._trainGlobals={
            name:jnp.asarray(value)
            for name,value in _GetInfo(record,"trainGlobals").items()
        }

        framework._trainLocals={
            name:jnp.asarray(value)
            for name,value in _GetInfo(record,"trainLocals").items()
        }

        framework._trainCholesky=(
            jnp.asarray(record["trainCholesky"])
            if "trainCholesky" in record.files
            else None
        )

        # Test posterior state
        framework._testGlobals={
            name:jnp.asarray(value)
            for name,value in _GetInfo(record,"testGlobals").items()
        }

        framework._testLocals={
            name:jnp.asarray(value)
            for name,value in _GetInfo(record,"testLocals").items()
        }

        framework._testCholesky=(
            jnp.asarray(record["testCholesky"])
            if "testCholesky" in record.files
            else None
        )

        # Optional cached model outputs
        trainModelOut=_GetInfo(record,"trainModelOut")
        testModelOut=_GetInfo(record,"testModelOut")

        framework._trainModelOut=(
            {
                name:jnp.asarray(value)
                for name,value in trainModelOut.items()
            }
            if trainModelOut
            else None
        )

        framework._testModelOut=(
            {
                name:jnp.asarray(value)
                for name,value in testModelOut.items()
            }
            if testModelOut
            else None
        )

        # General framework state
        framework._trainIndices=state.get("trainIndices")
        framework._testIndices=state.get("testIndices")

        if framework._trainIndices is not None:
            framework._trainIndices=np.asarray(framework._trainIndices)

        if framework._testIndices is not None:
            framework._testIndices=np.asarray(framework._testIndices)

        framework._key=(
            None
            if state.get("rngKey") is None
            else jnp.asarray(state["rngKey"])
        )

        # Warn if supplied functions do not match saved function names
        savedModelFnName=meta.get("modelFnName")
        currentModelFnName=framework._GetFnName(ModelFn)

        if savedModelFnName != currentModelFnName:
            warnings.warn(
                f"Saved ModelFn was '{savedModelFnName}', but the supplied "
                f"ModelFn is '{currentModelFnName}'.",
                RuntimeWarning,
            )

        savedGlobalFnName=meta.get("globalFnName")
        currentGlobalFnName=framework._GetFnName(GlobalParamFn)

        if savedGlobalFnName != currentGlobalFnName:
            warnings.warn(
                f"Saved GlobalParamFn was '{savedGlobalFnName}', but the "
                f"supplied GlobalParamFn is '{currentGlobalFnName}'.",
                RuntimeWarning,
            )

    return framework