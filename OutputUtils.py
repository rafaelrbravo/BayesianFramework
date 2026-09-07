from numpyro.diagnostics import summary
import numpy as np
import jax.numpy as jnp


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
    stats = summary(mcmc.get_samples(group_by_chain=True), prob=0.95)

    localNames = list(localNames)
    globalNames = [] if globalNames is None else list(globalNames)

    if title is not None:
        print(title)

    print(
        f"{'parameter':<20} {'mean':>10} {'std':>10} {'median':>10} "
        f"{'2.5%':>10} {'97.5%':>10} {'n_eff':>10} {'r_hat':>8}"
    )

    for name in globalNames:
        statName = f"__globalParam__{name}"
    
        if statName not in stats:
            continue
        
        shape = np.shape(stats[statName]["mean"])
    
        if not shape:
            _PrintRow(name, stats[statName])
        else:
            for inds in np.ndindex(shape):
                label = f"{name}[{','.join(map(str, inds))}]"
                _PrintRow(label, stats[statName], inds)

    if "__localSamples__" in stats:
        shape = np.shape(stats["__localSamples__"]["mean"])

        for patient in range(shape[0]):
            for i, name in enumerate(localNames):
                _PrintRow(
                    f"{name}[{patient}]",
                    stats["__localSamples__"],
                    (patient, i),
                )


def _PrintTestSummary(testMCMC, testNoMCMC, localParamNames, globalParamNames):
    if testMCMC is not None:
        for i, mcmc in enumerate(testMCMC):
            title = (
                "--= Test posterior summary =--"
                if len(testMCMC) == 1
                else f"--= Test posterior summary {i + 1} =--"
            )

            _PrintSummary(
                mcmc,
                localParamNames,
                globalParamNames,
                title,
            )

        return

    print("--= Test model output from posterior sampling =--")

    for name, value in testNoMCMC.items():
        finite = jnp.isfinite(value)
        percentFinite = 100 * jnp.mean(finite)

        print(f"\n{name}")
        print(f"Posterior samples:      {value.shape[0]}")
        print(f"Test entries:           {value.shape[1]}")
        print(f"Output shape per entry: {value.shape[2:]}")
        print(f"Full output shape:      {value.shape}")
        print(f"Finite values:          {percentFinite:.1f}%")


def _GetRecord(framework):
    return {
        "localParamNames": framework._localParamNames,
        "globalParamNames": framework._globalNames,
        "choleskyConcentration": framework._choleskyConcentration,
        "trainIndices": framework._trainIndices,
        "testIndices": framework._testIndices,
        "trainParams": framework._trainParams,
        "testParams": framework._testParams,
        "rngKey": None if framework._key is None else np.asarray(framework._key).tolist(),
        "modelFnName": framework._modelFnName,
        "globalFnName": framework._globalFnName,
    }