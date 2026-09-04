from numpyro.diagnostics import summary
import numpy as np


def _GetStat(stats, key, inds=()):
    x = np.asarray(stats[key])
    return x[inds] if inds else x.item()


def _PrintRow(label, stats, inds=()):
    print(
        f"{label:<20} "
        f"{_GetStat(stats, 'mean', inds):>10.3f} "
        f"{_GetStat(stats, 'std', inds):>10.3f} "
        f"{_GetStat(stats, 'median', inds):>10.3f} "
        f"{_GetStat(stats, '5.0%', inds):>10.3f} "
        f"{_GetStat(stats, '95.0%', inds):>10.3f} "
        f"{_GetStat(stats, 'n_eff', inds):>10.1f} "
        f"{_GetStat(stats, 'r_hat', inds):>8.2f}"
    )

def _PrintSummary(mcmc, localNames, globalNames):
    stats = summary(mcmc.get_samples(group_by_chain=True))
    localNames = list(localNames)
    globalNames = list(globalNames)

    print(
        f"{'parameter':<20} {'mean':>10} {'std':>10} {'median':>10} "
        f"{'5.0%':>10} {'95.0%':>10} {'n_eff':>10} {'r_hat':>8}"
    )

    for name in globalNames:
        if name not in stats:
            continue

        shape = np.shape(stats[name]["mean"])

        if not shape:
            _PrintRow(name, stats[name])
        else:
            for inds in np.ndindex(shape):
                label = f"{name}[{','.join(map(str, inds))}]"
                _PrintRow(label, stats[name], inds)

    if "__localSamples__" in stats:
        shape = np.shape(stats["__localSamples__"]["mean"])

        for patient in range(shape[0]):
            for i, name in enumerate(localNames):
                _PrintRow(
                    f"{name}[{patient}]",
                    stats["__localSamples__"],
                    (patient, i)
                )