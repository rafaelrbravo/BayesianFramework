import inspect
import numbers
import numpy as np
import jax


def _ValidateInit( modelDataFull, ModelFn, localParams=None, GlobalFn=None, CovariateFn=None, choleskyConcentration=2.0):
    # ================================================================
    # modelDataFull
    # ================================================================

    if not isinstance(modelDataFull, dict):
        raise TypeError(
            "modelDataFull must be a dictionary."
        )

    if len(modelDataFull) == 0:
        raise ValueError(
            "modelDataFull cannot be empty."
        )

    dataSize = None

    for name, value in modelDataFull.items():

        if not isinstance(name, str):
            raise TypeError(
                f"modelDataFull keys must be strings; got {type(name).__name__}."
            )

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


    # ================================================================
    # Functions
    # ================================================================

    def CheckCallable(fn, name, nArgs, optional=False):

        if fn is None:
            if optional:
                return
            raise TypeError(f"{name} cannot be None.")

        if not callable(fn):
            raise TypeError(
                f"{name} must be callable; got {type(fn).__name__}."
            )

        # Check that its Python signature can accept the number of
        # positional arguments the framework will supply.
        #
        # Some valid callable objects do not expose an inspectable
        # signature, so lack of a signature alone is not an error.
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


    # ModelFn(globalParams, localParams, modelData)
    CheckCallable(
        ModelFn,
        "ModelFn",
        3,
        optional=False
    )

    # GlobalFn()
    CheckCallable(
        GlobalFn,
        "GlobalFn",
        0,
        optional=True
    )

    # CovariateFn(globalParams, localParams, modelData)
    CheckCallable(
        CovariateFn,
        "CovariateFn",
        3,
        optional=True
    )


    # ================================================================
    # localParams
    # ================================================================

    if localParams is not None:

        if not isinstance(localParams, dict):
            raise TypeError(
                "localParams must be a dictionary or None."
            )

        for name, specification in localParams.items():

            if not isinstance(name, str):
                raise TypeError(
                    "localParams keys must be strings; "
                    f"got {type(name).__name__}."
                )

            # Must be convertible to a numeric array.
            try:
                spec = np.asarray(specification)
            except Exception as e:
                raise TypeError(
                    f"localParams['{name}'] must contain three numeric values."
                ) from e

            # Require precisely:
            #
            # [meanLocation, meanStd, priorStd]
            #
            if spec.ndim != 1 or len(spec) != 3:
                raise ValueError(
                    f"localParams['{name}'] must contain exactly three values: "
                    "[meanLocation, meanStd, priorStd]."
                )

            if not np.issubdtype(spec.dtype, np.number):
                raise TypeError(
                    f"localParams['{name}'] must contain numeric values."
                )

            if np.issubdtype(spec.dtype, np.complexfloating):
                raise TypeError(
                    f"localParams['{name}'] cannot contain complex values."
                )

            if not np.all(np.isfinite(spec)):
                raise ValueError(
                    f"localParams['{name}'] cannot contain NaN or infinity."
                )

            meanLocation, meanStd, priorStd = spec

            # Mean location may be any finite real value.

            if meanStd < 0:
                raise ValueError(
                    f"localParams['{name}'] meanStd must be >= 0; "
                    f"got {meanStd}."
                )

            if priorStd < 0:
                raise ValueError(
                    f"localParams['{name}'] priorStd must be >= 0; "
                    f"got {priorStd}."
                )


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
        raise TypeError(
            "choleskyConcentration must be a scalar."
        )

    if not np.issubdtype(concentration.dtype, np.number):
        raise TypeError(
            "choleskyConcentration must be numeric."
        )

    if np.issubdtype(concentration.dtype, np.complexfloating):
        raise TypeError(
            "choleskyConcentration must be real."
        )

    concentration = concentration.item()

    if not np.isfinite(concentration):
        raise ValueError(
            "choleskyConcentration must be finite."
        )

    if concentration <= 0:
        raise ValueError(
            "choleskyConcentration must be > 0."
        )