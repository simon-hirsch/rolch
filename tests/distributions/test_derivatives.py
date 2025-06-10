# %%
import numpy as np
import pytest
import rpy2.robjects as robjects

from .utils import get_distributions_with_gamlss

N = 100
CLIP_BOUNDS = (-1e5, 1e5)


@pytest.mark.parametrize(
    "distribution",
    get_distributions_with_gamlss(),
    ids=lambda dist: dist.__class__.__name__,
)
def test_distribution_derivatives(distribution):
    """Test that Python derivatives match R GAMLSS derivatives for different distributions."""

    # Set seed for reproducibility
    np.random.seed(42)

    # Generate random data within distribution support
    y = np.random.uniform(
        np.clip(distribution.distribution_support[0], *CLIP_BOUNDS),
        np.clip(distribution.distribution_support[1], *CLIP_BOUNDS),
        N,
    )
    robjects.globalenv["y"] = robjects.FloatVector(y)

    # Generate random parameters within support bounds
    theta = np.array(
        [
            np.random.uniform(
                np.clip(distribution.parameter_support[i][0], *CLIP_BOUNDS),
                np.clip(distribution.parameter_support[i][1], *CLIP_BOUNDS),
                N,
            )
            for i in range(distribution.n_params)
        ]
    ).T

    # Assign R variables programmatically
    for i, param_name in distribution.parameter_names.items():
        robjects.globalenv[param_name] = robjects.FloatVector(theta[:, i])

    # Obtain derivatives from R
    code = f"""
        dist <- gamlss.dist::{distribution.corresponding_gamlss}()
        all_derivative_funcs <- c("dldm", "dldd", "d2ldm2", "d2ldd2", "d2ldmdd")
        available_funcs <- all_derivative_funcs[sapply(all_derivative_funcs, function(f) !is.null(dist[[f]]))]
        setNames(lapply(available_funcs, function(f) {{
          do.call(dist[[f]], mget(names(formals(dist[[f]])), envir = .GlobalEnv))
        }}), available_funcs)
        """

    R_list = robjects.r(code)

    # Map R derivative names to Python method calls
    derivative_mapping = {
        "dldm": lambda: distribution.dl1_dp1(y, theta=theta, param=0),
        "dldd": lambda: distribution.dl1_dp1(y, theta=theta, param=1),
        "d2ldm2": lambda: distribution.dl2_dp2(y, theta=theta, param=0),
        "d2ldd2": lambda: distribution.dl2_dp2(y, theta=theta, param=1),
        "d2ldmdd": lambda: distribution.dl2_dpp(y, theta=theta, params=(0, 1)),
    }

    # Compare R and Python derivatives - only for available derivatives
    available_derivatives = R_list.names
    for key in available_derivatives:
        if key in derivative_mapping:
            assert np.allclose(derivative_mapping[key](), R_list.rx2(key)), (
                f"Derivative {key} doesn't match for {distribution.__class__.__name__}"
            )
