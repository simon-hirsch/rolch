# Welcome to `ROLCH` - Regularized Online Learning for Conditional Heteroskedasticity

`ROLCH` is a `Python` package for online distributional learning and online models for conditionally heteroskedastic data. We provide an online implementation of the well-known GAMLSS model using online coordinate descent (OCD).

!!! note
    `ROLCH` is currently in the first alpha phase. Please expect changes to happen frequently.

## Installation

`ROLCH` is available on the [Python Package Index](https://pypi.org/project/rolch/) and can be installed via `pip`:

```shell
pip install rolch
```

## Example

The following few lines give an introduction. We use the `diabetes` data set and model the response variable \(Y\) as Student-\(t\) distributed, where all distribution parameters (location, scale and tail) are modelled conditional on the explanatory variables in \(X\).


```python
import rolch
import numpy as np
from sklearn.datasets import load_diabetes

X, y = load_diabetes(return_X_y=True)

# Model coefficients 
equation = {
    0 : "all", # Can also use "intercept" or np.ndarray of integers / booleans
    1 : "all", 
    2 : "all", 
}

# Create the estimator
online_gamlss_lasso = rolch.OnlineGamlss(
    distribution=rolch.DistributionT(),
    method="lasso",
    equation=equation,
    fit_intercept=True,
    estimation_kwargs={"ic": {i: "bic" for i in range(dist.n_params)}},
)

# Initial Fit
online_gamlss_lasso.fit(
    X=X[:-11, :], 
    y=y[:-11], 
)
print("Coefficients for the first N-11 observations \n")
print(online_gamlss_lasso.betas)

# Update call
online_gamlss_lasso.update(
    X=X[[-11], :], 
    y=y[[-11]]
)
print("\nCoefficients after update call \n")
print(online_gamlss_lasso.betas)

# Prediction for the last 10 observations
prediction = online_gamlss_lasso.predict(
    X=X[-10:, :]
)

print("\n Predictions for the last 10 observations")
# Location, scale and shape (degrees of freedom)
print(prediction)
```
