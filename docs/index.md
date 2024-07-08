# Welcome to `ROLCH`

`ROLCH` is a `Python` package for online distributional learning and online models for conditionally heteroskedastic data. We provide an online implementation of the well-known GAMLSS model using online coordinate descent (OCD).

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

np.set_printoptions(precision=3, suppress=True)

# Load Diabetes data set
# Add intercept (will not be regularized)
X, y = load_diabetes(return_X_y=True)
X = np.hstack((np.ones((X.shape[0], 1)), X))

# Initialise the estimator
online_gamlss_lasso = rolch.OnlineGamlss(
    distribution=rolch.DistributionT(), 
    method="lasso", 
    estimation_kwargs={"ic" : "bic"}
)

# Fit the model and print coefficients
# We fit on all but the last data point
online_gamlss_lasso.fit(
    y[:-1], 
    X[:-1, :], 
    X[:-1, :], 
    X[:-1, :]
)

print("LASSO Coefficients \n")
print(np.vstack(online_gamlss_lasso.betas).T)

# Update the fit and print new coefficients
online_gamlss_lasso.update(
    y[[-1]], 
    X[[-1], :], 
    X[[-1], :], 
    X[[-1], :]
)

print("\nCoefficients after update call \n")
print(np.vstack(online_gamlss_lasso.betas).T)
```
