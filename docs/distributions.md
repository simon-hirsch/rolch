# Distributions

This serves as reference for all distribution objects that we implement in the `ondil` package. 

!!! note 
    This page is somewhat under construction, since `MkDocs` [does not support docstring inheritance at the moment](https://github.com/mkdocstrings/mkdocstrings/issues/78).

All distributions are based on `scipy.stats` distributions. We implement the probability density function (PDF), the cumulative density function (CDF), the percentage point or quantile function (PPF) and the random variates (RVS) accordingly as pass-through. The link functions are implemented in the same way as in GAMLSS ([Rigby & Stasinopoulos, 2005](https://academic.oup.com/jrsssc/article-abstract/54/3/507/7113027)). The link functions and their derivatives derive from the `LinkFunction` base class.


## Base Classes

| Base Distribution                          | Description                                                 |
| ------------------------------------------ | ----------------------------------------------------------- |
| [`Distribution`](#ondil.base.Distribution) | Base class for all distributions.                           |
| [`ScipyMixin`](#ondil.base.ScipyMixin)     | Base class for all distributions that are based on `scipy`. |


## List of Distributions

| Distribution                                                              | Description                            | `scipy` Equivalent      |
| ------------------------------------------------------------------------- | -------------------------------------- | ----------------------- |
| [`DistributionNormal`](#ondil.DistributionNormal)                         | Gaussian (mean and standard deviation) | `scipy.stats.norm`      |
| [`DistributionNormalMeanVariance`](#ondil.DistributionNormalMeanVariance) | Gaussian (mean and variance)           | `scipy.stats.norm`      |
| [`DistributionT`](#ondil.DistributionT)                                   | Student's $t$ distribution             | `scipy.stats.t`         |
| [`DistributionJSU`](#ondil.DistributionJSU)                               | Johnson's SU distribution              | `scipy.stats.johnsonsu` |
| [`DistributionGamma`](#ondil.DistributionGamma)                           | Gamma distribution                     | `scipy.stats.gamma`     |
| [`DistributionLogNormal`](#ondil.DistributionLogNormal)                   | Log-normal distribution                | `scipy.stats.lognorm`   |
| [`DistributionLogNormalMedian`](#ondil.DistributionLogNormalMedian)       | Log-normal distribution (median)       | -                       |
| [`DistributionLogistic`](#ondil.DistributionLogistic)                     | Logistic distribution                  | `scipy.stats.logistic`  |
| [`DistributionExponential`](#ondil.DistributionExponential)               | Exponential distribution               | `scipy.stats.expon`     |
| [`DistributionInverseGaussian`](#ondil.DistributionInverseGaussian)       | Inverse Gaussian distribution          | `scipy.stats.invgauss`  |
| [`DistributionBeta`](#ondil.DistributionBeta)                             | Beta distribution                      | `scipy.stats.beta`      |
| [`DistributionBetaInflated`](#ondil.DistributionBetaInflated)             | Beta Inflated distribution             | -                       |
| [`DistributionPowerExponential`](#ondil.DistributionPowerExponential)     | Power Exponential distribution         | -                       |

## API Reference

::: ondil.DistributionNormal

::: ondil.DistributionNormalMeanVariance

::: ondil.DistributionT

::: ondil.DistributionJSU

::: ondil.DistributionGamma

::: ondil.DistributionLogNormal

::: ondil.DistributionLogNormalMedian

::: ondil.DistributionLogistic

::: ondil.DistributionExponential

::: ondil.DistributionInverseGaussian

::: ondil.DistributionBeta

::: ondil.DistributionBetaInflated

::: ondil.DistributionPowerExponential


## Base Class

::: ondil.base.Distribution

::: ondil.base.ScipyMixin