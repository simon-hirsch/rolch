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

| Distribution                                      | Description                            | `scipy` Base            |
| ------------------------------------------------- | -------------------------------------- | ----------------------- |
| [`Normal`](#ondil.Normal)                         | Gaussian (mean and standard deviation) | `scipy.stats.norm`      |
| [`NormalMeanVariance`](#ondil.NormalMeanVariance) | Gaussian (mean and variance)           | `scipy.stats.norm`      |
| [`T`](#ondil.T)                                   | Student's $t$ distribution             | `scipy.stats.t`         |
| [`JSU`](#ondil.JSU)                               | Johnson's SU distribution              | `scipy.stats.johnsonsu` |
| [`Gamma`](#ondil.Gamma)                           | Gamma distribution                     | `scipy.stats.gamma`     |
| [`LogNormal`](#ondil.LogNormal)                   | Log-normal distribution                | `scipy.stats.lognorm`   |
| [`LogNormalMedian`](#ondil.LogNormalMedian)       | Log-normal distribution (median)       | -                       |
| [`Logistic`](#ondil.Logistic)                     | Logistic distribution                  | `scipy.stats.logistic`  |
| [`Exponential`](#ondil.Exponential)               | Exponential distribution               | `scipy.stats.expon`     |
| [`Beta`](#ondil.Beta)                             | Beta distribution                      | `scipy.stats.beta`      |
| [`Gumbel`](#ondil.Gumbel)                         | Gumbel distribution                    | `scipy.stats.gumbel_r`  |
| [`InverseGaussian`](#ondil.InverseGaussian)       | Inverse Gaussian distribution          | `scipy.stats.invgauss`  |
| [`BetaInflated`](#ondil.BetaInflated)             | Beta Inflated distribution             | -                       |
| [`ReverseGumbel`](#ondil.ReverseGumbel)           | Reverse Gumbel distribution            | `scipy.stats.gumbel_r`  |
| [`InverseGamma`](#ondil.InverseGamma)             | Inverse Gamma distribution             | `scipy.stats.invgamma`  |

## API Reference

::: ondil.Normal

::: ondil.NormalMeanVariance

::: ondil.T

::: ondil.JSU

::: ondil.Gamma

::: ondil.LogNormal

::: ondil.LogNormalMedian

::: ondil.Logistic

::: ondil.Exponential

::: ondil.InverseGaussian

::: ondil.Beta

::: ondil.Gumbel

::: ondil.BetaInflated

::: ondil.ReverseGumbel

::: ondil.InverseGamma


## Base Class

::: ondil.base.Distribution

::: ondil.base.ScipyMixin
