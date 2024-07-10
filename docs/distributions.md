# Distributions

This serves as reference for all distribution objects that we implement in the `ROLCH` package. 

!!! note 
    This page is somewhat under construction, since `MkDocs` [does not support docstring inheritance at the moment](https://github.com/mkdocstrings/mkdocstrings/issues/78).

All distributions are based on `scipy.stats` distributions. We implement the probability density function (PDF), the cumulative density function (CDF), the percentage point or quantile function (PPF) and the random variates (RVS) accordingly as pass-through. The link functions are implemented in the same way as in GAMLSS ([Rigby & Stasinopoulos, 2005](https://academic.oup.com/jrsssc/article-abstract/54/3/507/7113027)). The link functions and their derivatives derive from the `LinkFunction` base class.

## Base Class

::: rolch.abc.Distribution

## API Reference

::: rolch.DistributionNormal

::: rolch.DistributionT

::: rolch.DistributionJSU