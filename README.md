# ROLCH: Regularized Online Learning for Conditional Heteroskedasticity

[![Open Source Love](https://badges.frapsoft.com/os/v2/open-source.svg?v=103)](https://github.com/ellerbrock/open-source-badges/)

## Introduction

This package provides online estimation of models for distributional regression respectively models for conditional heteroskedastic data. The main contribution is an online/incremental implementation of the generalized additive models for location, shape and scale (GAMLSS, see [Rigby & Stasinopoulos, 2005](https://academic.oup.com/jrsssc/article-abstract/54/3/507/7113027)) developed in Hirsch, Berrisch & Ziel, 2024.

We're actively working on the package and welcome contributions from the community.

## Install from PyPI

The package is available from [pypi](https://pypi.org/project/rolch/).

1) `pip install rolch`. 
2) Enjoy

## Install from source:

1) Clone this repo.
2) Install the necessary dependencies from the `requirements.txt` using `conda create --name <env> --file requirements.txt`. 
3) Run `python3 -m build` to build the wheel.
4) Run `pip install dist/rolch-0.1.0-py3-none-any.whl` with the accurate version. If necessary, append `--force-reinstall`
5) Enjoy.

## Authors

- Simon Hirsch, University of Duisburg-Essen & Statkraft
- Jonathan Berrisch, University of Duisburg-Essen
- Florian Ziel, University of Duisburg-Essen

## Acknowledgements

Simon is employed at Statkraft and gratefully acknowledges support received from Statkraft for his PhD studies. This work contains the author's opinion and not necessarily reflects Statkraft's position.

## Dependencies

`ROLCH` is designed to have minimal dependencies. We rely on `python>=3.10`, `numpy`, `numba` and `scipy` in a reasonably up-to-date versions.

## Formater

We use `ruff` and `black`.
