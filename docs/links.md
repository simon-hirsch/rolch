# Link functions 

## Link functions 

A link function \(g(x)\) is a smooth, monotonic function of \(x\). 

For all link functions, we implement 

- the link \(g(x)\)
- the inverse \(g^{-1}(x)\)
- the derivative of the link function $\frac{\partial g(x)}{\partial x}$.
- the first derivative _of the inverse_ of the link function \(\frac{\partial g(x)^{-1}}{\partial x}\). The choice of the inverse is justified by Equation (7) in Hirsch, Berrisch & Ziel ([2024](https://github.com/simon-hirsch/rolch/blob/main/paper.pdf)). 

The link functions implemented in `ROLCH` implemenent these as class methods each. Currently, we have implemented the identity-link, log-link and  shifted log-link functions.

## Shifted Link Functions

Some link functions implement _shifted_ versions. The shifted link function is implemented in the sense that the shift is added to the inverse transformation. This way, we can ensure that distribution parameters can be modelled on the continuous space of the $\eta = g(\theta)$, but in the inverse transform fullfill certain additional constraints. A common example is the $t$ distribution, where we can use a `LogShift2Link` or `SqrtShift2Link` to ensure that $\hat{\theta} = g^-1(\eta) > 2$ and the variance exists.

## Base Class

::: rolch.base.LinkFunction

## API Reference

::: rolch.link.IdentityLink

### Log-Link Functions

::: rolch.link.LogLink

::: rolch.link.LogShiftValueLink

::: rolch.link.LogShiftTwoLink

::: rolch.link.LogIdentLink

### Square Root Link Functions

::: rolch.link.SqrtLink

::: rolch.link.SqrtShiftValueLink

::: rolch.link.SqrtShiftTwoLink