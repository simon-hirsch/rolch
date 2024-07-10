# Link functions 

## Link functions 

A link function \(g(x)\) is a smooth, monotonic function of \(x\). 

For all link functions, we implement 

- the link \(g(x)\)
- the inverse \(g^{-1}(x)\)
- the first derivative _of the inverse_ of the link function \(\frac{\partial g(x)^{-1}}{\partial x}\). The choice of the inverse is justified by Equation (7) in Hirsch, Berrisch & Ziel ([2024](https://github.com/simon-hirsch/rolch/blob/main/paper.pdf)). 

The link functions implemented in `ROLCH` implemenent these as class methods each. Currently, we have implemented the identity-link, log-link and  shifted log-link functions.

## Base Class

::: rolch.abc.LinkFunction

## API Reference

::: rolch.link.LogLink

::: rolch.link.IdentityLink

::: rolch.link.LogShiftValueLink

::: rolch.link.LogShiftTwoLink
