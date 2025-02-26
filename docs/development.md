# Development

First of all, we're happy that you want to take the time to contribute to this package and extending it. Thanks!

This page is trying to give some guidance on how to write additional link functions or distributions and some of the ideas underlying the [`LinkFunction`][rolch.base.LinkFunction], the [`Distribution`][rolch.base.Distribution] and the `ScipyMixin` classes.

## Writing Link Functions

Link functions map the predictors of the distribution parameter to the distribution parameter's support. Hence, we need to have the link function itself and its inverse. To calculate the score vector and the weights, we additionally need the first and second derivative of the link, as well as the derivative of the inverse.

The [`LinkFunction`][rolch.base.LinkFunction] abstract base class enforces, that `SomeNewLink()` implements:

- The link: `SomeNewLink().link()`.
- The inverse: `SomeNewLink().inverse()`.
- The first derivative of the link: `SomeNewLink().link_derivative()`.
- The second derivative of the link: `SomeNewLink().link_second_derivative()`.
- The derivative of the inverse: `SomeNewLink().inverse_derivative()`.

Additionally, each link defines the `SomeNewLink().link_support` as `tuple` of `float` values. This is used to ensure at the initialization of a distribution that the link functions support is inside the support of the distribution parameter. We allow the link to shorten the possible outcome space of the parameter, but we don't allow to return impossible values. As an example, you can constrain the degrees of freedom $\nu$ of the $t$-distribution by taking a [`LogShiftTwoLink()`][rolch.link.LogShiftTwoLink], which ensures that $\nu > 2$ (and therefore the variance exists), but you cannot choose the [`IdentityLink()`][rolch.link.IdentityLink], since the degrees of freedom must be positive.

There are two ways to implement the support of the link function:

- As a class attribute, i.e. before the `__init__()`. This is usually done if parameters passed to the link function *do not* affect the support. We write something like

    ```python
    class SomeNewLink(LinkFunction):
        link_support = (0, np.inf)

        def __init__(self)
            pass # or do something

        def all_the_necessary_methods():
            # implement links and derivative
    ```

- As property to the class. This is usually done if the link functions parameter affect its support. Then we do this:

    ```python
    class SomeNewLink(LinkFunction):

        def __init__(self, lower_support_bound):
            self.lower_support_bound = lower_support_bound

        @property
        def link_support(self):
            return (self.lower_support_bound, np.inf)

        def all_the_necessary_methods():
            # implement links and derivative
    ```

To ensure that we model the support correctly - think about, e.g. $\log(x)$, we can use `np.nextafter(a, b)`, which gives the next possible float after `a` in the direction of `b`. Hence, `np.nextafter(0, 1)` will return a very small positive float, which describes the support of $\log(x)$ well.
