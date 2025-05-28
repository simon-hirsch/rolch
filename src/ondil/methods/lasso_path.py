from .elasticnet import ElasticNetPathMethod


class LassoPathMethod(ElasticNetPathMethod):
    """
    Path-based lasso estimation.

    The lasso method runs coordinate descent along a (geometric) decreasing grid of regularization strengths (lambdas).
    We automatically calculate the maximum regularization strength for which all (not-regularized) coefficients are 0.
    The lower end of the lambda grid is defined as $$\\lambda_\min = \\lambda_\max * \\varepsilon_\\lambda.$$

    We allow to pass user-defined lower and upper bounds for the coefficients.
    The coefficient bounds must be an `numpy` array of the length of `X` respectively of the number of variables in the
    equation _plus the intercept, if you fit one_. This allows to box-constrain the coefficients to a certain range.

    Furthermore, we allow to choose the start value, i.e. whether you want an update to be warm-started on the previous fit's path
    or on the previous reguarlization strength or an average of both. If your data generating process is rather stable,
    the `"previous_fit"` should give considerable speed gains, since warm starting on the previous strength is effectively batch-fitting.

    Lastly, we have some rather technical parameters like the number of coordinate descent iterations,
    whether you want to cycle randomly and for which tolerance you want to break. We use active set iterations, i.e.
    after the first coordinate-wise update for each regularization strength, only non-zero coefficients are updated.

    We use `numba` to speed up the coordinate descent algorithm.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the lasso method with the specified parameters.

        Parameters:
            lambda_n (int): Number of lambda values to use in the path. Default is 100.
            lambda_eps (float): Minimum lambda value as a fraction of the maximum lambda. Default is 1e-4.
            early_stop (int): Early stopping criterion. Will stop if the number of non-zero parameters is reached. Default is 0 (no early stopping).
            start_value_initial (Literal["previous_lambda", "previous_fit", "average"]): Method to initialize the start value for the first lambda. Default is "previous_lambda".
            start_value_update (Literal["previous_lambda", "previous_fit", "average"]): Method to update the start value for subsequent lambdas. Default is "previous_fit".
            selection (Literal["cyclic", "random"]): Method to select features during the path. Default is "cyclic".
            beta_lower_bound (np.ndarray | None): Lower bound for the coefficients. Default is None.
            beta_upper_bound (np.ndarray | None): Upper bound for the coefficients. Default is None.
            tolerance (float): Tolerance for the optimization. Default is 1e-4.
            max_iterations (int): Maximum number of iterations for the optimization. Default is 1000.
        """
        super().__init__(alpha=1.0, *args, **kwargs)
