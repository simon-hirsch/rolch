from typing import Dict, Literal, Optional, Tuple, Union

import numpy as np

from rolch.coordinate_descent import (
    DEFAULT_ESTIMATOR_KWARGS,
    online_coordinate_descent_path,
)
from rolch.gram import init_gram, init_y_gram, update_gram, update_y_gram
from rolch.information_criteria import select_best_model_by_information_criterion
from rolch.scaler import OnlineScaler
from rolch.utils import calculate_effective_training_length


class OnlineLasso:

    "Simple Online Lasso regression for the expected value."

    def __init__(
        self,
        forget: float = 0,
        ic: Literal["aic", "bic", "hqc", "max"] = "bic",
        scale_inputs: bool = True,
        intercept_in_design: bool = True,
        lambda_n: int = 100,
        lambda_eps: float = 1e-4,
        start_value: str = "previous_fit",
        tolerance: float = 1e-4,
        max_iterations: int = 1000,
        selection: Literal["cyclic", "random"] = "cyclic",
    ):
        """Online LASSO estimator class.

        This class initializes the online linear regression fitted using LASSO. The estimator object provides three main methods,
        ``estimator.fit(X, y)``, ``estimator.update(X, y)`` and ``estimator.predict(X)``.

        Args:
            forget (float, optional): Exponential discounting of old observations. Defaults to 0.
            ic (Literal["aic", "bic", "hqc", "max"], optional): The information criteria for model selection. Defaults to "bic".
            scale_inputs (bool, optional): Whether to scale the $X$ matrix. Defaults to True.
            intercept_in_design (bool, optional): Whether the first column of $X$ corresponds to the intercept. In this case, the first beta will not be regularized. Defaults to True.
            lambda_n (int, optional): Length of the regularization path. Defaults to 100.
            lambda_eps (float, optional): The largest regularization is determined automatically such that the solution is fully regularized. The smallest regularization is taken as $\\varepsilon  \\lambda^\max$ and we will use an exponential grid. Defaults to 1e-4.
            start_value (str, optional): Whether to choose the previous fit or the previous regularization as start value. Defaults to 100.
            tolerance (float, optional): Tolerance for breaking the CD. Defaults to 1e-4.
            max_iterations (int, optional): Max number of CD iterations. Defaults to 1000.
            selection (Literal["cyclic", "random"], optional): Whether to cycle through all coordinates in order or random. For large problems, random might increase convergence. Defaults to 100.
        """

        self.ic = ic
        self.forget = forget
        self.lambda_n = lambda_n
        self.lambda_eps = lambda_eps

        self.start_value = start_value
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.selection = selection

        self.intercept_in_design = intercept_in_design
        self.scaler = OnlineScaler(
            forget=forget, do_scale=scale_inputs, intercept=intercept_in_design
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        beta_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> None:
        """Initial fit of the online LASSO.

        Args:
            X (np.ndarray): The design matrix $X$.
            y (np.ndarray): The response vector $y$.
            sample_weight (Optional[np.ndarray], optional): The sample weights. Defaults to None.
            beta_bounds (Optional[Tuple[np.ndarray, np.ndarray]], optional): Lower and upper bounds on the coefficient vector. `None` defaults to unconstrained coefficients.. Defaults to None.
        """
        self.N = X.shape[0]
        self.J = X.shape[1]

        if sample_weight is None:
            sample_weight = np.ones_like(y)

        if beta_bounds is None:
            self.beta_bounds = (np.repeat(-np.inf, self.J), np.repeat(np.inf, self.J))
        else:
            self.beta_bounds = beta_bounds

        self.training_length = calculate_effective_training_length(self.forget, self.N)

        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        self.beta_path = np.zeros((self.lambda_n,))
        self.x_gram = init_gram(X_scaled, sample_weight, self.forget)
        self.y_gram = init_y_gram(X_scaled, y, sample_weight, self.forget)

        self.beta_path = np.zeros((self.lambda_n, self.J))
        self.is_regularized = np.repeat(True, self.J)
        self.is_regularized[0] = self.intercept_in_design == False

        intercept = (
            self.y_gram[~self.is_regularized]
            / np.diag(self.x_gram)[~self.is_regularized]
        )

        self.lambda_max = np.max(
            np.abs(self.y_gram.flatten() - self.x_gram[0] * intercept)
        )
        self.lambda_path = np.geomspace(
            self.lambda_max, self.lambda_max * self.lambda_eps, self.lambda_n
        )
        self.beta_path = online_coordinate_descent_path(
            x_gram=self.x_gram,
            y_gram=self.y_gram.flatten(),
            beta_path=self.beta_path,
            lambda_path=self.lambda_path,
            is_regularized=self.is_regularized,
            beta_lower_bound=self.beta_bounds[0],
            beta_upper_bound=self.beta_bounds[1],
            which_start_value="previous_lambda",
            selection=self.selection,
            tolerance=self.tolerance,
            max_iterations=self.max_iterations,
        )[0]
        residuals = np.expand_dims(y, -1) - X_scaled @ self.beta_path.T
        self.rss = np.sum(residuals**2, axis=0)
        n_params = np.sum(~np.isclose(self.beta_path, 0), axis=1)
        best_ic = select_best_model_by_information_criterion(
            self.training_length, n_params, self.rss, self.ic
        )
        self.beta = self.beta_path[best_ic, :]

    def update(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> None:
        """Update the regression model.

        Args:
            X (np.ndarray): The new row of the design matrix $X$. Needs to be of shape 1 x J.
            y (np.ndarray): The new observation of $y$.
            sample_weight (Optional[np.ndarray], optional): The weight for the new observations. `None` implies all observations have weight 1. Defaults to None.
        """

        self.N += X.shape[0]
        self.training_length = calculate_effective_training_length(self.forget, self.N)
        self.scaler.partial_fit(X)
        X_scaled = self.scaler.transform(X)

        if sample_weight is None:
            sample_weight = np.ones(y.shape[0])

        self.x_gram = update_gram(
            self.x_gram, X_scaled, forget=self.forget, w=sample_weight
        )
        self.y_gram = update_y_gram(
            self.y_gram, X_scaled, y, forget=self.forget, w=sample_weight
        )

        intercept = (
            self.y_gram[~self.is_regularized]
            / np.diag(self.x_gram)[~self.is_regularized]
        )

        self.lambda_max = np.max(
            np.abs(self.y_gram.flatten() - self.x_gram[0] * intercept)
        )
        self.lambda_path = np.geomspace(
            self.lambda_max, self.lambda_max * self.lambda_eps, self.lambda_n
        )
        self.beta_path = online_coordinate_descent_path(
            x_gram=self.x_gram,
            y_gram=self.y_gram.flatten(),
            beta_path=self.beta_path,
            lambda_path=self.lambda_path,
            is_regularized=self.is_regularized,
            beta_lower_bound=self.beta_bounds[0],
            beta_upper_bound=self.beta_bounds[1],
            which_start_value=self.start_value,
            selection=self.selection,
            tolerance=self.tolerance,
            max_iterations=self.max_iterations,
        )[0]

        residuals = np.expand_dims(y, -1) - X_scaled @ self.beta_path.T
        self.rss = (1 - self.forget) * self.rss + np.sum(residuals**2, axis=0)
        n_params = np.sum(~np.isclose(self.beta_path, 0), axis=1)
        best_ic = select_best_model_by_information_criterion(
            self.training_length, n_params, self.rss, self.ic
        )
        self.beta = self.beta_path[best_ic, :]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the optimal IC selection.

        Args:
            X (np.ndarray): The design matrix $X$.

        Returns:
            np.ndarray: The predictions for the optimal IC.
        """
        X_scaled = self.scaler.transform(X)
        prediction = X_scaled @ self.beta.T
        return prediction

    def predict_path(self, X: np.ndarray) -> np.ndarray:
        """Predict the full regularization path.

        Args:
            X (np.ndarray): The design matrix $X$.

        Returns:
            np.ndarray: The predictions for the full path.
        """
        X_scaled = self.scaler.transform(X)
        prediction = X_scaled @ self.beta_path.T
        return prediction
