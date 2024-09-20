from typing import Dict, Literal, Optional

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
        forget=0,
        ic: Literal["aic", "bic", "hqc", "max"] = "bic",
        scale_inputs: bool = True,
        intercept_in_design: bool = True,
        lambda_n: int = 100,
        lambda_eps: float = 1e-4,
        estimation_kwargs: Optional[Dict] = None,
    ):
        self.ic = ic
        self.forget = forget
        self.lambda_n = lambda_n
        self.lambda_eps = lambda_eps
        self.intercept_in_design = intercept_in_design
        self.scaler = OnlineScaler(
            forget=forget, do_scale=scale_inputs, intercept=intercept_in_design
        )
        self.intercept = True

        for i, attribute in DEFAULT_ESTIMATOR_KWARGS.items():
            if (estimation_kwargs is not None) and (i in estimation_kwargs.keys()):
                setattr(self, i, estimation_kwargs[i])
            elif i == "lambda_eps":
                setattr(self, i, estimation_kwargs[i][0])
            else:
                setattr(self, i, attribute)

    def fit(self, y, X, sample_weight=None, beta_bounds=None):
        """Fit the regression model."""
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

    def update(self, y, X, sample_weight=None):
        """Update the regression model"""
        self.N += X.shape[0]
        self.training_length = calculate_effective_training_length(self.forget, self.N)
        self.scaler.partial_fit(X)
        X_scaled = self.scaler.transform(X)

        self.x_gram = update_gram(self.x_gram, X_scaled, self.forget, sample_weight)
        self.y_gram = update_y_gram(
            self.y_gram, X_scaled, y, self.forget, sample_weight
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

    def predict(self, X):
        """Predict using the optimal IC selection."""
        X_scaled = self.scaler.transform(X)
        prediction = X_scaled @ self.beta.T
        return prediction

    def predict_path(self, X):
        """Predict the full LASSO path."""
        X_scaled = self.scaler.transform(X)
        prediction = X_scaled @ self.beta_path.T
        return prediction
