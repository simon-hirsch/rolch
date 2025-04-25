import warnings
from typing import Literal, Optional

import numpy as np

from ..base import EstimationMethod, Estimator
from ..design_matrix import add_intercept
from ..information_criteria import InformationCriterion
from ..methods import get_estimation_method
from ..scaler import OnlineScaler
from ..utils import calculate_effective_training_length


class OnlineLinearModel(Estimator):
    "Simple Online Linear Regression for the expected value."

    def __init__(
        self,
        forget: float = 0,
        scale_inputs: bool | np.ndarray = True,
        fit_intercept: bool = True,
        regularize_intercept: bool = False,
        method: EstimationMethod | str = "ols",
        ic: Literal["aic", "bic", "hqc", "max"] = "bic",
    ):
        """The basic linear model for many different estimation techniques.

        Args:
            forget (float, optional): Exponential discounting of old observations. Defaults to 0.
            scale_inputs (bool, optional): Whether to scale the $X$ matrix. Defaults to True.
            fit_intercept (bool, optional): Whether to add an intercept in the estimation. Defaults to True.
            regularize_intercept (bool, optional): Whether to regularize the intercept. Defaults to False.
            method (EstimationMethod | str, optional): The estimation method. Can be a string or `EstimationMethod` class. Defaults to "ols".
            ic (Literal["aic", "bic", "hqc", "max"], optional): The information criteria for model selection. Defaults to "bic".
        Raises:
            ValueError: Will raise if you try to regularize the intercept, but not fit it.
        """

        self.forget = forget
        self.method = method
        self._method = get_estimation_method(self.method)
        self.fit_intercept = fit_intercept
        self.scale_inputs = scale_inputs
        self.scaler = OnlineScaler(forget=forget, to_scale=scale_inputs)

        self.regularize_intercept = regularize_intercept
        self.ic = ic

        if not self._method._path_based_method and self.regularize_intercept:
            warnings.warn(
                "Note that you have passed a non-regularized estimation method but want to regularize the intercept. Are you sure?"
            )
        if self.regularize_intercept and not self.fit_intercept:
            raise ValueError(
                "You want to regularize the intercept but not fit an intercept."
            )

    def get_design_matrix(self, X: np.ndarray):
        if self.fit_intercept:
            design = add_intercept(X)
        else:
            design = np.copy(X)
        return design

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> None:
        """Initial fit of the online regression model.

        Args:
            X (np.ndarray): The design matrix $X$.
            y (np.ndarray): The response vector $y$.
            sample_weight (Optional[np.ndarray], optional): The sample weights. Defaults to None.
        """
        self.n_observations = X.shape[0]
        self.J = X.shape[1] + int(self.fit_intercept)

        if sample_weight is None:
            sample_weight = np.ones_like(y)

        self.n_training = calculate_effective_training_length(
            self.forget, self.n_observations
        )

        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        X_scaled = self.get_design_matrix(X=X_scaled)

        self.is_regularized = np.repeat(True, self.J)
        if self.fit_intercept:
            self.is_regularized[0] = self.regularize_intercept

        self.x_gram = self._method.init_x_gram(
            X_scaled, weights=sample_weight, forget=self.forget
        )
        self.y_gram = self._method.init_y_gram(
            X_scaled, y, weights=sample_weight, forget=self.forget
        )

        if self._method._path_based_method:
            self.beta_path = self._method.fit_beta_path(
                x_gram=self.x_gram,
                y_gram=self.y_gram,
                is_regularized=self.is_regularized,
            )
            residuals = np.expand_dims(y, -1) - X_scaled @ self.beta_path.T
            self.rss = np.sum(residuals**2, axis=0)
            n_params = np.sum(~np.isclose(self.beta_path, 0), axis=1)
            ic = InformationCriterion(
                n_observations=self.n_training,
                n_parameters=n_params,
                criterion=self.ic,
            ).from_rss(rss=self.rss)
            best_ic = np.argmin(ic)
            self.beta = self.beta_path[best_ic, :]
        else:
            self.beta = self._method.fit_beta(
                self.x_gram, self.y_gram, self.is_regularized
            )

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

        self.n_observations += X.shape[0]
        self.n_training = calculate_effective_training_length(
            self.forget, self.n_observations
        )

        self.scaler.partial_fit(X)
        X_scaled = self.scaler.transform(X)
        X_scaled = self.get_design_matrix(X=X_scaled)

        if sample_weight is None:
            sample_weight = np.ones(y.shape[0])

        self.x_gram = self._method.update_x_gram(
            self.x_gram, X_scaled, forget=self.forget, weights=sample_weight
        )
        self.y_gram = self._method.update_y_gram(
            self.y_gram, X_scaled, y, forget=self.forget, weights=sample_weight
        )
        if self._method._path_based_method:
            self.beta_path = self._method.update_beta_path(
                x_gram=self.x_gram,
                y_gram=self.y_gram,
                beta_path=self.beta_path,
                is_regularized=self.is_regularized,
            )

            residuals = np.expand_dims(y, -1) - X_scaled @ self.beta_path.T
            self.rss = (1 - self.forget) * self.rss + np.sum(residuals**2, axis=0)
            n_params = np.sum(~np.isclose(self.beta_path, 0), axis=1)
            ic = InformationCriterion(
                n_observations=self.n_training,
                n_parameters=n_params,
                criterion=self.ic,
            ).from_rss(rss=self.rss)
            best_ic = np.argmin(ic)
            self.beta = self.beta_path[best_ic, :]
        else:
            self.beta = self._method.update_beta(
                x_gram=self.x_gram,
                y_gram=self.y_gram,
                beta=self.beta,
                is_regularized=self.is_regularized,
            )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the optimal IC selection.

        Args:
            X (np.ndarray): The design matrix $X$.

        Returns:
            np.ndarray: The predictions for the optimal IC.
        """
        design = self.scaler.transform(X)
        design = self.get_design_matrix(X=design)
        prediction = design @ self.beta.T
        return prediction

    def predict_path(self, X: np.ndarray) -> np.ndarray:
        """Predict the full regularization path.

        Args:
            X (np.ndarray): The design matrix $X$.

        Returns:
            np.ndarray: The predictions for the full path.
        """

        design = self.scaler.transform(X)
        design = self.get_design_matrix(X=design)
        prediction = design @ self.beta_path.T
        return prediction
        return prediction
