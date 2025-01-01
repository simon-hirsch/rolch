from typing import Literal

import numpy as np

from rolch.base import EstimationMethod
from rolch.coordinate_descent import online_coordinate_descent_path
from rolch.gram import init_gram, init_y_gram, update_gram, update_y_gram


class LassoPathMethod(EstimationMethod):
    def __init__(
        self,
        lambda_n: int = 100,
        lambda_eps: float = 1e-4,
        start_value_initial: Literal[
            "previous_lambda", "previous_fit", "average"
        ] = "previous_lambda",
        start_value_update: Literal[
            "previous_lambda", "previous_fit", "average"
        ] = "previous_fit",
        selection: Literal["cyclic", "random"] = "cyclic",
        beta_lower_bound: np.ndarray | None = None,
        beta_upper_bound: np.ndarray | None = None,
        tolerance: float = 1e-4,
        max_iterations: int = 1000,
    ):
        super().__init__(
            _path_based_method=True,
            _accepts_bounds=True,
            _accepts_selection=True,
        )
        self.beta_lower_bound = beta_lower_bound
        self.beta_upper_bound = beta_upper_bound
        self.lambda_n = lambda_n
        self.lambda_eps = lambda_eps
        self.start_value_initial = start_value_initial
        self.start_value_update = start_value_update
        self.selection = selection
        self.tolerance = tolerance
        self.max_iterations = max_iterations

        self._path_length = self.lambda_n

    @staticmethod
    def _get_lambda_max(x_gram, y_gram, is_regularized):
        if np.all(is_regularized):
            lambda_max = np.max(y_gram)
        elif np.sum(~is_regularized) == 1:
            intercept = y_gram[~is_regularized] / np.diag(x_gram)[~is_regularized]
            lambda_max = np.max(
                np.abs(y_gram.flatten() - x_gram[~is_regularized, :] * intercept)
            )
        else:
            raise NotImplementedError(
                "More than one not regularized value is currently not supported."
            )
        return lambda_max

    def _set_and_validate_bounds(self, x_gram: np.ndarray) -> None:
        J = x_gram.shape[1]
        if self.beta_lower_bound is None:
            self.beta_lower_bound = np.repeat(-np.inf, J)
        if self.beta_upper_bound is None:
            self.beta_upper_bound = np.repeat(np.inf, J)

        if len(self.beta_lower_bound) != J:
            raise ValueError("Lower bound does not have correct length")
        if len(self.beta_upper_bound) != J:
            raise ValueError("Upper bound does not have correct length")

    @staticmethod
    def init_x_gram(X, weights, forget):
        return init_gram(X=X, w=weights, forget=forget)

    @staticmethod
    def init_y_gram(X, y, weights, forget):
        return init_y_gram(X, y, w=weights, forget=forget)

    @staticmethod
    def update_x_gram(gram, X, weights, forget):
        return update_gram(gram, X, w=weights, forget=forget)

    @staticmethod
    def update_y_gram(gram, X, y, weights, forget):
        return update_y_gram(gram, X, y, forget=forget, w=weights)

    def fit_beta_path(self, x_gram, y_gram, is_regularized):
        self._set_and_validate_bounds(x_gram=x_gram)
        lambda_max = self._get_lambda_max(
            x_gram=x_gram, y_gram=y_gram, is_regularized=is_regularized
        )
        lambda_path = np.geomspace(
            lambda_max, lambda_max * self.lambda_eps, self.lambda_n
        )
        beta_path = np.zeros((self.lambda_n, x_gram.shape[0]))
        beta_path, _ = online_coordinate_descent_path(
            x_gram=x_gram,
            y_gram=y_gram.squeeze(-1),
            beta_path=beta_path,
            lambda_path=lambda_path,
            is_regularized=is_regularized,
            beta_lower_bound=self.beta_lower_bound,
            beta_upper_bound=self.beta_upper_bound,
            which_start_value=self.start_value_initial,
            selection=self.selection,
            tolerance=self.tolerance,
            max_iterations=self.max_iterations,
        )
        return beta_path

    def update_beta_path(self, x_gram, y_gram, beta_path, is_regularized):
        lambda_max = self._get_lambda_max(
            x_gram=x_gram, y_gram=y_gram, is_regularized=is_regularized
        )
        lambda_path = np.geomspace(
            lambda_max, lambda_max * self.lambda_eps, self.lambda_n
        )
        beta_path, _ = online_coordinate_descent_path(
            x_gram=x_gram,
            y_gram=y_gram.squeeze(-1),
            beta_path=beta_path,
            lambda_path=lambda_path,
            is_regularized=is_regularized,
            beta_lower_bound=self.beta_lower_bound,
            beta_upper_bound=self.beta_upper_bound,
            which_start_value=self.start_value_update,
            selection=self.selection,
            tolerance=self.tolerance,
            max_iterations=self.max_iterations,
        )
        return beta_path

    def fit_beta(self, x_gram, y_gram, is_regularized):
        pass

    def update_beta(self, x_gram, y_gram, is_regularized):
        pass
