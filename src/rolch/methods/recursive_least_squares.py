from rolch.base import EstimationMethod
from rolch.gram import (
    init_inverted_gram,
    init_y_gram,
    update_inverted_gram,
    update_y_gram,
)


class OrdinaryLeastSquaresMethod(EstimationMethod):

    def __init__(self):
        super().__init__(
            _path_based_method=False,
            _accepts_bounds=False,
            _accepts_selection=False,
        )

    @staticmethod
    def init_x_gram(X, weights, forget):
        return init_inverted_gram(X, w=weights, forget=forget)

    @staticmethod
    def init_y_gram(X, y, weights, forget):
        return init_y_gram(X, y, w=weights, forget=forget)

    @staticmethod
    def update_x_gram(gram, X, weights, forget):
        return update_inverted_gram(gram, X, w=weights, forget=forget)

    @staticmethod
    def update_y_gram(gram, X, y, weights, forget):
        return update_y_gram(gram, X, y, forget=forget, w=weights)

    def fit_beta_path(self, x_gram, y_gram, is_regularized):
        return super().fit_beta_path(x_gram, y_gram, is_regularized)

    def update_beta_path(self, x_gram, y_gram, beta_path, is_regularized):
        return super().update_beta_path(x_gram, y_gram, beta_path, is_regularized)

    def fit_beta(self, x_gram, y_gram, is_regularized=None):
        return (x_gram @ y_gram).squeeze()

    def update_beta(self, x_gram, y_gram, beta, is_regularized=None):
        return (x_gram @ y_gram).squeeze()
