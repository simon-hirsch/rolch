from importlib.metadata import version

from rolch.coordinate_descent import (
    online_coordinate_descent,
    online_coordinate_descent_path,
    soft_threshold,
)
from rolch.distributions import DistributionJSU, DistributionNormal, DistributionT
from rolch.gram import (
    init_forget_vector,
    init_gram,
    init_inverted_gram,
    init_y_gram,
    update_gram,
    update_inverted_gram,
    update_y_gram,
)
from rolch.information_criteria import (
    information_criterion,
    select_best_model_by_information_criterion,
)
from rolch.link import IdentityLink, LogLink, LogShiftTwoLink, LogShiftValueLink
from rolch.online_gamlss import OnlineGamlss
from rolch.online_lasso import OnlineLasso
from rolch.scaler import OnlineScaler
from rolch.utils import (
    calculate_asymptotic_training_length,
    calculate_effective_training_length,
)

__version__ = version("rolch")

__all__ = [
    OnlineScaler,
    OnlineGamlss,
    OnlineLasso,
    IdentityLink,
    LogLink,
    LogShiftTwoLink,
    LogShiftValueLink,
    DistributionNormal,
    DistributionT,
    DistributionJSU,
    init_forget_vector,
    init_gram,
    update_gram,
    init_inverted_gram,
    update_inverted_gram,
    init_y_gram,
    update_y_gram,
    online_coordinate_descent,
    online_coordinate_descent_path,
    soft_threshold,
    information_criterion,
    select_best_model_by_information_criterion,
    calculate_asymptotic_training_length,
    calculate_effective_training_length,
]
