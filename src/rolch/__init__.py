# ruff: noqa: E402

from importlib.metadata import version
from importlib.util import find_spec

HAS_PANDAS = False
HAS_POLARS = False

if find_spec("pandas") is not None:
    HAS_PANDAS = True

if find_spec("polars") is not None:
    HAS_POLARS = True

from rolch.coordinate_descent import (
    online_coordinate_descent,
    online_coordinate_descent_path,
    soft_threshold,
)
from rolch.distributions import (
    DistributionGamma,
    DistributionJSU,
    DistributionNormal,
    DistributionT,
)
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
from rolch.link import (
    IdentityLink,
    LogIdentLink,
    LogLink,
    LogShiftTwoLink,
    LogShiftValueLink,
    SqrtLink,
    SqrtShiftTwoLink,
    SqrtShiftValueLink,
)
from rolch.online_gamlss import OnlineGamlss
from rolch.online_lasso import OnlineLasso
from rolch.scaler import OnlineScaler
from rolch.utils import (
    calculate_asymptotic_training_length,
    calculate_effective_training_length,
)

__version__ = version("rolch")

__all__ = [
    "OnlineScaler",
    "OnlineGamlss",
    "OnlineLasso",
    "IdentityLink",
    "LogLink",
    "LogIdentLink",
    "LogShiftTwoLink",
    "LogShiftValueLink",
    "SqrtLink",
    "SqrtShiftValueLink",
    "SqrtShiftTwoLink",
    "DistributionNormal",
    "DistributionT",
    "DistributionJSU",
    "DistributionGamma",
    "init_forget_vector",
    "init_gram",
    "update_gram",
    "init_inverted_gram",
    "update_inverted_gram",
    "init_y_gram",
    "update_y_gram",
    "online_coordinate_descent",
    "online_coordinate_descent_path",
    "soft_threshold",
    "information_criterion",
    "select_best_model_by_information_criterion",
    "calculate_asymptotic_training_length",
    "calculate_effective_training_length",
]
