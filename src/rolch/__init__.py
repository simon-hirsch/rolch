# ruff: noqa: E402

from importlib.metadata import version
from importlib.util import find_spec

HAS_PANDAS = False
HAS_POLARS = False

if find_spec("pandas") is not None:
    HAS_PANDAS = True

if find_spec("polars") is not None:
    HAS_POLARS = True

from .coordinate_descent import (
    online_coordinate_descent,
    online_coordinate_descent_path,
    soft_threshold,
)
from .distributions import (
    DistributionGamma,
    DistributionJSU,
    DistributionNormal,
    DistributionNormalMeanVariance,
    DistributionT,
    DistributionLogNormalMedian,
    DistributionLogNormal,
)
from .error import OutOfSupportError
from .estimators import OnlineGamlss, OnlineLasso, OnlineLinearModel
from .gram import (
    init_forget_vector,
    init_gram,
    init_inverted_gram,
    init_y_gram,
    update_gram,
    update_inverted_gram,
    update_y_gram,
)
from .information_criteria import InformationCriterion
from .link import (
    IdentityLink,
    InverseSoftPlusLink,
    InverseSoftPlusShiftTwoLink,
    InverseSoftPlusShiftValueLink,
    LogIdentLink,
    LogLink,
    LogShiftTwoLink,
    LogShiftValueLink,
    SqrtLink,
    SqrtShiftTwoLink,
    SqrtShiftValueLink,
)
from .methods import (
    LassoPathMethod,
    OrdinaryLeastSquaresMethod,
    RidgeMethod,
    ElasticNetPathMethod,
)
from .scaler import OnlineScaler
from .utils import (
    calculate_asymptotic_training_length,
    calculate_effective_training_length,
)
from .warnings import OutOfSupportWarning

__version__ = version("rolch")

__all__ = [
    "OutOfSupportWarning",
    "OutOfSupportError",
    "OnlineScaler",
    "OnlineGamlss",
    "OnlineLinearModel",
    "OnlineLasso",
    "LassoPathMethod",
    "RidgeMethod",
    "ElasticNetPathMethod",
    "OrdinaryLeastSquaresMethod",
    "IdentityLink",
    "LogLink",
    "LogIdentLink",
    "LogShiftTwoLink",
    "LogShiftValueLink",
    "SqrtLink",
    "SqrtShiftValueLink",
    "SqrtShiftTwoLink",
    "InverseSoftPlusLink",
    "InverseSoftPlusShiftValueLink",
    "InverseSoftPlusShiftTwoLink",
    "DistributionNormal",
    "DistributionNormalMeanVariance",
    "DistributionLogNormalMedian",
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
    "InformationCriterion",
    "calculate_asymptotic_training_length",
    "calculate_effective_training_length",
]
