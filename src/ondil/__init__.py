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
    DistributionBeta,
    DistributionBetaInflated,
    DistributionExponential,
    DistributionGamma,
    DistributionGumbel,
    DistributionInverseGaussian,
    DistributionJSU,
    DistributionLogistic,
    DistributionLogNormal,
    DistributionLogNormalMedian,
    DistributionNormal,
    DistributionNormalMeanVariance,
    DistributionT,
    DistributionReverseGumbel,
    DistributionInverseGamma,
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
    LogitLink,
    LogLink,
    LogShiftTwoLink,
    LogShiftValueLink,
    SqrtLink,
    SqrtShiftTwoLink,
    SqrtShiftValueLink,
)
from .methods import (
    ElasticNetPathMethod,
    LassoPathMethod,
    OrdinaryLeastSquaresMethod,
    RidgeMethod,
)
from .scaler import OnlineScaler
from .utils import (
    calculate_asymptotic_training_length,
    calculate_effective_training_length,
)
from .warnings import OutOfSupportWarning

__version__ = version("ondil")

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
    "LogitLink",
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
    "DistributionLogistic",
    "DistributionLogNormalMedian",
    "DistributionT",
    "DistributionJSU",
    "DistributionGamma",
    "DistributionBeta",
    "DistributionBetaInflated",
    "DistributionLogNormal",
    "DistributionExponential",
    "DistributionGumbel",
    "DistributionInverseGaussian",
    "DistributionInverseGamma",
    "DistributionReverseGumbel",
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
