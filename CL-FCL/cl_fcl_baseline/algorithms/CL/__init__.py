from .base import ContinualLearner
from .ewc import EWCLearner
from .gem import GEMLearner, project_gradient
from .iCaRL import ICaRLLearner
from .lwf import LwFLearner

__all__ = [
    "ContinualLearner",
    "EWCLearner",
    "GEMLearner",
    "project_gradient",
    "LwFLearner",
    "ICaRLLearner",
]
