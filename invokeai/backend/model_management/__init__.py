"""
Initialization file for invokeai.backend.model_management
"""
# This import must be first
from .lora import ModelPatcher, ONNXModelPatcher  # noqa: F401
from .model_cache import ModelCache  # noqa: F401
from .model_manager import AddModelResult, ModelInfo, ModelManager, SchedulerPredictionType  # noqa: F401 isort: split

# This import must be last
from .model_merge import MergeInterpolationMethod, ModelMerger  # noqa: F401 isort: split
from .models import (  # noqa: F401
    BaseModelType,
    DuplicateModelException,
    ModelNotFoundException,
    ModelType,
    ModelVariantType,
    SubModelType,
)
