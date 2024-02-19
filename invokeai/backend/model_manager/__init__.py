"""Re-export frequently-used symbols from the Model Manager backend."""
from .config import (
    AnyModel,
    AnyModelConfig,
    BaseModelType,
    InvalidModelConfigException,
    ModelConfigFactory,
    ModelFormat,
    ModelRepoVariant,
    ModelType,
    ModelVariantType,
    SchedulerPredictionType,
    SubModelType,
)
from .load import LoadedModel
from .probe import ModelProbe
from .search import ModelSearch

__all__ = [
    "AnyModel",
    "AnyModelConfig",
    "BaseModelType",
    "ModelRepoVariant",
    "InvalidModelConfigException",
    "LoadedModel",
    "ModelConfigFactory",
    "ModelFormat",
    "ModelProbe",
    "ModelSearch",
    "ModelType",
    "ModelVariantType",
    "SchedulerPredictionType",
    "SubModelType",
]

########## to help populate the openapi_schema with format enums for each config ###########
# This code is no longer necessary?
# leave it here just in case
#
# import inspect
# from enum import Enum
# from typing import Any, Iterable, Dict, get_args, Set
# def _expand(something: Any) -> Iterable[type]:
#     if isinstance(something, type):
#         yield something
#     else:
#         for x in get_args(something):
#             for y in _expand(x):
#                 yield y

# def _find_format(cls: type) -> Iterable[Enum]:
#     if hasattr(inspect, "get_annotations"):
#         fields = inspect.get_annotations(cls)
#     else:
#         fields = cls.__annotations__
#     if "format" in fields:
#         for x in get_args(fields["format"]):
#             yield x
#     for parent_class in cls.__bases__:
#         for x in _find_format(parent_class):
#             yield x
#     return None

# def get_model_config_formats() -> Dict[str, Set[Enum]]:
#     result: Dict[str, Set[Enum]] = {}
#     for model_config in _expand(AnyModelConfig):
#         for field in _find_format(model_config):
#             if field is None:
#                 continue
#             if not result.get(model_config.__qualname__):
#                 result[model_config.__qualname__] = set()
#             result[model_config.__qualname__].add(field)
#     return result
