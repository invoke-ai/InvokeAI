"""
This file re-exports all the public API for invocations. This is the only file that should be imported by custom nodes.

TODO(psyche): Do we want to dogfood this?
"""

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    Classification,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.constants import SCHEDULER_NAME_VALUES
from invokeai.app.invocations.fields import (
    BoardField,
    ColorField,
    ConditioningField,
    DenoiseMaskField,
    FieldDescriptions,
    FieldKind,
    ImageField,
    Input,
    InputField,
    LatentsField,
    MetadataField,
    OutputField,
    UIComponent,
    UIType,
    WithMetadata,
    WithWorkflow,
)
from invokeai.app.invocations.latent import SchedulerOutput
from invokeai.app.invocations.metadata import MetadataItemField, MetadataItemOutput, MetadataOutput
from invokeai.app.invocations.model import (
    ClipField,
    CLIPOutput,
    LoraInfo,
    LoraLoaderOutput,
    LoRAModelField,
    MainModelField,
    ModelInfo,
    ModelLoaderOutput,
    SDXLLoraLoaderOutput,
    UNetField,
    UNetOutput,
    VaeField,
    VAEModelField,
    VAEOutput,
)
from invokeai.app.invocations.primitives import (
    BooleanCollectionOutput,
    BooleanOutput,
    ColorCollectionOutput,
    ColorOutput,
    ConditioningCollectionOutput,
    ConditioningOutput,
    DenoiseMaskOutput,
    FloatCollectionOutput,
    FloatOutput,
    ImageCollectionOutput,
    ImageOutput,
    IntegerCollectionOutput,
    IntegerOutput,
    LatentsCollectionOutput,
    LatentsOutput,
    StringCollectionOutput,
    StringOutput,
)
from invokeai.app.services.boards.boards_common import BoardDTO
from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.image_records.image_records_common import ImageCategory
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.services.workflow_records.workflow_records_common import WorkflowWithoutID
from invokeai.app.util.misc import SEED_MAX, get_random_seed
from invokeai.backend.model_management.model_manager import LoadedModelInfo
from invokeai.backend.model_management.models.base import BaseModelType, ModelType, SubModelType
from invokeai.backend.stable_diffusion.diffusers_pipeline import PipelineIntermediateState
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import (
    BasicConditioningInfo,
    ConditioningFieldData,
    ExtraConditioningInfo,
    SDXLConditioningInfo,
)
from invokeai.backend.util.devices import CPU_DEVICE, CUDA_DEVICE, MPS_DEVICE, choose_precision, choose_torch_device
from invokeai.version import __version__

__all__ = [
    # invokeai.app.invocations.baseinvocation
    "BaseInvocation",
    "BaseInvocationOutput",
    "Classification",
    "invocation",
    "invocation_output",
    # invokeai.app.services.shared.invocation_context
    "InvocationContext",
    # invokeai.app.invocations.fields
    "BoardField",
    "ColorField",
    "ConditioningField",
    "DenoiseMaskField",
    "FieldDescriptions",
    "FieldKind",
    "ImageField",
    "Input",
    "InputField",
    "LatentsField",
    "MetadataField",
    "OutputField",
    "UIComponent",
    "UIType",
    "WithMetadata",
    "WithWorkflow",
    # invokeai.app.invocations.latent
    "SchedulerOutput",
    # invokeai.app.invocations.metadata
    "MetadataItemField",
    "MetadataItemOutput",
    "MetadataOutput",
    # invokeai.app.invocations.model
    "ModelInfo",
    "LoraInfo",
    "UNetField",
    "ClipField",
    "VaeField",
    "MainModelField",
    "LoRAModelField",
    "VAEModelField",
    "UNetOutput",
    "VAEOutput",
    "CLIPOutput",
    "ModelLoaderOutput",
    "LoraLoaderOutput",
    "SDXLLoraLoaderOutput",
    # invokeai.app.invocations.primitives
    "BooleanCollectionOutput",
    "BooleanOutput",
    "ColorCollectionOutput",
    "ColorOutput",
    "ConditioningCollectionOutput",
    "ConditioningOutput",
    "DenoiseMaskOutput",
    "FloatCollectionOutput",
    "FloatOutput",
    "ImageCollectionOutput",
    "ImageOutput",
    "IntegerCollectionOutput",
    "IntegerOutput",
    "LatentsCollectionOutput",
    "LatentsOutput",
    "StringCollectionOutput",
    "StringOutput",
    # invokeai.app.services.image_records.image_records_common
    "ImageCategory",
    # invokeai.app.services.boards.boards_common
    "BoardDTO",
    # invokeai.backend.stable_diffusion.diffusion.conditioning_data
    "BasicConditioningInfo",
    "ConditioningFieldData",
    "ExtraConditioningInfo",
    "SDXLConditioningInfo",
    # invokeai.backend.stable_diffusion.diffusers_pipeline
    "PipelineIntermediateState",
    # invokeai.app.services.workflow_records.workflow_records_common
    "WorkflowWithoutID",
    # invokeai.app.services.config.config_default
    "InvokeAIAppConfig",
    # invokeai.backend.model_management.model_manager
    "LoadedModelInfo",
    # invokeai.backend.model_management.models.base
    "BaseModelType",
    "ModelType",
    "SubModelType",
    # invokeai.app.invocations.constants
    "SCHEDULER_NAME_VALUES",
    # invokeai.version
    "__version__",
    # invokeai.backend.util.devices
    "choose_precision",
    "choose_torch_device",
    "CPU_DEVICE",
    "CUDA_DEVICE",
    "MPS_DEVICE",
    # invokeai.app.util.misc
    "SEED_MAX",
    "get_random_seed",
]
