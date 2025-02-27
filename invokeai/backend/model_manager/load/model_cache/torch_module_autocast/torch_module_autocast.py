from typing import TypeVar

import torch

from invokeai.backend.flux.modules.layers import RMSNorm
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.custom_modules.custom_conv1d import (
    CustomConv1d,
)
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.custom_modules.custom_conv2d import (
    CustomConv2d,
)
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.custom_modules.custom_embedding import (
    CustomEmbedding,
)
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.custom_modules.custom_flux_rms_norm import (
    CustomFluxRMSNorm,
)
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.custom_modules.custom_group_norm import (
    CustomGroupNorm,
)
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.custom_modules.custom_linear import (
    CustomLinear,
)
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.custom_modules.custom_module_mixin import (
    CustomModuleMixin,
)

AUTOCAST_MODULE_TYPE_MAPPING: dict[type[torch.nn.Module], type[torch.nn.Module]] = {
    torch.nn.Linear: CustomLinear,
    torch.nn.Conv1d: CustomConv1d,
    torch.nn.Conv2d: CustomConv2d,
    torch.nn.GroupNorm: CustomGroupNorm,
    torch.nn.Embedding: CustomEmbedding,
    RMSNorm: CustomFluxRMSNorm,
}

try:
    # These dependencies are not expected to be present on MacOS.
    from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.custom_modules.custom_invoke_linear_8_bit_lt import (
        CustomInvokeLinear8bitLt,
    )
    from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.custom_modules.custom_invoke_linear_nf4 import (
        CustomInvokeLinearNF4,
    )
    from invokeai.backend.quantization.bnb_llm_int8 import InvokeLinear8bitLt
    from invokeai.backend.quantization.bnb_nf4 import InvokeLinearNF4

    AUTOCAST_MODULE_TYPE_MAPPING[InvokeLinear8bitLt] = CustomInvokeLinear8bitLt
    AUTOCAST_MODULE_TYPE_MAPPING[InvokeLinearNF4] = CustomInvokeLinearNF4
except ImportError:
    pass


AUTOCAST_MODULE_TYPE_MAPPING_INVERSE = {v: k for k, v in AUTOCAST_MODULE_TYPE_MAPPING.items()}


T = TypeVar("T", bound=torch.nn.Module)


def wrap_custom_layer(module_to_wrap: torch.nn.Module, custom_layer_type: type[T]) -> T:
    # HACK(ryand): We use custom initialization logic so that we can initialize a new custom layer instance from an
    # existing layer instance without calling __init__() on the original layer class. We achieve this by copying
    # the attributes from the original layer instance to the new instance.
    custom_layer = custom_layer_type.__new__(custom_layer_type)
    # Note that we share the __dict__.
    # TODO(ryand): In the future, we may want to do a shallow copy of the __dict__.
    custom_layer.__dict__ = module_to_wrap.__dict__

    # Initialize the CustomModuleMixin fields.
    CustomModuleMixin.__init__(custom_layer)  # type: ignore
    return custom_layer


def unwrap_custom_layer(custom_layer: torch.nn.Module, original_layer_type: type[torch.nn.Module]):
    # HACK(ryand): We use custom initialization logic so that we can initialize a new custom layer instance from an
    # existing layer instance without calling __init__() on the original layer class. We achieve this by copying
    # the attributes from the original layer instance to the new instance.
    original_layer = original_layer_type.__new__(original_layer_type)
    # Note that we share the __dict__.
    # TODO(ryand): In the future, we may want to do a shallow copy of the __dict__ and strip out the CustomModuleMixin
    # fields.
    original_layer.__dict__ = custom_layer.__dict__
    return original_layer


def apply_custom_layers_to_model(module: torch.nn.Module, device_autocasting_enabled: bool = False):
    for name, submodule in module.named_children():
        override_type = AUTOCAST_MODULE_TYPE_MAPPING.get(type(submodule), None)
        if override_type is not None:
            custom_layer = wrap_custom_layer(submodule, override_type)
            # TODO(ryand): In the future, we should manage this flag on a per-module basis.
            custom_layer.set_device_autocasting_enabled(device_autocasting_enabled)
            setattr(module, name, custom_layer)
        else:
            # Recursively apply to submodules
            apply_custom_layers_to_model(submodule, device_autocasting_enabled)


def remove_custom_layers_from_model(module: torch.nn.Module):
    for name, submodule in module.named_children():
        override_type = AUTOCAST_MODULE_TYPE_MAPPING_INVERSE.get(type(submodule), None)
        if override_type is not None:
            setattr(module, name, unwrap_custom_layer(submodule, override_type))
        else:
            remove_custom_layers_from_model(submodule)
