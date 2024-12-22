import torch

from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.autocast_modules import (
    CustomConv1d,
    CustomConv2d,
    CustomEmbedding,
    CustomGroupNorm,
    CustomInvokeLinear8bitLt,
    CustomLinear,
)
from invokeai.backend.quantization.bnb_llm_int8 import InvokeLinear8bitLt

AUTOCAST_MODULE_TYPE_MAPPING: dict[type[torch.nn.Module], type[torch.nn.Module]] = {
    torch.nn.Linear: CustomLinear,
    torch.nn.Conv1d: CustomConv1d,
    torch.nn.Conv2d: CustomConv2d,
    torch.nn.GroupNorm: CustomGroupNorm,
    torch.nn.Embedding: CustomEmbedding,
    InvokeLinear8bitLt: CustomInvokeLinear8bitLt,
}


def apply_custom_layers_to_model(model: torch.nn.Module):
    def apply_custom_layers(module: torch.nn.Module):
        override_type = AUTOCAST_MODULE_TYPE_MAPPING.get(type(module), None)
        if override_type is not None:
            module.__class__ = override_type

    # model.apply(...) calls apply_custom_layers(...) on each module in the model.
    model.apply(apply_custom_layers)


def remove_custom_layers_from_model(model: torch.nn.Module):
    # Invert AUTOCAST_MODULE_TYPE_MAPPING.
    original_module_type_mapping = {v: k for k, v in AUTOCAST_MODULE_TYPE_MAPPING.items()}

    def remove_custom_layers(module: torch.nn.Module):
        override_type = original_module_type_mapping.get(type(module), None)
        if override_type is not None:
            module.__class__ = override_type

    # model.apply(...) calls remove_custom_layers(...) on each module in the model.
    model.apply(remove_custom_layers)