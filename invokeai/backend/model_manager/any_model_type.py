from typing import Union

import torch
from diffusers.models.modeling_utils import ModelMixin

from invokeai.backend.ip_adapter.ip_adapter import IPAdapter
from invokeai.backend.lora.lora_model import LoRAModelRaw
from invokeai.backend.onnx.onnx_runtime import IAIOnnxRuntimeModel
from invokeai.backend.textual_inversion import TextualInversionModelRaw

# ModelMixin is the base class for all diffusers and transformers models
AnyModel = Union[ModelMixin, torch.nn.Module, IPAdapter, LoRAModelRaw, TextualInversionModelRaw, IAIOnnxRuntimeModel]
