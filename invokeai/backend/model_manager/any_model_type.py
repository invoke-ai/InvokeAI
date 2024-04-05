from typing import Union

import torch
from diffusers.models.modeling_utils import ModelMixin

from invokeai.backend.ip_adapter.ip_adapter import IPAdapter
from invokeai.backend.onnx.onnx_runtime import IAIOnnxRuntimeModel
from invokeai.backend.peft.peft_model import PeftModel
from invokeai.backend.textual_inversion import TextualInversionModelRaw

# ModelMixin is the base class for all diffusers and transformers models
AnyModel = Union[ModelMixin, torch.nn.Module, IPAdapter, PeftModel, TextualInversionModelRaw, IAIOnnxRuntimeModel]
