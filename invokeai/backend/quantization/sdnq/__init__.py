"""SDNQ (SD.Next Quantization) support for InvokeAI.

This module provides support for loading SDNQ quantized models with on-the-fly
CPU dequantization, similar to GGUF support.
"""

from invokeai.backend.quantization.sdnq.loaders import has_sdnq_keys, has_sdnq_tensors, sdnq_sd_loader
from invokeai.backend.quantization.sdnq.sdnq_tensor import SDNQTensor
from invokeai.backend.quantization.sdnq.utils import SDNQQuantizationType

__all__ = [
    "SDNQTensor",
    "sdnq_sd_loader",
    "has_sdnq_tensors",
    "has_sdnq_keys",
    "SDNQQuantizationType",
]
