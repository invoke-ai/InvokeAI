"""Linears that keep their weight quantized in buffers and materialize it on every forward.

Two schemes store their Linears this way, ``int8_convrot`` and ``nvfp4``, and the code around them
needs the same two answers about either: whether a LoRA has to ride along as a sidecar, and how much
memory one forward needs on top of the model's resident size. Both are asked of the loaded module tree,
so they live here beside the base class the modules share rather than inside either scheme -- a new
scheme answers them by subclassing instead of every caller learning its name.
"""

import abc
from typing import Any

import torch
import torch.nn.functional as F

from invokeai.backend.model_manager.taxonomy import ModelFormat


class DequantizingLinear(torch.nn.Module, abc.ABC):
    """A Linear whose weight is kept quantized as buffers and dequantized on every forward.

    Subclasses register their payload as persistent buffers under the checkpoint's own key names, so
    ``load_state_dict`` assigns the quantized tensors straight into them and the model cache moves them
    between devices like any other weight. ``bias`` is a buffer as well, or ``None``.

    ``forward`` takes the weight from :meth:`_dequantized_weight` on the input's device and in its dtype,
    so a partial load that left the buffers on the CPU streams them per call instead of failing. The model
    cache wraps these modules for sidecar patches (see ``AUTOCAST_MODULE_TYPE_MAPPING``).

    Both methods are abstract so that a scheme missing one fails where its modules are built, at load,
    rather than in the denoise node that sizes its working memory from them.
    """

    in_features: int
    out_features: int
    bias: torch.Tensor | None

    @abc.abstractmethod
    def _dequantized_weight(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """The weight as ``[out_features, in_features]`` in ``dtype`` on ``device``, for one forward."""

    @abc.abstractmethod
    def dequant_transient_bytes(self, compute_dtype: torch.dtype) -> int:
        """Peak bytes one forward of this layer allocates beyond its resident buffers."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self._dequantized_weight(x.device, x.dtype)
        bias = self.bias.to(device=x.device, dtype=x.dtype) if self.bias is not None else None
        return F.linear(x, weight, bias)


def requires_sidecar_patching(model: Any, model_format: ModelFormat) -> bool:
    """Whether LoRA has to be applied as a sidecar rather than written into the weights.

    The format alone does not answer this. A plain ``checkpoint`` may still hold
    :class:`DequantizingLinear` modules -- an ``int8_tensorwise`` or ``nvfp4`` build whose Linears the
    loader replaced -- and those keep their weights as quantized *buffers*, which a direct patch cannot
    write into. Worse, the patcher's own fallbacks iterate ``module.parameters()``, and these modules have
    none, so they answer "not quantized" and direct patching is chosen. So the loaded module tree is
    consulted, not just the config.
    """
    if model_format in (ModelFormat.GGUFQuantized, ModelFormat.SDNQQuantized):
        return True
    return any(isinstance(module, DequantizingLinear) for module in model.modules())


def peak_dequant_transient_bytes(model: torch.nn.Module, compute_dtype: torch.dtype) -> int:
    """Peak bytes one forward transiently needs to dequantize this model's quantized linears.

    A :class:`DequantizingLinear` materializes its weight per forward, so that allocation is not covered
    by the model's resident size and has to fit inside the calling node's working-memory reservation. The
    layers run one after another and free their weight before the next one allocates, so the peak is the
    largest single layer's transient. Zero when the model holds no such layer.

    Not covered: the matmul's own cuBLAS workspace, which a dense Linear of the same shape allocates too
    and which a node's activation estimate already includes.
    """
    return max(
        (
            module.dequant_transient_bytes(compute_dtype)
            for module in model.modules()
            if isinstance(module, DequantizingLinear)
        ),
        default=0,
    )
