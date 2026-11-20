import torch

from invokeai.backend.patches.layers.base_layer_patch import BaseLayerPatch
from invokeai.backend.patches.layers.dora_layer import DoRALayer
from invokeai.backend.quantization.fp8_scaled import dequantize_weight
from invokeai.backend.util.fp8 import FP8_STORAGE_DTYPES

# At most 4 MiB per float32 row slab (or one row for wider weights). The norm pass keeps only
# a base slab, a direction slab, the low-rank factors and an input-width accumulator, never fused W.
_NORM_BLOCK_ELEMENTS = 1 << 20


class FusedInputDoRALayer(BaseLayerPatch):
    """Unbound fused DoRA: loading knows the factors, but only application knows the Q/K/V bases.

    Cached instances contain no base weights or computed norms. Binding happens once per adapter
    application, before any of its projections change, and returns ordinary local DoRA patches.
    """

    def __init__(self, layer: DoRALayer, keys: tuple[str, str, str]):
        self.layer = layer
        self.keys = keys

    @torch.no_grad()
    def bind(
        self, model: torch.nn.Module, base_dtypes: tuple[torch.dtype | None, ...] = (None, None, None)
    ) -> list[tuple[str, DoRALayer]]:
        modules = [model.get_submodule(key) for key in self.keys]
        for module in modules:
            # Binding needs all three base weights before patching; packed modules and tensor subclasses
            # do not provide the ordinary floating-point weight slabs used by this norm pass.
            if (
                not isinstance(module, torch.nn.Linear)
                or module.weight.device.type == "meta"
                or type(module.weight.data) is not torch.Tensor
                or not module.weight.is_floating_point()
            ):
                raise RuntimeError(
                    "Fused input-axis DoRA requires ordinary floating-point Q/K/V weights; "
                    "this base model quantization format is not supported for that operation."
                )

        layer = self.layer
        up_parts = layer.up.chunk(3, dim=0)
        norm_sq = torch.zeros(layer.down.shape[1], device=modules[0].weight.device, dtype=torch.float32)
        for module, up, base_dtype in zip(modules, up_parts, base_dtypes, strict=True):
            weight = module.weight
            down = layer.down.to(device=weight.device, dtype=torch.float32)
            scale = getattr(module, "weight_scale", None) if weight.dtype in FP8_STORAGE_DTYPES else None
            step = max(1, _NORM_BLOCK_ELEMENTS // weight.shape[1])
            for start in range(0, weight.shape[0], step):
                end = min(start + step, weight.shape[0])
                slab_scale = scale
                if scale is not None and scale.numel() > 1:
                    # Select compact scale rows before expanding, including block-wise scales.
                    rows_per_scale = weight.shape[0] // scale.shape[0]
                    indices = torch.arange(start, end, device=scale.device) // rows_per_scale
                    slab_scale = scale.index_select(0, indices)
                base = dequantize_weight(weight[start:end], slab_scale, base_dtype or torch.float32).float()
                direction = torch.addmm(
                    base, up[start:end].to(device=weight.device, dtype=torch.float32), down, alpha=layer.scale()
                )
                norm_sq.add_(direction.square_().sum(dim=0).to(norm_sq.device))
                del base, direction
        norm = norm_sq.sqrt_().reshape(1, -1)
        bias = layer.bias.to_dense() if layer.bias is not None and layer.bias.is_sparse else layer.bias
        bias_parts = (None, None, None) if bias is None else bias.chunk(3, dim=0)
        return [
            (
                key,
                DoRALayer(
                    up=up,
                    down=layer.down,
                    dora_scale=layer.dora_scale.reshape(1, -1),
                    alpha=layer._alpha,
                    bias=bias,
                    direction_norm=norm,
                ),
            )
            for key, up, bias in zip(self.keys, up_parts, bias_parts, strict=True)
        ]

    def binding_workspace_bytes(self, module: torch.nn.Linear, dtype: torch.dtype) -> int:
        """Bytes for one projection's FP32 norm pass, apart from the bound norm itself."""
        rows, columns = module.out_features, module.in_features
        slab_rows = min(rows, max(1, _NORM_BLOCK_ELEMENTS // columns))
        fp32 = torch.float32.itemsize
        scale = getattr(module, "weight_scale", None)
        # A scaled FP8 slab can have a cast, indexed scales, expanded scales, and multiply output
        # in flight. The FP32 addmm needs a base and a direction slab after dequantization.
        slab_bytes = (
            4 * slab_rows * columns * max(fp32, dtype.itemsize, scale.element_size() if scale is not None else 0)
        )
        rank = self.layer.down.shape[0]
        # One up slab and a down factor. Replacing down between projections can briefly hold two.
        factor_bytes = (slab_rows * rank + 2 * self.layer.down.numel()) * fp32
        vector_bytes = (slab_rows + columns) * (2 * torch.int64.itemsize + 2 * fp32)
        return slab_bytes + factor_bytes + vector_bytes

    def get_parameters(self, orig_parameters: dict[str, torch.Tensor], weight: float) -> dict[str, torch.Tensor]:
        raise RuntimeError("Fused input-axis DoRA must be bound to its base model before application.")

    def to(self, device: torch.device | None = None, dtype: torch.dtype | None = None):
        self.layer.to(device=device, dtype=dtype)

    def calc_size(self) -> int:
        return self.layer.calc_size()
