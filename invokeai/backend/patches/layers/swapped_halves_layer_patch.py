import torch

from invokeai.backend.patches.layers.base_layer_patch import BaseLayerPatch


def _swap_halves(t: torch.Tensor) -> torch.Tensor:
    # index_select rather than chunk + cat: it also handles the sparse COO bias a LyCORIS layer can carry.
    n = t.shape[0]
    return t.index_select(0, torch.arange(n, device=t.device).roll(n // 2))


class SwappedHalvesLayerPatch(BaseLayerPatch):
    """Applies a patch written for a layer whose output is split into two halves, to a module that stores those halves
    in the opposite order along the first (output) axis.

    The FLUX final-layer modulation is the case this exists for: diffusers' `AdaLayerNormContinuous` (`norm_out.linear`)
    orders its output as (scale, shift), BFL's `LastLayer` (`final_layer.adaLN_modulation.1`) as (shift, scale).
    Wrapping the built patch works for every layer type (LoRA, DoRA, LoHA, LoKR, full) without rearranging their
    factors.
    """

    def __init__(self, layer: BaseLayerPatch):
        super().__init__()
        self.layer = layer

    def get_parameters(self, orig_parameters: dict[str, torch.Tensor], weight: float) -> dict[str, torch.Tensor]:
        # The inner layer sees the original parameters in its own layout, so layers that read their values (DoRA)
        # stay exact. Swapping meta tensors keeps their shapes valid for shape-only layers.
        source_layout_parameters = {name: _swap_halves(param) for name, param in orig_parameters.items()}
        out_parameters = self.layer.get_parameters(source_layout_parameters, weight)
        return {name: _swap_halves(param) for name, param in out_parameters.items()}

    def to(self, device: torch.device | None = None, dtype: torch.dtype | None = None):
        self.layer.to(device=device, dtype=dtype)

    def calc_size(self) -> int:
        return self.layer.calc_size()
