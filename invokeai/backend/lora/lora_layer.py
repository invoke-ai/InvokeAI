from typing import Optional

import torch

from invokeai.backend.lora.lora_layer_base import LoRALayerBase


class LoRALayer(LoRALayerBase):
    def __init__(
        self,
        layer_key: str,
        values: dict[str, torch.Tensor],
    ):
        super().__init__(layer_key, values)

        self.up = values["lora_up.weight"]
        self.down = values["lora_down.weight"]

        self.mid: Optional[torch.Tensor] = values.get("lora_mid.weight", None)
        self.dora_scale: Optional[torch.Tensor] = values.get("dora_scale", None)
        self.rank = self.down.shape[0]

    def _apply_dora(self, orig_weight: torch.Tensor, lora_weight: torch.Tensor) -> torch.Tensor:
        """Apply DoRA to the weight matrix.

        This function is based roughly on the reference implementation in PEFT, but handles scaling in a slightly
        different way:
        https://github.com/huggingface/peft/blob/26726bf1ddee6ca75ed4e1bfd292094526707a78/src/peft/tuners/lora/layer.py#L421-L433

        """
        # Merge the original weight with the LoRA weight.
        merged_weight = orig_weight + lora_weight

        # Calculate the vector-wise L2 norm of the weight matrix across each column vector.
        weight_norm: torch.Tensor = torch.linalg.norm(merged_weight, dim=1)

        dora_factor = self.dora_scale / weight_norm
        new_weight = dora_factor * merged_weight

        # TODO(ryand): This is wasteful. We already have the final weight, but we calculate the diff, because that is
        # what the `get_weight()` API is expected to return. If we do refactor this, we'll have to give some thought to
        # how lora weight scaling should be applied - having the full weight diff makes this easy.
        weight_diff = new_weight - orig_weight
        return weight_diff

    def get_weight(self, orig_weight: Optional[torch.Tensor]) -> torch.Tensor:
        if self.mid is not None:
            up = self.up.reshape(self.up.shape[0], self.up.shape[1])
            down = self.down.reshape(self.down.shape[0], self.down.shape[1])
            weight = torch.einsum("m n w h, i m, n j -> i j w h", self.mid, up, down)
        else:
            weight = self.up.reshape(self.up.shape[0], -1) @ self.down.reshape(self.down.shape[0], -1)

        if self.dora_scale is not None:
            assert orig_weight is not None
            weight = self._apply_dora(orig_weight, weight)

        return weight

    def calc_size(self) -> int:
        model_size = super().calc_size()
        for val in [self.up, self.mid, self.down]:
            if val is not None:
                model_size += val.nelement() * val.element_size()
        return model_size

    def to(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().to(device=device, dtype=dtype)

        self.up = self.up.to(device=device, dtype=dtype)
        self.down = self.down.to(device=device, dtype=dtype)

        if self.mid is not None:
            self.mid = self.mid.to(device=device, dtype=dtype)

        if self.dora_scale is not None:
            self.dora_scale = self.dora_scale.to(device=device, dtype=dtype)
