import torch

from invokeai.backend.patches.layers.lora_layer import LoRALayer


class FluxControlLoRALayer(LoRALayer):
    """A special case of LoRALayer for use with FLUX Control LoRAs that pads the target parameter with zeros if the
    shapes don't match.
    """

    def get_parameters(self, orig_parameters: dict[str, torch.Tensor], weight: float) -> dict[str, torch.Tensor]:
        """This overrides the base class behavior to skip the reshaping step."""
        scale = self.scale()
        params = {"weight": self.get_weight(orig_parameters["weight"]) * (weight * scale)}
        bias = self.get_bias(orig_parameters.get("bias", None))
        if bias is not None:
            params["bias"] = bias * (weight * scale)

        return params
