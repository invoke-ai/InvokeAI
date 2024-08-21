from typing import Any, Dict

import torch
from optimum.quanto.quantize import _quantize_submodule


def requantize(
    model: torch.nn.Module,
    state_dict: Dict[str, Any],
    quantization_map: Dict[str, Dict[str, str]],
    device: torch.device | None = None,
):
    """This function was initially copied from:
    https://github.com/huggingface/optimum-quanto/blob/832f7f5c3926c91fe4f923aaaf037a780ac3e6c3/optimum/quanto/quantize.py#L101

    The function was modified to remove the `freeze()` call. The `freeze()` call is very slow and unnecessary when the
    weights are about to be loaded from a state_dict.

    TODO(ryand): Unless I'm overlooking something, this should be contributed upstream to the `optimum-quanto` library.
    """
    if device is None:
        device = next(model.parameters()).device
        if device.type == "meta":
            device = torch.device("cpu")

    # Quantize the model with parameters from the quantization map
    for name, m in model.named_modules():
        qconfig = quantization_map.get(name, None)
        if qconfig is not None:
            weights = qconfig["weights"]
            if weights == "none":
                weights = None
            activations = qconfig["activations"]
            if activations == "none":
                activations = None
            _quantize_submodule(model, name, m, weights=weights, activations=activations)

    # Move model parameters and buffers to CPU before materializing quantized weights
    for name, m in model.named_modules():

        def move_tensor(t, device):
            if t.device.type == "meta":
                return torch.empty_like(t, device=device)
            return t.to(device)

        for name, param in m.named_parameters(recurse=False):
            setattr(m, name, torch.nn.Parameter(move_tensor(param, "cpu")))
        for name, param in m.named_buffers(recurse=False):
            setattr(m, name, move_tensor(param, "cpu"))

    # Freeze model and move to target device
    # freeze(model)
    # model.to(device)

    # Load the quantized model weights
    model.load_state_dict(state_dict, strict=False)
