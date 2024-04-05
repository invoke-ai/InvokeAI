from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Tuple

import torch

from invokeai.backend.peft.peft_model import PeftModel


class PeftModelPatcher:
    @classmethod
    @contextmanager
    @torch.no_grad()
    def apply_peft_patch(
        cls,
        model: torch.nn.Module,
        peft_models: Iterator[Tuple[PeftModel, float]],
        prefix: str,
    ):
        original_weights = {}

        model_state_dict = model.state_dict()
        try:
            for peft_model, peft_model_weight in peft_models:
                for layer_key, layer in peft_model.state_dict.items():
                    if not layer_key.startswith(prefix):
                        continue

                    module_key = layer_key.replace(prefix + ".", "")
                    module_key = module_key.split
                    # TODO(ryand): Make this work.
                    module = model_state_dict[module_key]

                    # All of the LoRA weight calculations will be done on the same device as the module weight.
                    # (Performance will be best if this is a CUDA device.)
                    device = module.weight.device
                    dtype = module.weight.dtype

                    if module_key not in original_weights:
                        # TODO(ryand): Set non_blocking = True?
                        original_weights[module_key] = module.weight.detach().to(device="cpu", copy=True)

                    layer_scale = layer.alpha / layer.rank if (layer.alpha and layer.rank) else 1.0

                    # We intentionally move to the target device first, then cast. Experimentally, this was found to
                    # be significantly faster for 16-bit CPU tensors being moved to a CUDA device than doing the
                    # same thing in a single call to '.to(...)'.
                    layer.to(device=device)
                    layer.to(dtype=torch.float32)
                    # TODO(ryand): Using torch.autocast(...) over explicit casting may offer a speed benefit on CUDA
                    # devices here. Experimentally, it was found to be very slow on CPU. More investigation needed.
                    layer_weight = layer.get_weight(module.weight) * (lora_weight * layer_scale)
                    layer.to(device=torch.device("cpu"))

                    assert isinstance(layer_weight, torch.Tensor)  # mypy thinks layer_weight is a float|Any ??!
                    if module.weight.shape != layer_weight.shape:
                        # TODO: debug on lycoris
                        assert hasattr(layer_weight, "reshape")
                        layer_weight = layer_weight.reshape(module.weight.shape)

                    assert isinstance(layer_weight, torch.Tensor)  # mypy thinks layer_weight is a float|Any ??!
                    module.weight += layer_weight.to(dtype=dtype)
            yield
        finally:
            for module_key, weight in original_weights.items():
                model.get_submodule(module_key).weight.copy_(weight)
