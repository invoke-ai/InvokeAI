"""The BitsAndBytes layers transformers builds for the Qwen-Image text encoder accept a sidecar LoRA after the node's wrap.

bitsandbytes quantizes on the CPU as well (0.50), so this guards `BNB_LINEAR_TYPE_MAPPING` and the wrap in every CI
lane; `test_qwen_vl_bnb_lora.py` covers the same on a real model with CUDA. float32 throughout: CPU bf16 matmul is
unsafe on part of the CI fleet.
"""

import pytest
import torch

bnb = pytest.importorskip("bitsandbytes")

from invokeai.app.invocations.text_encoder.qwen_image_text_encoder import _make_bnb_layers_patchable  # noqa: E402
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.custom_modules.custom_invoke_linear_8_bit_lt import (  # noqa: E402
    CustomInvokeLinear8bitLt,
)
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.custom_modules.custom_invoke_linear_nf4 import (  # noqa: E402
    CustomInvokeLinearNF4,
)
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.custom_modules.custom_linear import (  # noqa: E402
    CustomLinear,
)
from invokeai.backend.patches.layer_patcher import LayerPatcher  # noqa: E402
from invokeai.backend.patches.layers.utils import any_lora_layer_from_state_dict  # noqa: E402
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw  # noqa: E402

FEATURES = 64
RANK = 4
WEIGHT = 0.6


class _TinyEncoder(torch.nn.Module):
    """One quantized Linear as transformers builds it, and one that transformers leaves alone (like `lm_head`)."""

    def __init__(self, quantization: str) -> None:
        super().__init__()
        torch.manual_seed(0)
        weight = torch.randn(FEATURES, FEATURES)
        if quantization == "int8":
            self.quantized = bnb.nn.Linear8bitLt(FEATURES, FEATURES, bias=False, has_fp16_weights=False)
            self.quantized.weight = bnb.nn.Int8Params(weight, requires_grad=False, has_fp16_weights=False)
        else:
            self.quantized = bnb.nn.Linear4bit(
                FEATURES, FEATURES, bias=False, compute_dtype=torch.float32, quant_type="nf4"
            )
            self.quantized.weight = bnb.nn.Params4bit(weight, requires_grad=False, quant_type="nf4")
        self.plain = torch.nn.Linear(FEATURES, FEATURES, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.plain(self.quantized(x))


def _build(quantization: str) -> _TinyEncoder:
    try:
        # Moving the module is what quantizes the weights.
        return _TinyEncoder(quantization).to("cpu")
    except (NotImplementedError, RuntimeError) as exc:
        pytest.skip(f"bitsandbytes {bnb.__version__} cannot quantize {quantization} on the CPU: {exc}")


@pytest.mark.parametrize("quantization", ["int8", "nf4"])
def test_a_sidecar_lora_applies_to_wrapped_bnb_layers_and_leaves_no_trace(quantization: str) -> None:
    reference = _build(quantization)
    model = _build(quantization)
    x = torch.randn(3, FEATURES, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        expected_output = reference(x)

    # Before the first forward, as in the node: the plain int8 forward moves CB/SCB off the weight.
    _make_bnb_layers_patchable(model)

    assert type(model.quantized) is (CustomInvokeLinear8bitLt if quantization == "int8" else CustomInvokeLinearNF4)
    assert type(model.plain) is CustomLinear

    generator = torch.Generator().manual_seed(2)
    down, up = torch.randn(RANK, FEATURES, generator=generator), torch.randn(FEATURES, RANK, generator=generator)
    patch = ModelPatchRaw(
        layers={
            f"lora_qwen_vl-{name}": any_lora_layer_from_state_dict({"lora_down.weight": down, "lora_up.weight": up})
            for name in ("quantized", "plain")
        }
    )
    with torch.no_grad():
        unpatched = model(x)
        # A wrapped layer without a patch computes what the plain one does.
        torch.testing.assert_close(unpatched, expected_output, atol=0, rtol=0)

        with LayerPatcher.apply_smart_model_patches(
            model=model,
            patches=[(patch, WEIGHT)],
            prefix="lora_qwen_vl-",
            dtype=torch.float32,
            force_sidecar_patching=True,
        ):
            patched_quantized = model.quantized(x)
            patched_plain = model.plain(x)
        restored = model(x)

    shift = x @ (up @ down).T * WEIGHT
    torch.testing.assert_close(patched_quantized - reference.quantized(x), shift, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(patched_plain - reference.plain(x), shift, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(restored, expected_output, atol=0, rtol=0)
    assert model.quantized.get_num_patches() == 0 and model.plain.get_num_patches() == 0
