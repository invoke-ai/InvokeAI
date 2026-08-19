"""The scaled-fp8 FLUX.1 key layout must be recognized, scales and all.

Before scaled-fp8 support reached `FluxCheckpointModel`, this checkpoint did not load at all: the
loader knew nothing about `.scale_weight`/`.scale_input`, so all 629 scale and marker keys reached
`load_state_dict` as unexpected keys and it raised.

The fixture is a real key layout; the tensors are synthetic because only shapes and dtypes matter
to the code under test.
"""

import torch

from invokeai.backend.model_manager.util.model_util import convert_bundle_to_flux_transformer_checkpoint
from invokeai.backend.quantization.fp8_scaled import (
    FP8_DTYPE,
    can_stay_quantized,
    extract_comfy_quant_hints,
    extract_fp8_scaled_layers,
    is_scale_metadata_key,
)
from tests.backend.model_manager.load.state_dicts.flux1_transformer_scaled_fp8_keys import (
    state_dict_keys as scaled_keys,
)

_DTYPES = {"F8_E4M3": FP8_DTYPE, "F32": torch.float32, "BF16": torch.bfloat16}


def _build_state_dict() -> dict[str, torch.Tensor]:
    sd: dict[str, torch.Tensor] = {}
    for key, (shape, dtype) in scaled_keys.items():
        torch_dtype = _DTYPES[dtype]
        if torch_dtype is FP8_DTYPE:
            sd[key] = torch.zeros(shape, dtype=torch.float32).to(FP8_DTYPE)
        elif key.endswith((".scale_weight", ".scale_input")):
            # Real checkpoints carry calibrated scales; 1.0 is the placeholder value that
            # `_usable_input_scale` deliberately rejects, so it must not be used here.
            sd[key] = torch.full(shape, 2.5, dtype=torch_dtype)
        else:
            sd[key] = torch.zeros(shape, dtype=torch_dtype)
    return sd


def test_every_quantized_linear_is_recognized() -> None:
    """Each fp8 `.weight` in the fixture must come back as a scaled layer, not be left behind."""
    sd = _build_state_dict()
    fp8_weights = {k for k, v in sd.items() if v.dtype is FP8_DTYPE and k.endswith(".weight")}
    assert fp8_weights, "fixture carries no fp8 weights"

    layers = extract_fp8_scaled_layers(sd, layer_hints=extract_comfy_quant_hints(sd))

    assert {f"{path}.weight" for path in layers} == fp8_weights


def test_the_fp8_biases_must_not_stay_quantized() -> None:
    """This checkpoint quantizes the biases too -- 314 of them, alongside the 314 weights.

    An fp8 bias saves nothing usable and breaks inference: the value reaches the activations and
    the next Linear receives an fp8 *input*, which dies in `x.abs()` with
    `"abs_cuda" not implemented for 'Float8_e4m3fn'`. Only 2-D `nn.Linear.weight` may stay
    quantized, which is what `can_stay_quantized` decides -- a bias is 1-D and fails it.
    """
    fp8_biases = [k for k, (_, dtype) in scaled_keys.items() if dtype == "F8_E4M3" and k.endswith(".bias")]
    assert fp8_biases, "fixture is expected to carry fp8 biases -- the real checkpoint does"

    sd = _build_state_dict()
    for key in fp8_biases:
        assert not can_stay_quantized(key, sd[key], None)


def test_the_second_spelling_is_read() -> None:
    """This checkpoint spells the scales `.scale_weight`/`.scale_input`.

    Reading only `.weight_scale`/`.input_scale` would drop every scale here, leaving each weight
    off by `1/scale` with nothing logged.
    """
    assert any(k.endswith(".scale_weight") for k in scaled_keys)
    assert not any(k.endswith(".weight_scale") for k in scaled_keys)

    layers = extract_fp8_scaled_layers(_build_state_dict())

    assert layers
    assert all(layer.weight_scale is not None for layer in layers.values())


def test_the_calibrated_input_scales_survive() -> None:
    """Every quantized Linear here ships a calibrated `scale_input`; none may be discarded.

    `_usable_input_scale` drops uncalibrated placeholders (exactly 1.0, non-finite, <= 0). A
    calibrated value must pass, otherwise every forward pays an amax reduction it does not need.
    """
    layers = extract_fp8_scaled_layers(_build_state_dict())

    assert all(layer.input_scale is not None for layer in layers.values())


def test_extraction_strips_every_scale_and_marker_key() -> None:
    """What is left must load into a model that knows nothing about fp8."""
    sd = _build_state_dict()
    assert "scaled_fp8" in sd, "fixture is missing the producer's marker key"

    extract_fp8_scaled_layers(sd)

    assert not [k for k in sd if is_scale_metadata_key(k)]
    assert "scaled_fp8" not in sd


def test_qkv_stays_fused_with_one_scalar_scale() -> None:
    """FLUX.1 needs no qkv split, unlike FLUX.2.

    InvokeAI's `Flux` implements the BFL layout, where `qkv` is a single Linear. The checkpoint
    matches it, so the per-tensor scale attaches to exactly the module it was computed for. If this
    ever changes, the scale would have to be *copied* to each split part rather than moved.
    """
    qkv = [k for k in scaled_keys if k.endswith(".qkv.weight")]
    assert qkv, "fixture carries no fused qkv weight"

    for key in qkv:
        shape, dtype = scaled_keys[key]
        assert dtype == "F8_E4M3"
        assert shape[0] == 3 * shape[1], f"{key} is not a fused qkv: {shape}"
        assert scaled_keys[key.replace(".weight", ".scale_weight")][0] == []


def test_bundle_conversion_keeps_the_scales_at_full_precision() -> None:
    """The bundle converter casts RMSNorm `.scale` to bf16; fp8 quantization scales must not follow.

    `.weight_scale` also ends in "scale". Folding an f32 quantization scale to bf16 leaves 8
    mantissa bits on a value every quantized weight is multiplied by.
    """
    sd = {
        "model.diffusion_model.double_blocks.0.img_attn.norm.key_norm.scale": torch.ones(128, dtype=torch.float32),
        "model.diffusion_model.double_blocks.0.img_attn.qkv.weight_scale": torch.full((), 2.5, dtype=torch.float32),
        "model.diffusion_model.double_blocks.0.img_attn.qkv.scale_weight": torch.full((), 2.5, dtype=torch.float32),
        "model.diffusion_model.double_blocks.0.img_attn.qkv.weight": torch.zeros(9216, 3072, dtype=torch.float32).to(
            FP8_DTYPE
        ),
    }

    converted = convert_bundle_to_flux_transformer_checkpoint(sd)

    assert converted["double_blocks.0.img_attn.norm.key_norm.scale"].dtype is torch.bfloat16
    assert converted["double_blocks.0.img_attn.qkv.weight_scale"].dtype is torch.float32
    assert converted["double_blocks.0.img_attn.qkv.scale_weight"].dtype is torch.float32
