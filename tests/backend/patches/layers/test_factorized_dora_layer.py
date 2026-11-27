"""LoKr and LoHa layers that carry a DoRA magnitude (LyCORIS ``dora_wd=True``) apply the magnitude.

Each expected update is written from the LyCORIS definition in float64, independently of the layer classes:
``W' = m * (W + s * dW) / ||W + s * dW||``, the norm taken over the axis the magnitude does not index, applied as the
residual ``W' - W``.
"""

import copy
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
import torch

import invokeai.backend.patches.layers.dora_layer as dora_module
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
    apply_custom_layers_to_model,
)
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.patches.layers.dora_layer import DoRALayer, FactorizedDoRALayer, reads_weight_values
from invokeai.backend.patches.layers.loha_layer import LoHALayer
from invokeai.backend.patches.layers.lokr_layer import LoKRLayer
from invokeai.backend.patches.layers.merged_layer_patch import MergedLayerPatch, Range
from invokeai.backend.patches.layers.swapped_halves_layer_patch import SwappedHalvesLayerPatch
from invokeai.backend.patches.layers.utils import any_lora_layer_from_state_dict
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw

F64 = torch.float64


@dataclass
class Case:
    state_dict: dict[str, torch.Tensor]
    weight_shape: tuple[int, ...]
    update: torch.Tensor  # alpha / rank * dW, in the weight's shape


def _randn(generator: torch.Generator, *shape: int) -> torch.Tensor:
    return torch.randn(*shape, generator=generator, dtype=F64) * 0.4


def _kron(w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    """kron(w1, w2) over the first two dims, with the kernel dims of a convolution's `w2` kept."""
    return torch.einsum("ij,kl...->ikjl...", w1, w2).reshape(
        w1.shape[0] * w2.shape[0], w1.shape[1] * w2.shape[1], *w2.shape[2:]
    )


def _lokr_whole(g: torch.Generator) -> Case:
    w1, w2 = _randn(g, 3, 4), _randn(g, 4, 2)
    return Case({"lokr_w1": w1, "lokr_w2": w2}, (12, 8), _kron(w1, w2))


def _lokr_factorized(g: torch.Generator) -> Case:
    rank = 2
    w1_a, w1_b, w2_a, w2_b = _randn(g, 3, rank), _randn(g, rank, 4), _randn(g, 4, rank), _randn(g, rank, 2)
    sd = {
        "lokr_w1_a": w1_a,
        "lokr_w1_b": w1_b,
        "lokr_w2_a": w2_a,
        "lokr_w2_b": w2_b,
        "alpha": torch.tensor(5.0, dtype=F64),
    }
    return Case(sd, (12, 8), _kron(w1_a @ w1_b, w2_a @ w2_b) * (5.0 / rank))


def _lokr_conv(g: torch.Generator) -> Case:
    w1, w2 = _randn(g, 2, 3), _randn(g, 4, 2, 3, 3)
    return Case({"lokr_w1": w1, "lokr_w2": w2}, (8, 6, 3, 3), _kron(w1, w2))


def _lokr_tucker(g: torch.Generator) -> Case:
    rank = 3
    w1 = _randn(g, 2, 3)
    t2, w2_a, w2_b = _randn(g, rank, rank, 3, 3), _randn(g, rank, 4), _randn(g, rank, 2)
    w2 = torch.einsum("ijkl,ip,jr->prkl", t2, w2_a, w2_b)
    sd = {"lokr_w1": w1, "lokr_t2": t2, "lokr_w2_a": w2_a, "lokr_w2_b": w2_b, "alpha": torch.tensor(1.5, dtype=F64)}
    return Case(sd, (8, 6, 3, 3), _kron(w1, w2) * (1.5 / rank))


def _loha(g: torch.Generator) -> Case:
    rank = 3
    w1_a, w1_b, w2_a, w2_b = _randn(g, 12, rank), _randn(g, rank, 8), _randn(g, 12, rank), _randn(g, rank, 8)
    sd = {
        "hada_w1_a": w1_a,
        "hada_w1_b": w1_b,
        "hada_w2_a": w2_a,
        "hada_w2_b": w2_b,
        "alpha": torch.tensor(1.0, dtype=F64),
    }
    return Case(sd, (12, 8), (w1_a @ w1_b) * (w2_a @ w2_b) * (1.0 / rank))


def _loha_tucker(g: torch.Generator) -> Case:
    rank = 3
    t1, t2 = _randn(g, rank, rank, 3, 3), _randn(g, rank, rank, 3, 3)
    w1_a, w1_b, w2_a, w2_b = _randn(g, rank, 8), _randn(g, rank, 6), _randn(g, rank, 8), _randn(g, rank, 6)
    first = torch.einsum("ijkl,jr,ip->prkl", t1, w1_b, w1_a)
    second = torch.einsum("ijkl,jr,ip->prkl", t2, w2_b, w2_a)
    sd = {
        "hada_w1_a": w1_a,
        "hada_w1_b": w1_b,
        "hada_w2_a": w2_a,
        "hada_w2_b": w2_b,
        "hada_t1": t1,
        "hada_t2": t2,
        "alpha": torch.tensor(2.0, dtype=F64),
    }
    return Case(sd, (8, 6, 3, 3), first * second * (2.0 / rank))


CASES = {
    "lokr_whole": _lokr_whole,
    "lokr_factorized": _lokr_factorized,
    "lokr_conv": _lokr_conv,
    "lokr_tucker": _lokr_tucker,
    "loha": _loha,
    "loha_tucker": _loha_tucker,
}


def _magnitude(g: torch.Generator, weight_shape: tuple[int, ...], *, out_axis: bool) -> torch.Tensor:
    """LyCORIS' shapes: (out, 1[, 1, 1]) on the output axis, (1, in[, 1, 1]) on the input axis."""
    trailing = [1] * (len(weight_shape) - 2)
    shape = (weight_shape[0], 1, *trailing) if out_axis else (1, weight_shape[1], *trailing)
    return torch.rand(*shape, generator=g, dtype=F64) + 0.5


def _reference_residual(
    w: torch.Tensor, update: torch.Tensor, magnitude: torch.Tensor, *, out_axis: bool
) -> torch.Tensor:
    direction = w + update
    reduced = [d for d in range(direction.dim()) if d != (0 if out_axis else 1)]
    norm = torch.sqrt((direction**2).sum(dim=reduced, keepdim=True))
    return magnitude * direction / norm - w


def _build(case_name: str, *, out_axis: bool, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    case = CASES[case_name](g)
    magnitude = _magnitude(g, case.weight_shape, out_axis=out_axis)
    weight = _randn(g, *case.weight_shape)
    layer = any_lora_layer_from_state_dict({**case.state_dict, "dora_scale": magnitude})
    expected = _reference_residual(weight, case.update, magnitude, out_axis=out_axis)
    return layer, weight, expected


@pytest.mark.parametrize("out_axis", [True, False], ids=["out_axis", "in_axis"])
@pytest.mark.parametrize("case_name", CASES)
def test_the_residual_is_the_lycoris_weight_decomposition(case_name: str, out_axis: bool) -> None:
    layer, weight, expected = _build(case_name, out_axis=out_axis)

    assert isinstance(layer, FactorizedDoRALayer)
    assert layer.magnitude_is_out_dim is out_axis
    # The LoRA weight multiplies the residual, as it does for a low-rank DoRA.
    params = layer.get_parameters({"weight": weight}, weight=0.6)
    torch.testing.assert_close(params["weight"], expected * 0.6, rtol=1e-9, atol=1e-9)
    assert params["weight"].shape == weight.shape


@pytest.mark.parametrize("case_name", ["lokr_whole", "loha"])
def test_a_peft_style_output_axis_magnitude_is_read_per_row(case_name: str) -> None:
    g = torch.Generator().manual_seed(1)
    case = CASES[case_name](g)
    magnitude = torch.rand(case.weight_shape[0], generator=g, dtype=F64) + 0.5
    weight = _randn(g, *case.weight_shape)

    layer = any_lora_layer_from_state_dict({**case.state_dict, "dora_magnitude": magnitude})

    expected = _reference_residual(weight, case.update, magnitude[:, None], out_axis=True)
    torch.testing.assert_close(layer.get_parameters({"weight": weight}, 1.0)["weight"], expected)


def test_the_magnitude_is_taken_of_the_updated_weight_not_the_original() -> None:
    """ComfyUI normalizes by the norm of W; LyCORIS (and DoRALayer) by the norm of W + dW."""
    layer, weight, expected = _build("lokr_whole", out_axis=True)
    magnitude = layer.dora_scale
    wrong = (
        magnitude
        * (weight + _lokr_whole(torch.Generator().manual_seed(0)).update)
        / torch.linalg.norm(weight, dim=1, keepdim=True)
        - weight
    )

    actual = layer.get_parameters({"weight": weight}, 1.0)["weight"]
    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(actual, wrong, atol=1e-3)


def test_a_bias_is_scaled_by_alpha_over_rank_and_the_lora_weight() -> None:
    g = torch.Generator().manual_seed(2)
    case = _lokr_factorized(g)
    bias = _randn(g, 12)
    layer = any_lora_layer_from_state_dict(
        {**case.state_dict, "dora_scale": _magnitude(g, case.weight_shape, out_axis=True)}
    )
    assert isinstance(layer, FactorizedDoRALayer)
    assert isinstance(layer.layer, LoKRLayer)
    layer.layer.bias = bias

    params = layer.get_parameters({"weight": _randn(g, *case.weight_shape), "bias": torch.zeros(12, dtype=F64)}, 0.5)

    torch.testing.assert_close(params["bias"], bias * (5.0 / 2) * 0.5)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_factors_and_weight_are_promoted_to_float32(dtype: torch.dtype) -> None:
    """A small update on a half-precision weight is not rounded away: the residual is formed in float32."""
    g = torch.Generator().manual_seed(3)
    w1, w2 = _randn(g, 3, 4) * 0.01, _randn(g, 4, 2) * 0.01
    magnitude = torch.rand(12, 1, generator=g, dtype=F64) + 0.5
    weight = _randn(g, 12, 8)
    layer = any_lora_layer_from_state_dict({"lokr_w1": w1, "lokr_w2": w2, "dora_scale": magnitude})
    layer.to(dtype=dtype)
    half_weight = weight.to(dtype)

    params = layer.get_parameters({"weight": half_weight}, 1.0)

    assert params["weight"].dtype == torch.float32
    # The reference sees the same rounded inputs.
    update = _kron(w1.to(dtype).to(F64), w2.to(dtype).to(F64))
    expected = _reference_residual(half_weight.to(F64), update, magnitude, out_axis=True)
    torch.testing.assert_close(params["weight"].to(F64), expected, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_the_magnitude_stays_float32_beside_low_precision_factors(dtype: torch.dtype) -> None:
    """The residual is the small difference between two nearly equal magnitudes, so a magnitude rounded to the model's
    dtype would erase it: with a magnitude within 1% of the norm of the weight and an update of 1% of it, rounding it
    to bfloat16 moves the residual by about its own size."""
    g = torch.Generator().manual_seed(11)
    w1, w2 = _randn(g, 8, 8), _randn(g, 8, 6)
    weight = _randn(g, 64, 48)
    w1 = w1 * (0.01 * weight.norm() / _kron(w1, w2).norm())
    half_weight = weight.to(dtype)
    update = _kron(w1.to(dtype).to(F64), w2.to(dtype).to(F64))
    direction = half_weight.to(F64) + update
    magnitude = direction.norm(dim=1, keepdim=True) * (1 + 0.01 * (torch.rand(64, 1, generator=g, dtype=F64) - 0.5))
    layer = any_lora_layer_from_state_dict({"lokr_w1": w1, "lokr_w2": w2, "dora_scale": magnitude})
    layer.to(dtype=dtype)

    params = layer.get_parameters({"weight": half_weight}, 1.0)

    assert layer.dora_scale.dtype == torch.float32
    expected = _reference_residual(half_weight.to(F64), update, magnitude, out_axis=True)
    torch.testing.assert_close(params["weight"].to(F64), expected, rtol=0, atol=1e-4 * float(expected.abs().max()))


def test_moving_the_layer_does_not_move_a_shared_copy() -> None:
    layer, *_ = _build("lokr_whole", out_axis=True)
    assert isinstance(layer, FactorizedDoRALayer)
    clone = copy.copy(layer)
    clone.to(dtype=torch.float32)

    assert layer.dora_scale.dtype == F64
    assert isinstance(layer.layer, LoKRLayer) and layer.layer.w1 is not None and layer.layer.w1.dtype == F64


def test_an_unreadable_weight_applies_the_plain_update_and_warns_once_per_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """bitsandbytes weights have no readable values (they arrive as meta tensors). A LoKr/LoHa with a magnitude used to
    apply as the plain layer there and still does, with one warning however many layers and forwards."""
    warning = MagicMock()
    monkeypatch.setattr(dora_module.logger, "warning", warning)
    monkeypatch.setattr(dora_module, "_magnitude_unreadable_warned", False)
    for case_name in ("lokr_factorized", "loha", "lokr_factorized"):
        g = torch.Generator().manual_seed(7)
        case = CASES[case_name](g)
        layer = any_lora_layer_from_state_dict(
            {**case.state_dict, "dora_scale": _magnitude(g, case.weight_shape, out_axis=True)}
        )
        meta_weight = torch.empty(case.weight_shape, device="meta")

        params = layer.get_parameters({"weight": meta_weight}, weight=0.6)

        torch.testing.assert_close(params["weight"], case.update * 0.6)
    warning.assert_called_once()
    assert "bitsandbytes" in warning.call_args.args[0]


def test_a_low_rank_dora_still_refuses_an_unreadable_weight() -> None:
    dora = DoRALayer(
        up=torch.randn(4, 2),
        down=torch.randn(2, 3),
        dora_scale=torch.ones(4),
        alpha=None,
        bias=None,
        magnitude_is_out_dim=True,
    )

    with pytest.raises(RuntimeError, match="not compatible with DoRA patches"):
        dora.get_parameters({"weight": torch.empty(4, 3, device="meta")}, 1.0)


def test_a_magnitude_less_layer_stays_a_plain_lokr_or_loha() -> None:
    g = torch.Generator().manual_seed(4)
    lokr = any_lora_layer_from_state_dict(_lokr_whole(g).state_dict)
    loha = any_lora_layer_from_state_dict(_loha(g).state_dict)

    assert type(lokr) is LoKRLayer and type(loha) is LoHALayer
    assert not reads_weight_values(lokr) and not reads_weight_values(loha)


def test_wrapped_decomposed_layers_read_the_weight_and_apply_to_their_own_rows() -> None:
    top, w_top, expected_top = _build("lokr_whole", out_axis=True, seed=5)
    bottom, w_bottom, expected_bottom = _build("loha", out_axis=False, seed=6)
    full_weight = torch.cat([w_top, w_bottom])
    merged = MergedLayerPatch([top, bottom], [Range(0, 12), Range(12, 24)])

    assert reads_weight_values(merged)
    assert reads_weight_values(SwappedHalvesLayerPatch(top))
    params = merged.get_parameters({"weight": full_weight}, 1.0)
    torch.testing.assert_close(params["weight"], torch.cat([expected_top, expected_bottom]))


def test_calc_size_and_to_cover_the_factors_and_the_magnitude() -> None:
    g = torch.Generator().manual_seed(0)
    case = _lokr_factorized(g)
    magnitude = _magnitude(g, case.weight_shape, out_axis=True)
    state_dict = {**case.state_dict, "dora_scale": magnitude}
    layer = any_lora_layer_from_state_dict(state_dict)
    assert isinstance(layer, FactorizedDoRALayer)

    expected_bytes = sum(t.numel() * t.element_size() for key, t in state_dict.items() if key != "alpha")
    assert layer.calc_size() == expected_bytes

    layer.to(dtype=torch.float32)
    assert layer.calc_size() == expected_bytes // 2
    assert layer.dora_scale.dtype == torch.float32


@pytest.mark.parametrize("module_kind", ["linear", "conv2d"])
@pytest.mark.parametrize("out_axis", [True, False], ids=["out_axis", "in_axis"])
def test_a_custom_module_sidecar_applies_the_decomposition(module_kind: str, out_axis: bool) -> None:
    case_name = "lokr_whole" if module_kind == "linear" else "lokr_conv"
    layer, weight, expected = _build(case_name, out_axis=out_axis)
    weight = weight.float()
    layer.to(dtype=torch.float32)
    if module_kind == "linear":
        module: torch.nn.Module = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        x = torch.randn(4, weight.shape[1])
    else:
        module = torch.nn.Conv2d(weight.shape[1], weight.shape[0], 3, padding=1, bias=False)
        x = torch.randn(2, weight.shape[1], 5, 5)
    module.weight.data = weight.clone()
    model = torch.nn.Module()
    model.proj = module
    apply_custom_layers_to_model(model)

    patches = [(ModelPatchRaw(layers={"p-proj": layer}), 0.7)]
    with LayerPatcher.apply_smart_model_patches(
        model=model, patches=patches, prefix="p-", dtype=torch.float32, force_sidecar_patching=True
    ):
        patched = model.proj(x)

    expected_weight = (weight + 0.7 * expected.float()).detach()
    reference = (
        torch.nn.functional.linear(x, expected_weight)
        if module_kind == "linear"
        else torch.nn.functional.conv2d(x, expected_weight, padding=1)
    )
    torch.testing.assert_close(patched, reference, rtol=1e-4, atol=1e-4)


def test_direct_patching_merges_the_residual_and_restores_the_weight() -> None:
    layer, weight, expected = _build("loha", out_axis=False)
    weight = weight.float()
    layer.to(dtype=torch.float32)
    model = torch.nn.Module()
    model.proj = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
    model.proj.weight.data = weight.clone()

    patches = [(ModelPatchRaw(layers={"p-proj": layer}), 1.0)]
    with LayerPatcher.apply_smart_model_patches(
        model=model, patches=patches, prefix="p-", dtype=torch.float32, force_direct_patching=True
    ):
        torch.testing.assert_close(model.proj.weight.data, weight + expected.float(), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(model.proj.weight.data, weight)


def _lokr_square(g: torch.Generator) -> Case:
    w1, w2 = _randn(g, 3, 3), _randn(g, 4, 4)
    return Case({"lokr_w1": w1, "lokr_w2": w2}, (12, 12), _kron(w1, w2))


def test_a_1d_magnitude_of_the_output_length_is_read_per_row() -> None:
    """A vector with one entry per output row, whose length is not the input size, is output-axis, as a PEFT magnitude
    is. Read per column it would not even broadcast onto the weight."""
    g = torch.Generator().manual_seed(21)
    case = _lokr_whole(g)  # 12 x 8
    magnitude = torch.rand(12, generator=g, dtype=F64) + 0.5
    weight = _randn(g, *case.weight_shape)

    layer = any_lora_layer_from_state_dict({**case.state_dict, "dora_scale": magnitude})

    assert isinstance(layer, FactorizedDoRALayer) and layer.magnitude_is_out_dim
    expected = _reference_residual(weight, case.update, magnitude[:, None], out_axis=True)
    torch.testing.assert_close(layer.get_parameters({"weight": weight}, 1.0)["weight"], expected)


def test_a_1d_magnitude_of_the_input_length_is_read_per_column() -> None:
    g = torch.Generator().manual_seed(22)
    case = _lokr_whole(g)  # 12 x 8
    magnitude = torch.rand(8, generator=g, dtype=F64) + 0.5
    weight = _randn(g, *case.weight_shape)

    layer = any_lora_layer_from_state_dict({**case.state_dict, "dora_scale": magnitude})

    assert isinstance(layer, FactorizedDoRALayer) and not layer.magnitude_is_out_dim
    expected = _reference_residual(weight, case.update, magnitude[None, :], out_axis=False)
    torch.testing.assert_close(layer.get_parameters({"weight": weight}, 1.0)["weight"], expected)


def test_a_1d_magnitude_of_a_square_layer_is_ignored_with_one_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    """The length of a 1-D vector on a square layer fits both axes and nothing in the file says which it indexes. No
    known writer emits one for a LoKr or LoHa, so the layer applies as the plain LoKr, as before magnitudes were read,
    rather than on a guessed axis; the warning is once per process."""
    warning = MagicMock()
    monkeypatch.setattr(dora_module.logger, "warning", warning)
    monkeypatch.setattr(dora_module, "_magnitude_ambiguous_warned", False)
    g = torch.Generator().manual_seed(23)
    case = _lokr_square(g)
    weight = _randn(g, *case.weight_shape)

    layers = [
        any_lora_layer_from_state_dict({**case.state_dict, "dora_scale": torch.rand(12, generator=g, dtype=F64) + 0.5})
        for _ in range(2)
    ]

    assert all(type(layer) is LoKRLayer for layer in layers)
    torch.testing.assert_close(layers[0].get_parameters({"weight": weight}, 0.5)["weight"], case.update * 0.5)
    warning.assert_called_once()
    assert "12x12" in warning.call_args.args[0]


@pytest.mark.parametrize("key", ["dora_scale", "dora_magnitude"])
@pytest.mark.parametrize(
    "shape", [(5, 1), (1, 7), (12, 8), (13,), (12, 2), (2, 12), (1,), (), (1, 4), (1, 2), (1, 1), (1, 8, 1, 1)]
)
def test_a_magnitude_that_fits_neither_axis_is_refused_at_load(key: str, shape: tuple[int, ...]) -> None:
    """Applied, such a magnitude fails in the middle of a generation, or a full (out, in) one is multiplied in as an
    elementwise factor. The weight is 12 x 8."""
    case = _lokr_whole(torch.Generator().manual_seed(24))

    with pytest.raises(ValueError, match="fits neither axis"):
        any_lora_layer_from_state_dict({**case.state_dict, key: torch.ones(shape, dtype=F64)})


@pytest.mark.parametrize("out_axis", [True, False], ids=["out_axis", "in_axis"])
@pytest.mark.parametrize("case_name", ["lokr_whole", "loha"])
def test_a_row_of_zero_norm_gives_zero_not_nan_at_any_strength(case_name: str, out_axis: bool) -> None:
    """A base row (or column) of zeros that the update leaves untouched has no norm to divide by; its residual is 0
    rather than NaN, which would survive a strength of 0 as NaN * 0."""
    g = torch.Generator().manual_seed(25)
    case = CASES[case_name](g)
    sd = {key: value.clone() for key, value in case.state_dict.items()}
    weight = _randn(g, *case.weight_shape)
    if case_name == "lokr_whole":
        # kron(w1, w2): a zero first row of w1 zeroes the first 4 rows, a zero first column the first 2 columns.
        if out_axis:
            sd["lokr_w1"][0] = 0
            weight[:4] = 0
        else:
            sd["lokr_w1"][:, 0] = 0
            weight[:, :2] = 0
    elif out_axis:
        sd["hada_w1_a"][0] = 0
        weight[0] = 0
    else:
        sd["hada_w1_b"][:, 0] = 0
        weight[:, 0] = 0
    layer = any_lora_layer_from_state_dict({**sd, "dora_scale": _magnitude(g, case.weight_shape, out_axis=out_axis)})

    for strength in (1.0, 0.0):
        residual = layer.get_parameters({"weight": weight}, strength)["weight"]
        assert torch.isfinite(residual).all()
        zeroed = {("lokr_whole", True): 4, ("lokr_whole", False): 2, ("loha", True): 1, ("loha", False): 1}
        count = zeroed[case_name, out_axis]
        zero_part = residual[:count] if out_axis else residual[:, :count]
        assert torch.equal(zero_part, torch.zeros_like(zero_part))


def _conv_state_dict(variant: str, out_channels: int) -> dict[str, torch.Tensor]:
    """A LoKr of a 3x3 conv from 8 channels: with a Tucker core (its factors show the kernel), or without one, where
    `w2_b` holds the flattened kernel and a Linear's factors look the same."""
    g = torch.Generator().manual_seed(31)
    rank, w1_rows = 2, out_channels // 4
    if variant == "tucker":
        return {
            "lokr_w1": _randn(g, w1_rows, 2),
            "lokr_t2": _randn(g, rank, rank, 3, 3),
            "lokr_w2_a": _randn(g, rank, 4),
            "lokr_w2_b": _randn(g, rank, 4),
        }
    return {"lokr_w1": _randn(g, w1_rows, 2), "lokr_w2_a": _randn(g, 4, rank), "lokr_w2_b": _randn(g, rank, 4 * 9)}


@pytest.mark.parametrize(
    ("variant", "shape"),
    [
        ("flat", (1, 9, 1, 1)),
        ("flat", (1, 24, 1, 1)),
        ("tucker", (8,)),
        ("tucker", (1, 8)),
        ("tucker", (1, 72, 1, 1)),
    ],
)
def test_a_conv_magnitude_that_is_not_one_of_its_layouts_is_refused(variant: str, shape: tuple[int, ...]) -> None:
    """On a conv of 8 channels (8 -> 16) a per-column magnitude is `(1, 8, 1, 1)`, whatever the factors: divisors of
    the flattened columns, a 2-D one and, where the factors show the kernel, a 1-D one do not fit."""
    state_dict = _conv_state_dict(variant, 16)

    with pytest.raises(ValueError, match="fits neither axis"):
        any_lora_layer_from_state_dict({**state_dict, "dora_scale": torch.ones(shape, dtype=F64)})


@pytest.mark.parametrize("variant", ["flat", "tucker"])
@pytest.mark.parametrize("shape", [(16,), (16, 1, 1, 1), (1, 8, 1, 1)])
def test_a_conv_magnitude_in_a_lycoris_layout_loads_and_applies(variant: str, shape: tuple[int, ...]) -> None:
    # Without a Tucker core the 8 channels and the 3x3 kernel are the 72 flattened columns of the factors.
    state_dict = _conv_state_dict(variant, 16)
    layer = any_lora_layer_from_state_dict({**state_dict, "dora_scale": torch.rand(shape, dtype=F64) + 0.5})

    assert isinstance(layer, FactorizedDoRALayer)
    params = layer.get_parameters({"weight": _randn(torch.Generator().manual_seed(32), 16, 8, 3, 3)}, 1.0)
    assert torch.isfinite(params["weight"]).all()


def test_a_square_conv_1d_magnitude_falls_back_whether_or_not_the_kernel_shows() -> None:
    """Tucker or not, an 8-entry vector on an 8 -> 8 conv fits both axes of the layer's channels."""
    layer = any_lora_layer_from_state_dict({**_conv_state_dict("tucker", 8), "dora_scale": torch.ones(8, dtype=F64)})

    assert type(layer) is LoKRLayer
