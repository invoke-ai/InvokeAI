import pytest
import torch
from safetensors.torch import load, save

from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
    apply_custom_layers_to_model,
)
from invokeai.backend.model_manager.taxonomy import ModelFormat
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.patches.layers.dora_layer import DoRALayer
from invokeai.backend.patches.lora_conversions.flux_bfl_peft_lora_conversion_utils import (
    convert_bfl_lora_patch_to_diffusers,
    lora_model_from_flux2_bfl_peft_state_dict,
    lora_model_from_flux_bfl_peft_state_dict,
)
from invokeai.backend.patches.lora_conversions.flux_kohya_lora_conversion_utils import (
    lora_model_from_flux_kohya_state_dict,
)
from invokeai.backend.patches.lora_conversions.flux_lora_constants import FLUX_LORA_TRANSFORMER_PREFIX
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.quantization.dequantizing_linear import requires_sidecar_patching
from invokeai.backend.quantization.int8_convrot import Int8ConvrotLinear
from invokeai.backend.quantization.nvfp4 import NVFP4Linear


def _convert(source: str, stream: str, values: dict[str, torch.Tensor]) -> ModelPatchRaw:
    if source == "kohya":
        state = {f"lora_unet_double_blocks_0_{stream}_attn_qkv.{key}": value for key, value in values.items()}
        return convert_bfl_lora_patch_to_diffusers(lora_model_from_flux_kohya_state_dict(load(save(state))))

    suffixes = {"lora_up.weight": "lora_B.weight", "lora_down.weight": "lora_A.weight"}
    state = {
        f"diffusion_model.double_blocks.0.{stream}_attn.qkv.{suffixes.get(key, key)}": value
        for key, value in values.items()
    }
    state = load(save(state))
    if source == "bfl-built":
        return convert_bfl_lora_patch_to_diffusers(lora_model_from_flux_bfl_peft_state_dict(state))
    assert source == "bfl"
    return lora_model_from_flux2_bfl_peft_state_dict(state)


@pytest.fixture
def dora_values() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(10191)
    return {
        "lora_up.weight": torch.randn(12, 2, generator=generator),
        "lora_down.weight": torch.randn(2, 7, generator=generator),
        "dora_scale": torch.linspace(0.5, 2.0, 12).reshape(12, 1),
        "alpha": torch.tensor(3.0),
        "bias_indices": torch.tensor([[1, 4, 6, 11]]),
        "bias_values": torch.tensor([0.2, -0.3, 0.7, -0.8]),
        "bias_size": torch.tensor([12]),
    }


class _SplitQKV(torch.nn.Module):
    def __init__(self, stream: str, in_features: int):
        super().__init__()
        self.names = ("to_q", "to_k", "to_v") if stream == "img" else ("add_q_proj", "add_k_proj", "add_v_proj")
        attention = torch.nn.ModuleDict({name: torch.nn.Linear(in_features, 4) for name in self.names})
        self.transformer_blocks = torch.nn.ModuleList([torch.nn.ModuleDict({"attn": attention})])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = self.transformer_blocks[0]["attn"]
        return torch.cat([attention[name](x) for name in self.names], dim=-1)


@torch.no_grad()
@pytest.mark.parametrize("source", ["kohya", "bfl", "bfl-built"])
@pytest.mark.parametrize("stream", ["img", "txt"])
@pytest.mark.parametrize("sidecar", [False, True], ids=["direct", "sidecar"])
@pytest.mark.parametrize("strength", [-0.4, 0.65, 1.0])
def test_output_axis_dora_matches_full_fused_forward_and_restores(
    source: str, stream: str, sidecar: bool, strength: float, dora_values: dict[str, torch.Tensor]
) -> None:
    """Compare real patch application to the unsplit formula, including sparse bias and alpha/rank scaling."""
    patch = _convert(source, stream, dora_values)
    model = _SplitQKV(stream, in_features=7)
    if sidecar:
        apply_custom_layers_to_model(model)
    attention = model.transformer_blocks[0]["attn"]
    generator = torch.Generator().manual_seed(200)
    # Different Q/K/V base weights matter: DoRA's normalization depends on the base, not just the adapter.
    base_weight = torch.randn(12, 7, generator=generator)
    base_bias = torch.randn(12, generator=generator)
    for index, name in enumerate(model.names):
        attention[name].weight.copy_(base_weight[index * 4 : (index + 1) * 4])
        attention[name].bias.copy_(base_bias[index * 4 : (index + 1) * 4])
    original = {name: value.clone() for name, value in model.state_dict().items()}
    x = torch.randn(2, 3, 7, generator=generator)
    baseline = model(x)

    # Independent full fused DoRA formula: no split patch/get_weight/get_parameters in the oracle.
    scale = 3.0 / 2
    direction = base_weight + scale * (dora_values["lora_up.weight"] @ dora_values["lora_down.weight"])
    normalized = direction * dora_values["dora_scale"] / torch.linalg.vector_norm(direction, dim=1, keepdim=True)
    expected_weight = base_weight + strength * (normalized - base_weight)
    bias_delta = torch.zeros(12)
    bias_delta[dora_values["bias_indices"][0]] = dora_values["bias_values"]
    expected = torch.nn.functional.linear(x, expected_weight, base_bias + strength * scale * bias_delta)
    assert not torch.allclose(expected, baseline, atol=1e-3)
    assert set(patch.layers) == {f"{FLUX_LORA_TRANSFORMER_PREFIX}transformer_blocks.0.attn.{n}" for n in model.names}
    assert all(isinstance(layer, DoRALayer) and layer.magnitude_is_out_dim for layer in patch.layers.values())

    # Exercise normal cleanup and exceptional cleanup on the same real modules.
    for fail in (False, True):
        try:
            with LayerPatcher.apply_smart_model_patches(
                model=model,
                patches=[(patch, strength)],
                prefix=FLUX_LORA_TRANSFORMER_PREFIX,
                dtype=torch.float32,
                force_direct_patching=not sidecar,
                force_sidecar_patching=sidecar,
            ):
                torch.testing.assert_close(model(x), expected, atol=3e-6, rtol=3e-6)
                if fail:
                    raise RuntimeError("inference interrupted")
        except RuntimeError as error:
            assert fail and str(error) == "inference interrupted"
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, original[name], rtol=0, atol=0)
        torch.testing.assert_close(model(x), baseline, rtol=0, atol=0)
        if sidecar:
            assert all(attention[name].get_num_patches() == 0 for name in model.names)


@pytest.mark.parametrize("source", ["kohya", "bfl", "bfl-built"])
@pytest.mark.parametrize("stream", ["img", "txt"])
@pytest.mark.parametrize("in_features", [7, 12], ids=["rectangular", "square-fused"])
@pytest.mark.parametrize("vector", [False, True], ids=["row", "1d"])
def test_input_axis_dora_refuses_instead_of_silently_dropping(
    source: str, stream: str, in_features: int, vector: bool, dora_values: dict[str, torch.Tensor]
) -> None:
    dora_values["lora_down.weight"] = torch.ones(2, in_features)
    dora_values["dora_scale"] = torch.ones(in_features) if vector else torch.ones(1, in_features)
    with pytest.raises(ValueError, match="input-axis.*fused QKV DoRA") as error:
        _convert(source, stream, dora_values)
    assert "output-axis" in str(error.value) and "plain LoRA" in str(error.value)
    assert "transformer_blocks.0.attn." in str(error.value)


@pytest.mark.parametrize("source", ["kohya", "bfl"])
@pytest.mark.parametrize("in_features", [7, 12])
def test_output_axis_orientation_survives_square_fused_weights_and_trailing_singletons(
    source: str, in_features: int, dora_values: dict[str, torch.Tensor]
) -> None:
    dora_values["lora_down.weight"] = torch.ones(2, in_features)
    dora_values["dora_scale"] = dora_values["dora_scale"].reshape(12, 1, 1, 1)
    patch = _convert(source, "txt", dora_values)
    assert len(patch.layers) == 3
    for index, layer in enumerate(patch.layers.values()):
        assert isinstance(layer, DoRALayer) and layer.magnitude_is_out_dim
        torch.testing.assert_close(
            layer.dora_scale.flatten(), torch.linspace(0.5, 2.0, 12)[index * 4 : (index + 1) * 4]
        )


def test_explicit_output_magnitude_vector_stays_output_axis(dora_values: dict[str, torch.Tensor]) -> None:
    # The built-patch route also receives PEFT/ai-toolkit magnitudes, whose 1-D convention is output-axis.
    dora_values["dora_magnitude"] = dora_values.pop("dora_scale").flatten()
    layer = DoRALayer.from_state_dict_values(dora_values)
    source = ModelPatchRaw({f"{FLUX_LORA_TRANSFORMER_PREFIX}double_blocks.0.img_attn.qkv": layer})
    patch = convert_bfl_lora_patch_to_diffusers(source)
    assert len(patch.layers) == 3
    for index, split in enumerate(patch.layers.values()):
        assert isinstance(split, DoRALayer) and split.magnitude_is_out_dim
        assert split.down is layer.down
        assert split.up.untyped_storage().data_ptr() == layer.up.untyped_storage().data_ptr()
        torch.testing.assert_close(split.dora_scale, layer.dora_scale[index * 4 : (index + 1) * 4])


@pytest.mark.parametrize("source", ["kohya", "bfl"])
@pytest.mark.parametrize(
    ("key", "shape", "message"),
    [
        ("lora_up.weight", (10, 2), "output.*divisible by 3"),
        ("lora_up.weight", (0, 2), "output.*divisible by 3"),
        ("lora_up.weight", (12, 3), "factors"),
        ("lora_up.weight", (12, 2, 1), "factors"),
        ("lora_down.weight", (2, 7, 1), "factors"),
        ("lora_down.weight", (0, 7), "factors"),
        ("lora_down.weight", (2, 0), "factors"),
        ("dora_scale", (11, 1), "magnitude"),
        ("dora_scale", (12, 2), "magnitude"),
        ("dora_scale", (1, 6), "magnitude"),
        ("dora_scale", (), "magnitude"),
    ],
)
def test_malformed_fused_dora_fails_during_conversion(
    source: str, key: str, shape: tuple[int, ...], message: str, dora_values: dict[str, torch.Tensor]
) -> None:
    dora_values[key] = torch.ones(shape)
    with pytest.raises(ValueError, match=message):
        _convert(source, "img", dora_values)


@pytest.mark.parametrize("shape", [(11,), (12, 2)])
def test_malformed_fused_dora_bias_fails_during_conversion(
    shape: tuple[int, ...], dora_values: dict[str, torch.Tensor]
) -> None:
    layer = DoRALayer.from_state_dict_values(dora_values)
    layer.bias = torch.ones(shape)
    source = ModelPatchRaw({f"{FLUX_LORA_TRANSFORMER_PREFIX}double_blocks.0.img_attn.qkv": layer})
    with pytest.raises(ValueError, match="bias"):
        convert_bfl_lora_patch_to_diffusers(source)


@torch.no_grad()
@pytest.mark.parametrize("packed", ["int8", "nvfp4"])
def test_converted_dora_retains_packed_sidecar_refusal_and_cleanup(packed: str) -> None:
    """Conversion support does not imply packed-base support: sidecars expose only meta base weights."""
    values = {
        "lora_up.weight": torch.ones(384, 2),
        "lora_down.weight": torch.ones(2, 64),
        "dora_scale": torch.ones(384, 1),
    }
    patch = _convert("kohya", "img", values)
    assert len(patch.layers) == 3
    model = _SplitQKV("img", in_features=64)
    attention = model.transformer_blocks[0]["attn"]
    for name in model.names:
        if packed == "int8":
            attention[name] = Int8ConvrotLinear(
                torch.ones(128, 64, dtype=torch.int8), torch.ones(128, 1), convrot=False
            )
        else:
            attention[name] = NVFP4Linear(
                torch.full((128, 32), 0x22, dtype=torch.uint8),
                torch.ones(128, 4).to(torch.float8_e4m3fn),
                torch.tensor(1.0),
            )
    apply_custom_layers_to_model(model)
    original = {name: value.clone() for name, value in model.state_dict().items()}
    x = torch.ones(1, 64, dtype=torch.float32)
    baseline = model(x)
    # This is the decision and patch setup used by FLUX2DenoiseInvocation.
    force_sidecar = requires_sidecar_patching(model, ModelFormat.Checkpoint)
    assert force_sidecar
    with pytest.raises(RuntimeError, match="base model quantization format.*not compatible with DoRA"):
        with LayerPatcher.apply_smart_model_patches(
            model=model,
            patches=[(patch, 0.7)],
            prefix=FLUX_LORA_TRANSFORMER_PREFIX,
            dtype=torch.float32,
            force_sidecar_patching=force_sidecar,
        ):
            model(x)
    assert all(attention[name].get_num_patches() == 0 for name in model.names)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value.float(), original[name].float(), rtol=0, atol=0)
    torch.testing.assert_close(model(x), baseline, rtol=0, atol=0)
