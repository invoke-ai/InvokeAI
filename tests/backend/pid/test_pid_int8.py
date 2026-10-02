"""Comfy-Org's `int8_tensorwise` PiD builds: the marked Linears stay int8 and decode like the dense ones.

`pid_1.5_*_int8_convrot` stores each patch- and pixel-block Linear as int8 codes rotated along the input dimension,
with a per-row float32 scale and a `comfy_quant` marker. The rotation group differs per layer — 256 for the 1536-wide
patch blocks, 64 for the 1152-wide pixel attention — so the tiny net below mixes two groups the same way. The
reference quantizes it the way the scheme is defined — a grouped matmul with the regular Hadamard (Kronecker powers of
the 4x4 seed), then symmetric per-row int8 — rather than with the code under test.
"""

import json
import re

import pytest
import torch

from invokeai.backend.model_manager.taxonomy import BaseModelType
from invokeai.backend.pid import decode as pid_decode_module
from invokeai.backend.pid._src.networks.pid_net import PidNet
from invokeai.backend.pid.decode import estimate_pid_decode_working_memory, load_pid_decoder
from invokeai.backend.quantization.int8_convrot import Int8ConvrotLinear

_SEED = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float32) / 2
# Normalized regular Hadamards: symmetric and orthonormal, so each is its own inverse.
_HADAMARDS = {4: _SEED, 16: torch.kron(_SEED, _SEED)}


def _tiny_net() -> PidNet:
    net = PidNet(
        in_channels=3,
        num_groups=2,
        hidden_size=16,
        pixel_hidden_size=4,
        pixel_attn_hidden_size=8,
        pixel_num_groups=2,
        patch_depth=2,
        pixel_depth=1,
        num_text_blocks=1,
        patch_size=2,
        txt_embed_dim=6,
        txt_max_length=4,
        rope_mode="original",
        rope_ref_h=4,
        rope_ref_w=4,
        lq_in_channels=0,
        lq_latent_channels=2,
        lq_hidden_dim=4,
        lq_num_res_blocks=1,
        lq_interval=1,
        sr_scale=1,
        latent_spatial_down_factor=2,
    ).eval()
    with torch.no_grad():
        # The backbone zero-inits its final layer, which would make every output zero.
        torch.nn.init.normal_(net.final_layer.linear.weight)
    return net


def _group_size(in_features: int) -> int | None:
    """The widest rotation group that divides the layer, as Comfy-Org picks it per layer."""
    return next((group for group in sorted(_HADAMARDS, reverse=True) if in_features % group == 0), None)


def _quantized_layers(net: PidNet) -> dict[str, int]:
    return {
        name: group
        for name, module in net.named_modules()
        if isinstance(module, torch.nn.Linear)
        and name.startswith(("patch_blocks.", "pixel_blocks."))
        and (group := _group_size(module.in_features)) is not None
    }


def _int8_checkpoint(net: PidNet) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """The net's state dict with its block Linears quantized, and the weights those codes decode to."""
    state_dict = {k: v.clone() for k, v in net.state_dict().items()}
    decoded = {}
    for layer, group in _quantized_layers(net).items():
        weight = state_dict[f"{layer}.weight"]
        out_features, in_features = weight.shape
        hadamard = _HADAMARDS[group]
        rotated = (weight.view(out_features, in_features // group, group) @ hadamard).view(weight.shape)
        scale = rotated.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127
        codes = torch.round(rotated / scale).clamp(-127, 127).to(torch.int8)
        marker = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": group}
        state_dict[f"{layer}.weight"] = codes
        state_dict[f"{layer}.weight_scale"] = scale
        state_dict[f"{layer}.comfy_quant"] = torch.frombuffer(bytearray(json.dumps(marker).encode()), dtype=torch.uint8)
        grouped = (codes.float() * scale).view(out_features, in_features // group, group)
        decoded[f"{layer}.weight"] = (grouped @ hadamard).view(weight.shape)
    return state_dict, decoded


def _forward(net: PidNet) -> torch.Tensor:
    torch.manual_seed(3)
    with torch.no_grad():
        return net(
            torch.randn(1, 3, 4, 4),
            torch.tensor([500.0]),
            torch.randn(1, 4, 6),
            lq_latent=torch.randn(1, 2, 2, 2),
            degrade_sigma=torch.tensor([0.3]),
        )


def test_an_int8_checkpoint_keeps_its_block_linears_int8_and_decodes_like_the_dense_net(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(0)
    source = _tiny_net()
    state_dict, decoded = _int8_checkpoint(source)
    reference = _tiny_net()
    reference.load_state_dict(source.state_dict() | decoded)
    monkeypatch.setattr(pid_decode_module, "build_pid_net", lambda backbone, version: _tiny_net())

    net = load_pid_decoder(dict(state_dict), BaseModelType.Flux)

    layers = _quantized_layers(source)
    assert set(layers.values()) == {4, 16}, "the fixture must mix rotation groups, as the released files do"
    for layer, group in layers.items():
        module = net.get_submodule(layer)
        assert isinstance(module, Int8ConvrotLinear), layer
        assert module.weight.dtype is torch.int8, layer
        assert module.group_size == group, layer
    torch.testing.assert_close(_forward(net), _forward(reference))


def test_a_purely_scaled_fp8_checkpoint_is_refused_too(monkeypatch: pytest.MonkeyPatch) -> None:
    """The same check, on a file with no int8 marker anywhere.

    It used to live inside `if int8_markers:`, so it only ever saw a *mixed* repack. A checkpoint
    that is scaled fp8 throughout produces no int8 markers at all -- `extract_int8_convrot_markers`
    takes only `int8_tensorwise`, and the unmarked-weight check filters on the int8 dtype -- so it
    walked past both gates and had its fp8 codes copied straight into the float32 parameters, off by
    `1/weight_scale`, with the orphaned scales reported only at DEBUG.
    """
    torch.manual_seed(0)
    state_dict, _ = _int8_checkpoint(_tiny_net())
    net = _tiny_net()
    for name in _quantized_layers(net):
        state_dict.pop(f"{name}.comfy_quant", None)
        state_dict[f"{name}.weight"] = state_dict[f"{name}.weight"].float().to(torch.float8_e4m3fn)
    monkeypatch.setattr(pid_decode_module, "build_pid_net", lambda backbone, version: net)

    with pytest.raises(ValueError, match=r"\.weight_scale"):
        load_pid_decoder(state_dict, BaseModelType.Flux)


def test_a_scale_from_another_scheme_is_refused_rather_than_loaded_unscaled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The int8 branch runs no fp8 pipeline, so a scaled-fp8 layer that came along in a mixed repack
    would have its weight copied into a float32 parameter *without* its scale -- off by
    `1/weight_scale` -- while the orphaned scale is swallowed by the `strict=False` load. This path
    ran every other step of the int8 install and not this check."""
    torch.manual_seed(0)
    state_dict, _ = _int8_checkpoint(_tiny_net())
    net = _tiny_net()
    # A Linear the net really has, so the check can tell it from a bundled submodel's keys.
    foreign = next(name for name in _quantized_layers(net) if f"{name}.comfy_quant" in state_dict)
    del state_dict[f"{foreign}.comfy_quant"]
    state_dict[f"{foreign}.weight"] = state_dict[f"{foreign}.weight"].float().to(torch.float8_e4m3fn)
    monkeypatch.setattr(pid_decode_module, "build_pid_net", lambda backbone, version: net)

    with pytest.raises(ValueError, match=rf"{re.escape(foreign)}\.weight_scale"):
        load_pid_decoder(state_dict, BaseModelType.Flux)


def test_an_int8_weight_no_marker_claims_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """Loaded as the float parameter it replaces, the raw codes would decode noise without any error — including
    when no marker at all is left to put the loader on its int8 path."""
    torch.manual_seed(0)
    state_dict, _ = _int8_checkpoint(_tiny_net())
    state_dict = {k: v for k, v in state_dict.items() if not k.endswith((".comfy_quant", ".weight_scale"))}
    monkeypatch.setattr(pid_decode_module, "build_pid_net", lambda backbone, version: _tiny_net())

    with pytest.raises(ValueError, match="no `comfy_quant` marker"):
        load_pid_decoder(state_dict, BaseModelType.Flux)


def test_the_working_memory_estimate_reserves_the_int8_dequant_transient_on_top() -> None:
    """The largest int8 Linear dequantizes at float32 — the dtype PidNet runs in — on top of the activations."""
    decoder = torch.nn.Module()
    decoder.small = Int8ConvrotLinear(torch.zeros(8, 4, dtype=torch.int8), torch.ones(8, 1), convrot=False)
    decoder.large = Int8ConvrotLinear(torch.zeros(16, 8, dtype=torch.int8), torch.ones(16, 1), convrot=False)
    latent = torch.zeros(1, 16, 64, 64)

    dense = estimate_pid_decode_working_memory(latent, BaseModelType.Flux)

    assert estimate_pid_decode_working_memory(latent, BaseModelType.Flux, pid_net=decoder) > dense
    assert estimate_pid_decode_working_memory(latent, BaseModelType.Flux, pid_net=torch.nn.Linear(4, 4)) == dense
