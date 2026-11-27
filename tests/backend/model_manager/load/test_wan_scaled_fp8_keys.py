"""What Comfy-Org's repackaged scaled-fp8 Wan 2.2 actually contains.

Wan's identification is *layout-sensitive* in a way the other architectures' is not. Five probes ask,
in order: is this a Wan transformer at all, is it Wan 2.1 rather than 2.2, is its layout native or
diffusers, which variant is it, and is it a main model rather than a LoRA. Every one of them reads
key names, and two read one tensor's extents. Until this capture all five were exercised only against
`_wan_a14b_native_state_dict()`, a synthetic dict written from the same understanding as the probes
themselves. A probe and its test agreeing about invented names is the circularity these captured
layouts exist to break.

So the cells here drive the real probes over the real released names. The scale spelling is *not*
what makes this fixture worth having: `flux1_transformer_scaled_fp8_keys.py` already records
`.scale_weight`/`.scale_input` and `test_flux1_scaled_fp8_keys.py` already drives a reader over it,
as does `test_scaled_fp8_fold_axis.py` for both spellings through the very fold Wan uses.

Captured over an HTTP range read: the build is ~14 GB and is not on this machine.
"""

import torch

from invokeai.backend.model_manager.configs.main import (
    _detect_wan_variant_from_state_dict,
    _find_wan_2_1_marker,
    _has_wan_keys,
    _has_wan_transformer_block_weights,
    _is_native_wan_layout,
)
from invokeai.backend.model_manager.taxonomy import WanVariantType
from invokeai.backend.quantization.fp8_scaled import COMFY_QUANT_SUFFIX, is_scale_metadata_key
from tests.backend.model_manager.load.state_dicts import wan22_t2v_14b_scaled_fp8_keys as fixture
from tests.backend.model_manager.load.state_dicts.utils import token_extents

KEYS = fixture.state_dict_keys

# `patch_embedding.weight` is the one tensor whose real extents a probe reads, so it keeps them;
# everything else is shrunk, because nothing under test looks at its size.
EXACT = ("patch_embedding.weight",)

DTYPES = {"F16": torch.float16, "F32": torch.float32, "F8_E4M3": torch.float8_e4m3fn, "BF16": torch.bfloat16}


def _state_dict() -> dict[str, torch.Tensor]:
    return {
        key: torch.zeros(shape if key in EXACT else token_extents(shape), dtype=DTYPES[dtype])
        for key, (shape, dtype) in KEYS.items()
    }


def test_the_released_names_are_recognised_as_a_wan_transformer_at_all() -> None:
    """The first question asked of the file, before layout or variant: is this a Wan transformer?

    It is answered by `patch_embedding.weight` plus a text-projection fingerprint, and the whole
    point of that pairing is not to collide with FLUX, Qwen-Image or Z-Image. Driven here over the
    released names rather than a dict written from the same understanding as the probe.
    """
    assert _has_wan_keys(_state_dict()) is True


def test_the_build_carries_none_of_the_wan_2_1_markers() -> None:
    """Wan 2.1 and 2.2 share a key layout and are separated only by architecture, so a 2.2 build has
    to be *negative* here -- a false marker would reject a supported file by name.

    This is also the second consumer of the exact `patch_embedding.weight` extent, which is why that
    one tensor keeps its real shape in `_state_dict`.
    """
    assert _find_wan_2_1_marker(_state_dict()) is None


def test_the_released_native_names_satisfy_the_layout_probe() -> None:
    """`_is_native_wan_layout` decides whether `_convert_wan_native_to_diffusers` runs at all.

    It keys on `text_embedding.0.weight`, and this file really does spell it that way -- under no
    prefix, as ComfyUI's repackaging ships it. A release that renamed that one key would send the
    file down the diffusers branch, where every key is then unexpected.
    """
    assert _is_native_wan_layout(_state_dict()) is True


def test_the_patch_embedding_shape_identifies_the_variant() -> None:
    """The variant comes from `patch_embedding.weight`'s `[inner_dim, in_channels, T, H, W]`, and
    getting it wrong is expensive rather than loud: a mislabelled variant pins the wrong expert,
    picks the wrong default settings, and hides the low-noise partner picker.

    This is the T2V A14B build, so 16 channels at 5120 wide. The shape is recorded from the file,
    and the probe requires *both* numbers precisely because in_channels alone is ambiguous across
    the wider Wan family.
    """
    assert KEYS["patch_embedding.weight"][0] == [5120, 16, 1, 2, 2]

    assert _detect_wan_variant_from_state_dict(_state_dict()) is WanVariantType.T2V_A14B


def test_a_real_block_weight_separates_the_build_from_a_wan_lora() -> None:
    """An I2V LoRA ships a full replacement `patch_embedding` and a `text_embedding`, so it matches
    the coarse Wan probes; an undecorated `blocks.0.self_attn.q.weight` is what says "main model".

    Pinned against the released names because a LoRA spells the same layer
    `...self_attn.q.lora_A.weight`, and the separation is an exact key match.
    """
    assert "blocks.0.self_attn.q.weight" in KEYS

    assert _has_wan_transformer_block_weights(_state_dict()) is True


def test_the_only_thing_the_build_declares_about_itself_is_the_bare_marker() -> None:
    """No `_quantization_metadata`, no per-layer `.comfy_quant`: one zero-element `scaled_fp8` key.

    So the fp8 branch is entered on the key names alone, and that single key has to be recognised as
    metadata -- left in the dict it is an unexpected key at `load_state_dict`, and it names no layer
    for a hint to hang on.
    """
    assert not hasattr(fixture, "layer_hints")
    assert not hasattr(fixture, "markers")
    assert not any(key.endswith(COMFY_QUANT_SUFFIX) for key in KEYS)

    assert KEYS["scaled_fp8"] == ([0], "F8_E4M3")
    assert is_scale_metadata_key("scaled_fp8")
