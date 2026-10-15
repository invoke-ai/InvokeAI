"""LTX-2 LoRA conversion: does a published LoRA land on the modules it names?

The conversion's whole job is a rename, and a rename that goes wrong does not raise -- it produces
layer paths that resolve to nothing, and the LoRA silently applies zero layers. So the test that
matters crosses the seam: it converts the paths a real published LoRA carries and resolves every
one of them against a real ``LTX2VideoTransformer3DModel``.

The paths below are every distinct layer family of the 2.5 distilled accelerator
(``ltx-2.5-22b-distilled-lora-450``, 1660 layers over 48 blocks), with block indices collapsed to
one representative each. The weights themselves are not needed and are not downloaded: the model
is instantiated on the meta device and the tensors are zero-storage expansions.
"""

import pytest
import torch

from invokeai.backend.patches.lora_conversions.ltx2_lora_constants import LTX2_LORA_TRANSFORMER_PREFIX
from invokeai.backend.patches.lora_conversions.ltx2_lora_conversion_utils import (
    is_state_dict_likely_in_ltx2_format,
    lora_model_from_ltx2_state_dict,
)

# (published path, rank, in_features, out_features) read from the released file.
PUBLISHED_LAYERS = (
    ("adaln_single.emb.timestep_embedder.linear_1", 256, 256, 4096),
    ("adaln_single.emb.timestep_embedder.linear_2", 450, 4096, 4096),
    ("adaln_single.linear", 450, 4096, 36864),
    ("audio_adaln_single.emb.timestep_embedder.linear_1", 256, 256, 2048),
    ("audio_adaln_single.emb.timestep_embedder.linear_2", 450, 2048, 2048),
    ("audio_adaln_single.linear", 450, 2048, 18432),
    ("audio_patchify_proj", 128, 128, 2048),
    ("audio_proj_out", 128, 2048, 128),
    ("audio_prompt_adaln_single.emb.timestep_embedder.linear_1", 256, 256, 2048),
    ("audio_prompt_adaln_single.emb.timestep_embedder.linear_2", 450, 2048, 2048),
    ("audio_prompt_adaln_single.linear", 450, 2048, 4096),
    ("av_ca_a2v_gate_adaln_single.emb.timestep_embedder.linear_1", 256, 256, 4096),
    ("av_ca_a2v_gate_adaln_single.emb.timestep_embedder.linear_2", 450, 4096, 4096),
    ("av_ca_a2v_gate_adaln_single.linear", 450, 4096, 4096),
    ("av_ca_audio_scale_shift_adaln_single.emb.timestep_embedder.linear_1", 256, 256, 2048),
    ("av_ca_audio_scale_shift_adaln_single.emb.timestep_embedder.linear_2", 450, 2048, 2048),
    ("av_ca_audio_scale_shift_adaln_single.linear", 450, 2048, 8192),
    ("av_ca_v2a_gate_adaln_single.emb.timestep_embedder.linear_1", 256, 256, 2048),
    ("av_ca_v2a_gate_adaln_single.emb.timestep_embedder.linear_2", 450, 2048, 2048),
    ("av_ca_v2a_gate_adaln_single.linear", 450, 2048, 2048),
    ("av_ca_video_scale_shift_adaln_single.emb.timestep_embedder.linear_1", 256, 256, 4096),
    ("av_ca_video_scale_shift_adaln_single.emb.timestep_embedder.linear_2", 450, 4096, 4096),
    ("av_ca_video_scale_shift_adaln_single.linear", 450, 4096, 16384),
    ("patchify_proj", 128, 128, 4096),
    ("proj_out", 128, 4096, 128),
    ("prompt_adaln_single.emb.timestep_embedder.linear_1", 256, 256, 4096),
    ("prompt_adaln_single.emb.timestep_embedder.linear_2", 450, 4096, 4096),
    ("prompt_adaln_single.linear", 450, 4096, 8192),
    ("transformer_blocks.0.attn1.to_gate_logits", 32, 4096, 32),
    ("transformer_blocks.0.attn1.to_k", 450, 4096, 4096),
    ("transformer_blocks.0.attn1.to_out.0", 450, 4096, 4096),
    ("transformer_blocks.0.attn1.to_q", 450, 4096, 4096),
    ("transformer_blocks.0.attn1.to_v", 450, 4096, 4096),
    ("transformer_blocks.0.attn2.to_gate_logits", 32, 4096, 32),
    ("transformer_blocks.0.attn2.to_k", 450, 4096, 4096),
    ("transformer_blocks.0.attn2.to_out.0", 450, 4096, 4096),
    ("transformer_blocks.0.attn2.to_q", 450, 4096, 4096),
    ("transformer_blocks.0.attn2.to_v", 450, 4096, 4096),
    ("transformer_blocks.0.audio_attn1.to_gate_logits", 32, 2048, 32),
    ("transformer_blocks.0.audio_attn1.to_k", 450, 2048, 2048),
    ("transformer_blocks.0.audio_attn1.to_out.0", 450, 2048, 2048),
    ("transformer_blocks.0.audio_attn1.to_q", 450, 2048, 2048),
    ("transformer_blocks.0.audio_attn1.to_v", 450, 2048, 2048),
    ("transformer_blocks.0.audio_attn2.to_gate_logits", 32, 2048, 32),
    ("transformer_blocks.0.audio_attn2.to_k", 450, 2048, 2048),
    ("transformer_blocks.0.audio_attn2.to_out.0", 450, 2048, 2048),
    ("transformer_blocks.0.audio_attn2.to_q", 450, 2048, 2048),
    ("transformer_blocks.0.audio_attn2.to_v", 450, 2048, 2048),
    ("transformer_blocks.0.audio_ff.net.0.proj", 450, 2048, 8192),
    ("transformer_blocks.0.audio_ff.net.2", 450, 8192, 2048),
    ("transformer_blocks.0.audio_to_video_attn.to_gate_logits", 32, 4096, 32),
    ("transformer_blocks.0.audio_to_video_attn.to_k", 450, 2048, 2048),
    ("transformer_blocks.0.audio_to_video_attn.to_out.0", 450, 2048, 4096),
    ("transformer_blocks.0.audio_to_video_attn.to_q", 450, 4096, 2048),
    ("transformer_blocks.0.audio_to_video_attn.to_v", 450, 2048, 2048),
    ("transformer_blocks.0.ff.net.0.proj", 450, 4096, 16384),
    ("transformer_blocks.0.ff.net.2", 450, 16384, 4096),
    ("transformer_blocks.0.video_to_audio_attn.to_gate_logits", 32, 2048, 32),
    ("transformer_blocks.0.video_to_audio_attn.to_k", 450, 4096, 2048),
    ("transformer_blocks.0.video_to_audio_attn.to_out.0", 450, 2048, 2048),
    ("transformer_blocks.0.video_to_audio_attn.to_q", 450, 2048, 2048),
    ("transformer_blocks.0.video_to_audio_attn.to_v", 450, 4096, 2048),
)

PUBLISHED_LAYER_PATHS = tuple(path for path, _rank, _in, _out in PUBLISHED_LAYERS)


def _transformer_modules() -> dict[str, torch.nn.Module]:
    from diffusers import LTX2VideoTransformer3DModel

    from invokeai.backend.ltx2 import component_configs as cc

    with torch.device("meta"):
        model = LTX2VideoTransformer3DModel(**cc.LTX2_5_TRANSFORMER_CONFIG)
    return dict(model.named_modules())


def _z(*shape: int) -> torch.Tensor:
    """A zero-storage stand-in, so full-width LoRA shapes cost nothing to hold."""
    return torch.zeros(1, dtype=torch.float32).expand(shape)


def _published_state_dict(prefix: str = "diffusion_model.") -> dict[str, torch.Tensor]:
    """A LoRA shaped exactly like the published one, at the shapes read from the release."""
    state_dict: dict[str, torch.Tensor] = {}
    for path, rank, in_features, out_features in PUBLISHED_LAYERS:
        state_dict[f"{prefix}{path}.lora_A.weight"] = _z(rank, in_features)
        state_dict[f"{prefix}{path}.lora_B.weight"] = _z(out_features, rank)
    return state_dict


def test_every_published_layer_resolves_to_a_module_on_the_transformer() -> None:
    """The failure this exists for is silent: a path that resolves to nothing applies nothing."""
    modules = _transformer_modules()
    patch = lora_model_from_ltx2_state_dict(_published_state_dict())

    assert len(patch.layers) == len(PUBLISHED_LAYER_PATHS)
    for name, layer in patch.layers.items():
        assert name.startswith(LTX2_LORA_TRANSFORMER_PREFIX), name
        path = name[len(LTX2_LORA_TRANSFORMER_PREFIX) :]
        module = modules.get(path)
        assert isinstance(module, torch.nn.Linear), f"{path!r} is not a Linear on the transformer"
        # The patch has to be applicable, not merely addressable.
        assert layer.up.shape[0] == module.out_features, path
        assert layer.down.shape[1] == module.in_features, path


def test_the_official_names_the_2_0_map_misses_are_still_renamed() -> None:
    """Diffusers' own converter is written for LTX-2.0 and leaves the 2.5-only modulation heads
    alone; this repo's `finish_transformer_keys` completes them. Driving only the diffusers half
    would leave these pointing at modules that do not exist."""
    patch = lora_model_from_ltx2_state_dict(_published_state_dict())
    renamed = {name[len(LTX2_LORA_TRANSFORMER_PREFIX) :] for name in patch.layers}

    assert "audio_prompt_adaln.linear" in renamed
    assert "prompt_adaln.linear" in renamed
    # And the ones diffusers does handle, so a regression in either half is visible here.
    assert "time_embed.linear" in renamed
    assert "av_cross_attn_video_a2v_gate.linear" in renamed
    assert "proj_in" in renamed


@pytest.mark.parametrize("prefix", ["", "diffusion_model.", "transformer.", "base_model.model.transformer."])
def test_the_peft_prefixes_seen_in_the_wild_all_strip(prefix: str) -> None:
    patch = lora_model_from_ltx2_state_dict(_published_state_dict(prefix=prefix))

    assert len(patch.layers) == len(PUBLISHED_LAYER_PATHS)


def test_an_unknown_layer_keeps_its_own_name_so_the_patcher_can_report_it() -> None:
    """A layer this architecture does not have must not be dropped and must not be renamed into
    something plausible: `LayerPatcher` logs the key it cannot resolve, and that log is only useful
    if the key is the one the file actually carried."""
    state_dict = _published_state_dict()
    state_dict["diffusion_model.transformer_blocks.0.invented_attn.to_q.lora_A.weight"] = _z(8, 4096)
    state_dict["diffusion_model.transformer_blocks.0.invented_attn.to_q.lora_B.weight"] = _z(4096, 8)

    patch = lora_model_from_ltx2_state_dict(state_dict)
    expected = f"{LTX2_LORA_TRANSFORMER_PREFIX}transformer_blocks.0.invented_attn.to_q"

    assert expected in patch.layers
    assert len(patch.layers) == len(PUBLISHED_LAYERS) + 1


def test_a_peft_lora_bias_file_converts_rather_than_being_refused() -> None:
    """PEFT's `lora_bias=True` emits a bias beside each pair. It is a real published shape, and the
    layer builder consumes it -- refusing it would reject a usable file at the point where the 22B
    transformer has already loaded."""
    state_dict = _published_state_dict()
    for path, _rank, _in, out_features in PUBLISHED_LAYERS[:3]:
        state_dict[f"diffusion_model.{path}.lora_B.bias"] = _z(out_features)

    patch = lora_model_from_ltx2_state_dict(state_dict)

    assert len(patch.layers) == len(PUBLISHED_LAYERS)


def test_half_a_low_rank_pair_is_named_rather_than_a_bare_keyerror() -> None:
    """A truncated or hand-edited file used to surface as `KeyError: 'lora_B.weight'` from deep in
    the patch build, with nothing saying which layer or why."""
    state_dict = _published_state_dict()
    victim = f"diffusion_model.{PUBLISHED_LAYERS[0][0]}.lora_B.weight"
    del state_dict[victim]

    with pytest.raises(ValueError, match="missing one half of its low-rank pair"):
        lora_model_from_ltx2_state_dict(state_dict)


def test_lycoris_variants_are_refused_by_name() -> None:
    state_dict = _published_state_dict()
    state_dict["diffusion_model.transformer_blocks.0.attn1.to_q.dora_scale"] = _z(4096)

    with pytest.raises(ValueError, match="LoKR/LoHA/DoRA"):
        lora_model_from_ltx2_state_dict(state_dict)


def test_the_published_layout_is_recognized_as_ltx2() -> None:
    assert is_state_dict_likely_in_ltx2_format(_published_state_dict())
