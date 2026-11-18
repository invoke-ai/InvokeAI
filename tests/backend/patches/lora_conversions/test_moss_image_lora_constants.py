"""What a MOSS-Image LoRA may hold, against the released transformer and the headers of published LoRAs.

The constants state the released geometry by hand, so the tests read it back from the vendored transformer at that
geometry (on the meta device): a module or shape the table gets wrong shows here, not as a layer that a generation
skips. The captured headers are what the published LoRAs are -- ai-toolkit, ComfyUI, kohya, DiffSynth, diffusers and
SimpleTuner layouts -- and each must land on Linears of the right shape.
"""

import re

import accelerate
import pytest
import torch

from invokeai.backend.moss_image.pipeline_moss_image import MOSSImageTransformer, _RMSNorm
from invokeai.backend.patches.lora_conversions import moss_image_lora_constants as constants
from invokeai.backend.patches.lora_conversions.peft_adapter_utils import normalize_peft_adapter_names
from tests.backend.patches.lora_conversions.lora_state_dicts import (
    moss_image_lora_aitoolkit_dora_format,
    moss_image_lora_aitoolkit_format,
    moss_image_lora_aitoolkit_fused_format,
    moss_image_lora_bare_format,
    moss_image_lora_comfy_format,
    moss_image_lora_comfy_fused_format,
    moss_image_lora_comfy_fused_partial_format,
    moss_image_lora_control_format,
    moss_image_lora_diffsynth_format,
    moss_image_lora_diffusers_format,
    moss_image_lora_kohya_format,
    moss_image_lora_kohya_fused_format,
    moss_image_lora_loha_format,
    moss_image_lora_lokr_auto_format,
    moss_image_lora_lokr_factor4_format,
    moss_image_lora_lokr_factor16_format,
    moss_image_lora_lycoris_dora_format,
    moss_image_lora_norm_diff_format,
    moss_image_lora_prefixed_alpha_format,
)

# `OpenMOSS-Team/MOSS-Image-Base`'s `transformer/config.json`, as `test_moss_image_loader.RELEASED_GEOMETRY` pins it.
_RELEASED = {
    "dim": 3072,
    "ffn_dim": 8192,
    "freq_dim": 256,
    "latent_channels": 128,
    "num_heads": 24,
    "num_layers": 42,
    "text_dim": 4096,
}
_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}


@pytest.fixture(scope="module")
def released_transformer() -> MOSSImageTransformer:
    with accelerate.init_empty_weights():
        return MOSSImageTransformer(**_RELEASED)


def _header(keys: dict[str, tuple[list[int], str]]) -> dict[str, torch.Tensor]:
    """A captured header as tensors on the meta device, but for the 0-D alphas, which a layer reads a value of."""
    return normalize_peft_adapter_names(
        {
            key: torch.zeros(()) if not shape else torch.empty(shape, dtype=_DTYPES[dtype], device="meta")
            for key, (shape, dtype) in keys.items()
        }
    )


def _lora_of(module: torch.nn.Module, path: str, value_key: str) -> dict[str, torch.Tensor]:
    return {f"diffusion_model.{path}.{value_key}": torch.empty(module.weight.shape, device="meta")}


@pytest.mark.parametrize("fused", [False, True], ids=["split", "fused-qkv"])
def test_every_linear_of_the_released_transformer_takes_a_full_update_of_its_shape(
    fused: bool, released_transformer
) -> None:
    """A full `diff` of exactly a weight's shape is the plainest layer there is: it applies where the table has the
    module and its shape. The checkpoints fuse a block's q, k and v into one `proj.qkv` of three times the rows."""
    state_dict: dict[str, torch.Tensor] = {}
    for path, module in released_transformer.named_modules():
        if isinstance(module, torch.nn.Linear) and not (fused and re.fullmatch(r"layers\.\d+\.proj\.[qkv]", path)):
            state_dict |= _lora_of(module, path, "diff")
    if fused:
        for index in range(_RELEASED["num_layers"]):
            state_dict[f"diffusion_model.layers.{index}.proj.qkv.diff"] = torch.empty(3 * 3072, 3072, device="meta")

    assert constants.has_moss_image_lora_keys(state_dict)
    assert constants.unsupported_keys(state_dict) == {}


def test_a_full_update_of_another_shape_is_refused_for_each_linear(released_transformer) -> None:
    for path, module in released_transformer.named_modules():
        if isinstance(module, torch.nn.Linear) and path in ("layers.7.proj.ffn_down", "time_embedding.0"):
            out_features, in_features = module.weight.shape
            wrong = {f"diffusion_model.{path}.diff": torch.empty(out_features, in_features + 1, device="meta")}

            assert set(constants.unsupported_keys(wrong).values()) == {constants.WRONG_SHAPE}


def test_every_norm_of_the_released_transformer_is_a_module_a_lora_cannot_patch(released_transformer) -> None:
    """Named, so a norm `diff` is refused as a norm and not as a module the table does not know."""
    norms = {path for path, module in released_transformer.named_modules() if isinstance(module, _RMSNorm)}
    assert len(norms) == 42 * 4 + 2 * 4 + 1

    for path in norms:
        refused = constants.unsupported_keys({f"diffusion_model.{path}.diff": torch.empty(3072, device="meta")})
        assert set(refused.values()) == {constants.NORM_WEIGHT}, path


_APPLYING = {
    "ai-toolkit": moss_image_lora_aitoolkit_format,
    "ai-toolkit-fused": moss_image_lora_aitoolkit_fused_format,
    "kohya": moss_image_lora_kohya_format,
    "kohya-fused": moss_image_lora_kohya_fused_format,
    "prefixed-alpha": moss_image_lora_prefixed_alpha_format,
    "diffsynth": moss_image_lora_diffsynth_format,
    "bare": moss_image_lora_bare_format,
    "comfy": moss_image_lora_comfy_format,
    "comfy-fused": moss_image_lora_comfy_fused_format,
    "comfy-fused-partial": moss_image_lora_comfy_fused_partial_format,
    "lokr-factor-16": moss_image_lora_lokr_factor16_format,
    "lokr-factor-4": moss_image_lora_lokr_factor4_format,
    "lokr-auto": moss_image_lora_lokr_auto_format,
    "loha": moss_image_lora_loha_format,
    "ai-toolkit-dora": moss_image_lora_aitoolkit_dora_format,
    "lycoris-dora": moss_image_lora_lycoris_dora_format,
    "diffusers": moss_image_lora_diffusers_format,
}


@pytest.mark.parametrize("layout", list(_APPLYING))
def test_a_published_lora_lands_on_linears_of_its_shape(layout: str, released_transformer) -> None:
    state_dict = _header(_APPLYING[layout].state_dict_keys)
    keys = constants.read_moss_image_lora_keys(state_dict)

    assert constants.has_moss_image_lora_keys(state_dict)
    assert constants.unsupported_keys(state_dict) == {}
    modules = dict(released_transformer.named_modules())
    for module, sources in keys.layers.items():
        values = {value_key: state_dict[key] for value_key, key in sources.items()}
        targets = constants.split_qkv_modules(module) if constants.is_fused_qkv(module) else (module,)
        linear = modules[targets[0]]
        assert all(isinstance(modules[target], torch.nn.Linear) for target in targets), module
        assert constants.update_shape(values) == (len(targets) * linear.out_features, linear.in_features), module


def test_a_lora_that_also_patches_norms_and_biases_is_refused_for_each() -> None:
    refused = constants.unsupported_keys(_header(moss_image_lora_norm_diff_format.state_dict_keys))

    assert set(refused.values()) == {constants.NORM_WEIGHT, constants.NOT_A_LORA_PARAMETER}
    assert {key for key, reason in refused.items() if reason == constants.NORM_WEIGHT} == {
        key for key in moss_image_lora_norm_diff_format.state_dict_keys if key.endswith(".diff")
    }
    assert all(key.endswith(".diff_b") for key, reason in refused.items() if reason == constants.NOT_A_LORA_PARAMETER)
    assert constants.describe_refusal(refused).startswith("This MOSS-Image LoRA does not apply in full: ")


def test_a_control_lora_is_refused_for_the_weights_it_replaces() -> None:
    """Its low-rank layers apply; the widened `video_embedder` it ships beside them is a weight, not a LoRA parameter."""
    refused = constants.unsupported_keys(_header(moss_image_lora_control_format.state_dict_keys))

    assert refused == dict.fromkeys(("video_embedder.weight", "video_embedder.bias"), constants.NOT_A_LORA_PARAMETER)


@pytest.mark.parametrize(
    "key",
    [
        # MOSS-Image's edit LoRAs: bare `lora_A`/`lora_B`, without PEFT's `.weight`.
        "layers.0.proj.q.lora_A",
        # FLUX diffusers' text embedder and Z-Image's attention are other transformers' modules.
        "transformer.time_text_embed.text_embedder.linear_1.lora_A.weight",
        "transformer.layers.0.attention.to_q.lora_A.weight",
        # A timestep MLP other transformers have under that name, alone.
        "diffusion_model.time_embedding.0.lora_A.weight",
        "lora_unet_time_embedding_0.lora_down.weight",
        # A checkpoint, not a LoRA.
        "layers.0.proj.q.weight",
    ],
)
def test_keys_that_only_look_like_moss_image_are_not_its_marker(key: str) -> None:
    assert not constants.has_moss_image_lora_keys([key, 3])


@pytest.mark.parametrize("prefix", ["base_model.model.", "base_model.model.transformer."])
def test_a_raw_peft_lora_is_recognized_and_refused_for_its_missing_alpha(prefix: str) -> None:
    """Raw PEFT keeps its alpha in an `adapter_config.json` that is not in the file. Recognized, so another family's
    probe does not take the file; refused, so it does not install as a LoRA of the wrong scale."""
    state_dict = {
        f"{prefix}layers.0.proj.q.lora_A.weight": torch.empty(8, 3072, device="meta"),
        f"{prefix}layers.0.proj.q.lora_B.weight": torch.empty(3072, 8, device="meta"),
    }

    assert constants.has_moss_image_lora_keys(state_dict)
    assert set(constants.unsupported_keys(state_dict).values()) == {constants.RAW_PEFT}


def test_bundled_embeddings_are_skipped_as_the_loader_does() -> None:
    """Some trainers bundle textual-inversion embeddings into a LoRA; `LoRALoader` drops them, so they are no reason to
    refuse the file."""
    state_dict = _header(moss_image_lora_aitoolkit_format.state_dict_keys)
    state_dict["bundle_emb.emb_params.string_to_param.*"] = torch.empty(4, 4096, device="meta")

    assert constants.has_moss_image_lora_keys(state_dict)
    assert constants.unsupported_keys(state_dict) == {}
