"""Qwen-Image LoRA identification across the key layouts trainers save.

Most Qwen-Image LoRAs on Hugging Face prefix their keys with ComfyUI's `diffusion_model.`, and raw PEFT adapters
with `base_model.model.`. Other families' LoRAs carry the same prefixes, so a prefixed file is claimed only when
it has low-rank weights on a module that Qwen-Image's dual-stream blocks alone have.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from invokeai.backend.model_manager.configs.base import Config_Base
from invokeai.backend.model_manager.configs.identification_utils import NotAMatchError
from invokeai.backend.model_manager.configs.lora import LoRA_LyCORIS_FLUX_Config, LoRA_LyCORIS_QwenImage_Config
from invokeai.backend.model_manager.taxonomy import ModelType

# Every Linear in a Qwen-Image transformer block, with its (in, out) features.
_QWEN_IMAGE_BLOCK_LINEARS = {
    "attn.to_q": (3072, 3072),
    "attn.to_k": (3072, 3072),
    "attn.to_v": (3072, 3072),
    "attn.to_out.0": (3072, 3072),
    "attn.add_q_proj": (3072, 3072),
    "attn.add_k_proj": (3072, 3072),
    "attn.add_v_proj": (3072, 3072),
    "attn.to_add_out": (3072, 3072),
    "img_mod.1": (3072, 18432),
    "img_mlp.net.0.proj": (3072, 12288),
    "img_mlp.net.2": (12288, 3072),
    "txt_mod.1": (3072, 18432),
    "txt_mlp.net.0.proj": (3072, 12288),
    "txt_mlp.net.2": (12288, 3072),
}


def _z(*shape: int) -> torch.Tensor:
    """A zero tensor of the given shape with a single element of storage: the probes read names and shapes only."""
    return torch.zeros((), dtype=torch.float32).expand(shape)


def _lora(layers: dict[str, tuple[int, int]]) -> dict[str, torch.Tensor]:
    sd: dict[str, torch.Tensor] = {}
    for layer, (in_features, out_features) in layers.items():
        sd[f"{layer}.lora_A.weight"] = _z(16, in_features)
        sd[f"{layer}.lora_B.weight"] = _z(out_features, 16)
    return sd


def _qwen_image_lora(prefix: str) -> dict[str, torch.Tensor]:
    return _lora(
        {
            f"{prefix}transformer_blocks.{block}.{module}": features
            for block in range(2)
            for module, features in _QWEN_IMAGE_BLOCK_LINEARS.items()
        }
    )


def _make_mod(path: Path, sd: dict[str, torch.Tensor]) -> MagicMock:
    path.touch()
    mod = MagicMock()
    mod.path = path
    mod.load_state_dict.return_value = sd
    mod.metadata.return_value = {}
    return mod


def _overrides(path: Path) -> dict[str, str | int]:
    return {
        "hash": "test-hash",
        "path": str(path),
        "file_size": 0,
        "name": path.stem,
        "source": str(path),
        "source_type": "path",
    }


def _accepting_probes(sd: dict[str, torch.Tensor], tmp_path: Path) -> list[str]:
    path = tmp_path / "lora.safetensors"
    mod = _make_mod(path, sd)

    accepting = []
    for cls in Config_Base.CONFIG_CLASSES:
        if getattr(cls.model_fields.get("type"), "default", None) is not ModelType.LoRA:
            continue
        try:
            cls.from_model_on_disk(mod, _overrides(path))
        except Exception:
            # As in the model config factory, anything but a returned config is no match.
            continue
        accepting.append(cls.__name__)
    return accepting


@pytest.mark.parametrize("prefix", ["", "transformer.", "diffusion_model.", "base_model.model."])
def test_only_the_qwen_image_probe_claims_a_qwen_image_lora(prefix: str, tmp_path: Path) -> None:
    """`Config_Base.CONFIG_CLASSES` is a set: a second LoRA probe accepting the file would make the result depend
    on iteration order."""
    assert _accepting_probes(_qwen_image_lora(prefix), tmp_path) == [LoRA_LyCORIS_QwenImage_Config.__name__]


@pytest.mark.parametrize(
    ("prefix", "text_encoder_prefix"),
    [
        ("", "text_encoder."),
        ("base_model.model.", "base_model.model.text_encoder."),
        ("diffusion_model.", "text_encoders.qwen25_7b.transformer."),
    ],
)
def test_a_lora_with_text_encoder_layers_is_still_claimed_by_the_qwen_image_probe_alone(
    prefix: str, text_encoder_prefix: str, tmp_path: Path
) -> None:
    """Qwen2.5-VL layers in the file are applied to the encoder, but must not change what the file is identified as."""
    path = tmp_path / "lora.safetensors"
    sd = _qwen_image_lora(prefix)
    sd.update(
        _lora(
            {
                f"{text_encoder_prefix}model.language_model.layers.0.self_attn.q_proj": (3584, 3584),
                f"{text_encoder_prefix}model.language_model.layers.0.mlp.down_proj": (18944, 3584),
                f"{text_encoder_prefix}model.visual.blocks.0.attn.qkv": (1280, 3840),
            }
        )
    )
    mod = _make_mod(path, sd)

    accepting = []
    for cls in Config_Base.CONFIG_CLASSES:
        if getattr(cls.model_fields.get("type"), "default", None) is not ModelType.LoRA:
            continue
        try:
            cls.from_model_on_disk(mod, _overrides(path))
        except Exception:
            continue
        accepting.append(cls.__name__)

    assert accepting == [LoRA_LyCORIS_QwenImage_Config.__name__]


def test_a_onetrainer_kohya_lora_with_text_encoder_layers_is_claimed_by_the_qwen_image_probe_alone(
    tmp_path: Path,
) -> None:
    path = tmp_path / "lora.safetensors"
    sd: dict[str, torch.Tensor] = {}
    for block in range(2):
        for module, (in_features, out_features) in {
            "attn_to_q": (3072, 3072),
            "attn_add_q_proj": (3072, 3072),
            "img_mod_1": (3072, 18432),
            "img_mlp_net_0_proj": (3072, 12288),
            "txt_mlp_net_2": (12288, 3072),
        }.items():
            layer = f"lora_unet_transformer_blocks_{block}_{module}"
            sd[f"{layer}.lora_down.weight"] = _z(16, in_features)
            sd[f"{layer}.lora_up.weight"] = _z(out_features, 16)
            sd[f"{layer}.alpha"] = torch.tensor(16.0)
    for module, (in_features, out_features) in {
        "self_attn_q_proj": (3584, 3584),
        "mlp_down_proj": (18944, 3584),
    }.items():
        layer = f"lora_te_model_language_model_layers_0_{module}"
        sd[f"{layer}.lora_down.weight"] = _z(16, in_features)
        sd[f"{layer}.lora_up.weight"] = _z(out_features, 16)
        sd[f"{layer}.alpha"] = torch.tensor(16.0)
    mod = _make_mod(path, sd)

    accepting = []
    for cls in Config_Base.CONFIG_CLASSES:
        if getattr(cls.model_fields.get("type"), "default", None) is not ModelType.LoRA:
            continue
        try:
            cls.from_model_on_disk(mod, _overrides(path))
        except Exception:
            continue
        accepting.append(cls.__name__)

    assert accepting == [LoRA_LyCORIS_QwenImage_Config.__name__]


@pytest.mark.parametrize(
    "layers",
    [
        pytest.param(
            # FLUX's double blocks in diffusers naming share Qwen-Image's attention projections.
            {
                "base_model.model.transformer_blocks.0.attn.add_k_proj": (3072, 3072),
                "base_model.model.transformer_blocks.0.attn.to_add_out": (3072, 3072),
                "base_model.model.transformer_blocks.0.ff.net.0.proj": (3072, 12288),
                "base_model.model.transformer_blocks.0.norm1.linear": (3072, 18432),
            },
            id="flux-diffusers-naming",
        ),
        pytest.param(
            # FLUX's own naming has `img_mlp` and `img_mod` too, in `double_blocks`.
            {
                "base_model.model.double_blocks.0.img_attn.qkv": (3072, 9216),
                "base_model.model.double_blocks.0.img_mlp.0": (3072, 12288),
                "base_model.model.double_blocks.0.img_mod.lin": (3072, 18432),
            },
            id="flux-bfl-naming",
        ),
        pytest.param(
            # Attention alone names nothing only Qwen-Image has; Krea-2 installs this layout under an explicit base.
            {"diffusion_model.transformer_blocks.0.attn.to_q": (3072, 3072)},
            id="attention-only",
        ),
        pytest.param(
            # Qwen-Image-2.1's blocks name their MLP `img_mlp.proj` / `img_mlp.out`, not `img_mlp.net`.
            {
                "diffusion_model.transformer_blocks.0.img_mlp.proj": (4096, 16384),
                "diffusion_model.transformer_blocks.0.img_mlp.out": (16384, 4096),
            },
            id="qwen-image-2.1-mlp",
        ),
    ],
)
def test_a_prefixed_lora_without_a_qwen_image_only_module_is_not_claimed(
    layers: dict[str, tuple[int, int]], tmp_path: Path
) -> None:
    path = tmp_path / "lora.safetensors"

    with pytest.raises(NotAMatchError):
        LoRA_LyCORIS_QwenImage_Config.from_model_on_disk(_make_mod(path, _lora(layers)), _overrides(path))


def test_a_prefixed_peft_dora_is_not_claimed(tmp_path: Path) -> None:
    """PEFT DoRA magnitudes are not mapped yet, so claiming the file would install a LoRA that fails to apply."""
    sd = _qwen_image_lora("base_model.model.")
    for key in [key for key in sd if key.endswith(".lora_B.weight")]:
        sd[key.removesuffix(".lora_B.weight") + ".lora_magnitude_vector"] = _z(sd[key].shape[0])
    path = tmp_path / "adapter_model.safetensors"

    with pytest.raises(NotAMatchError):
        LoRA_LyCORIS_QwenImage_Config.from_model_on_disk(_make_mod(path, sd), _overrides(path))


def _lokr_layers(prefix: str, *, factorized_w1: bool = False) -> dict[str, torch.Tensor]:
    sd: dict[str, torch.Tensor] = {}
    for module, (in_features, out_features) in _QWEN_IMAGE_BLOCK_LINEARS.items():
        key = f"{prefix}{module}"
        if factorized_w1:
            sd[f"{key}.lokr_w1_a"] = _z(16, 4)
            sd[f"{key}.lokr_w1_b"] = _z(4, 16)
        else:
            sd[f"{key}.lokr_w1"] = _z(16, 16)
        sd[f"{key}.lokr_w2"] = _z(out_features // 16, in_features // 16)
    return sd


@pytest.mark.parametrize("factorized_w1", [False, True])
@pytest.mark.parametrize("prefix", ["transformer.", "diffusion_model.", "base_model.model."])
def test_a_dotted_lokr_with_qwen_image_only_modules_is_claimed(
    prefix: str, factorized_w1: bool, tmp_path: Path
) -> None:
    sd = _lokr_layers(f"{prefix}transformer_blocks.0.", factorized_w1=factorized_w1)

    assert _accepting_probes(sd, tmp_path) == [LoRA_LyCORIS_QwenImage_Config.__name__]


def _loha_layers(prefix: str) -> dict[str, torch.Tensor]:
    sd: dict[str, torch.Tensor] = {}
    for module, (in_features, out_features) in _QWEN_IMAGE_BLOCK_LINEARS.items():
        key = f"{prefix}{module}"
        sd[f"{key}.hada_w1_a"] = _z(out_features, 8)
        sd[f"{key}.hada_w1_b"] = _z(8, in_features)
        sd[f"{key}.hada_w2_a"] = _z(out_features, 8)
        sd[f"{key}.hada_w2_b"] = _z(8, in_features)
    return sd


@pytest.mark.parametrize("prefix", ["transformer.transformer_blocks.0.", "transformer_blocks.0."])
def test_a_dotted_loha_with_qwen_image_only_modules_is_claimed(prefix: str, tmp_path: Path) -> None:
    assert _accepting_probes(_loha_layers(prefix), tmp_path) == [LoRA_LyCORIS_QwenImage_Config.__name__]


@pytest.mark.parametrize("prefix", ["transformer.transformer_blocks.0.", "transformer_blocks.0."])
@pytest.mark.parametrize(
    "modules",
    [
        pytest.param(
            {
                "attn.to_q": (3072, 3072),
                "attn.add_q_proj": (3072, 3072),
                "ff.net.0.proj": (3072, 12288),
                "ff_context.net.2": (12288, 3072),
                "norm1.linear": (3072, 18432),
            },
            id="flux-diffusers-naming",
        ),
        pytest.param({"attn.to_q": (3072, 3072), "attn.to_k": (3072, 3072)}, id="attention-only"),
    ],
)
def test_a_dotted_loha_without_a_qwen_image_only_module_is_not_claimed(
    prefix: str, modules: dict[str, tuple[int, int]], tmp_path: Path
) -> None:
    """The LoHA suffixes are new to this probe, so an ambiguous LoHA stays unrecognized rather than landing in
    Qwen-Image."""
    sd: dict[str, torch.Tensor] = {}
    for module, (in_features, out_features) in modules.items():
        for param, shape in (("w1_a", (out_features, 8)), ("w1_b", (8, in_features))) + (
            ("w2_a", (out_features, 8)),
            ("w2_b", (8, in_features)),
        ):
            sd[f"{prefix}{module}.hada_{param}"] = _z(*shape)

    assert LoRA_LyCORIS_QwenImage_Config.__name__ not in _accepting_probes(sd, tmp_path)


# Flattened LyCORIS and ai-toolkit spellings: `lycoris_transformer_blocks_N_<module>` and
# `lora_transformer_transformer_blocks_N_<module>`.


def _flattened(prefix: str, sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename `<module.with.dots>.<param>` keys to `<prefix><module_with_underscores>.<param>`."""
    flat: dict[str, torch.Tensor] = {}
    for key, value in sd.items():
        module, param = key.rsplit(".", 1)
        flat[f"{prefix}{module.replace('.', '_')}.{param}"] = value
    return flat


@pytest.mark.parametrize("scalar", [False, True])
@pytest.mark.parametrize("factorized_w1", [False, True])
def test_a_lycoris_prefixed_lokr_is_claimed(factorized_w1: bool, scalar: bool, tmp_path: Path) -> None:
    """Trained LyCORIS LoKr files save `lycoris_transformer_blocks_N_<module>` keys, some with a learned scalar."""
    sd = _flattened("lycoris_transformer_blocks_0_", _lokr_layers("", factorized_w1=factorized_w1))
    for module in {key.split(".", 1)[0] for key in sd}:
        sd[f"{module}.alpha"] = _z()
        if scalar:
            sd[f"{module}.scalar"] = _z()

    assert _accepting_probes(sd, tmp_path) == [LoRA_LyCORIS_QwenImage_Config.__name__]


def test_a_lora_transformer_prefixed_loha_with_dora_is_qwen_image_not_flux(tmp_path: Path) -> None:
    """The `lora_transformer_transformer_blocks_` spelling (ai-toolkit LoHa) is also OneTrainer's for FLUX, and the
    FLUX probe accepts any module name, so only Qwen-Image's own modules tell them apart."""
    sd = _flattened("lora_transformer_transformer_blocks_0_", _loha_layers(""))
    for module in {key.split(".", 1)[0] for key in sd}:
        sd[f"{module}.dora_scale"] = _z(8, 1)
        sd[f"{module}.alpha"] = _z()
        sd[f"{module}.scalar"] = _z()

    assert _accepting_probes(sd, tmp_path) == [LoRA_LyCORIS_QwenImage_Config.__name__]


def test_a_flux_onetrainer_lora_is_still_flux(tmp_path: Path) -> None:
    """The same spelling with FLUX's own modules (diffusers naming) stays with the FLUX probe."""
    sd: dict[str, torch.Tensor] = {}
    flux_modules = {
        "lora_transformer_transformer_blocks_0_": {
            "attn_to_q": (3072, 3072),
            "attn_add_k_proj": (3072, 3072),
            "attn_to_add_out": (3072, 3072),
            "ff_net_0_proj": (3072, 12288),
            "ff_context_net_2": (12288, 3072),
            "norm1_linear": (3072, 18432),
        },
        "lora_transformer_single_transformer_blocks_0_": {
            "attn_to_q": (3072, 3072),
            "proj_mlp": (3072, 12288),
            "proj_out": (15360, 3072),
        },
    }
    for prefix, modules in flux_modules.items():
        for module, (in_features, out_features) in modules.items():
            sd[f"{prefix}{module}.lora_down.weight"] = _z(16, in_features)
            sd[f"{prefix}{module}.lora_up.weight"] = _z(out_features, 16)
            sd[f"{prefix}{module}.alpha"] = _z()

    assert _accepting_probes(sd, tmp_path) == [LoRA_LyCORIS_FLUX_Config.__name__]


@pytest.mark.parametrize(
    "layers",
    [
        pytest.param(
            # FLUX in diffusers naming, without single blocks: attention and feed-forward modules only.
            {
                "lycoris_transformer_blocks_0_attn_to_q": (3072, 3072),
                "lycoris_transformer_blocks_0_attn_add_k_proj": (3072, 3072),
                "lycoris_transformer_blocks_0_ff_net_0_proj": (3072, 12288),
            },
            id="flux-diffusers-naming",
        ),
        pytest.param(
            # LTX-2: video attention plus the audio and cross-modal towers.
            {
                "lycoris_transformer_blocks_0_attn1_to_q": (4096, 4096),
                "lycoris_transformer_blocks_0_audio_attn1_to_q": (2048, 2048),
                "lycoris_transformer_blocks_0_audio_to_video_attn_to_q": (2048, 2048),
            },
            id="ltx2",
        ),
        pytest.param(
            # Qwen-Image-2.1: a gated MLP instead of `img_mlp.net`.
            {
                "lycoris_transformer_blocks_0_attn_to_q": (4096, 4096),
                "lycoris_transformer_blocks_0_img_mlp_proj": (4096, 16384),
                "lycoris_transformer_blocks_0_img_mlp_out": (16384, 4096),
            },
            id="qwen-image-2.1",
        ),
    ],
)
def test_a_lycoris_prefixed_lokr_of_another_family_is_not_claimed(
    layers: dict[str, tuple[int, int]], tmp_path: Path
) -> None:
    sd: dict[str, torch.Tensor] = {}
    for key, (in_features, out_features) in layers.items():
        sd[f"{key}.lokr_w1"] = _z(16, 16)
        sd[f"{key}.lokr_w2"] = _z(out_features // 16, in_features // 16)

    assert LoRA_LyCORIS_QwenImage_Config.__name__ not in _accepting_probes(sd, tmp_path)
