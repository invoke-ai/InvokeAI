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
from invokeai.backend.model_manager.configs.lora import LoRA_LyCORIS_QwenImage_Config
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


@pytest.mark.parametrize("prefix", ["", "transformer.", "diffusion_model.", "base_model.model."])
def test_only_the_qwen_image_probe_claims_a_qwen_image_lora(prefix: str, tmp_path: Path) -> None:
    """`Config_Base.CONFIG_CLASSES` is a set: a second LoRA probe accepting the file would make the result depend
    on iteration order."""
    path = tmp_path / "lora.safetensors"
    mod = _make_mod(path, _qwen_image_lora(prefix))

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


def test_a_prefixed_lokr_is_not_claimed(tmp_path: Path) -> None:
    """The converter cannot group dotted LoKr keys, so claiming one would install a LoRA that fails to apply."""
    sd: dict[str, torch.Tensor] = {}
    for module, (in_features, out_features) in _QWEN_IMAGE_BLOCK_LINEARS.items():
        sd[f"diffusion_model.transformer_blocks.0.{module}.lokr_w1"] = _z(16, 16)
        sd[f"diffusion_model.transformer_blocks.0.{module}.lokr_w2"] = _z(out_features // 16, in_features // 16)
    path = tmp_path / "lokr.safetensors"

    with pytest.raises(NotAMatchError):
        LoRA_LyCORIS_QwenImage_Config.from_model_on_disk(_make_mod(path, sd), _overrides(path))
