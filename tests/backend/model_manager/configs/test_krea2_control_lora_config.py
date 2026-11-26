"""Krea-2 Control LoRA identification through the config factory, against every other config class.

The file layout is the published `Patil/Krea-2-depth-controlnet` checkpoint's: 28 blocks of native-named `.A`/`.B`
LoRA halves at the real module widths (rank 1 here, 64 there) and a `first` projection widened to 128 inputs.
"""

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from invokeai.backend.model_manager.configs.factory import ModelConfigFactory
from invokeai.backend.model_manager.configs.lora import ControlLoRA_LyCORIS_Krea2_Config, LoRA_LyCORIS_Krea2_Config
from invokeai.backend.model_manager.taxonomy import ModelType

# (in, out) of each LoRA'd Linear in a Krea-2 block.
_MODULE_WIDTHS = {
    "attn.wq": (6144, 6144),
    "attn.wk": (6144, 1536),
    "attn.wv": (6144, 1536),
    "attn.wo": (6144, 6144),
    "attn.gate": (6144, 6144),
    "mlp.gate": (6144, 16384),
    "mlp.up": (6144, 16384),
    "mlp.down": (16384, 6144),
}


def _block_halves(num_blocks: int = 28) -> dict[str, torch.Tensor]:
    return {
        key: tensor
        for block in range(num_blocks)
        for module, (width_in, width_out) in _MODULE_WIDTHS.items()
        for key, tensor in (
            (f"blocks.{block}.{module}.A", torch.zeros(1, width_in)),
            (f"blocks.{block}.{module}.B", torch.zeros(width_out, 1)),
        )
    }


def _projection(width_in: int = 128) -> dict[str, torch.Tensor]:
    return {"first.weight": torch.zeros(6144, width_in), "first.bias": torch.zeros(6144)}


def _identify(path: Path, tensors: dict[str, torch.Tensor]):
    save_file(tensors, str(path))
    return ModelConfigFactory.from_model_on_disk(path, allow_unknown=False)


@pytest.fixture(scope="module")
def control_lora_file(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("krea2_control_lora") / "depth-control-lora.safetensors"
    save_file({**_block_halves(), **_projection()}, str(path))
    return path


def test_the_published_layout_is_a_krea2_control_lora_and_nothing_else(control_lora_file: Path) -> None:
    result = ModelConfigFactory.from_model_on_disk(control_lora_file, allow_unknown=False)

    assert [type(match) for match in result.all_matches] == [ControlLoRA_LyCORIS_Krea2_Config]
    assert result.config is not None and result.config.type is ModelType.ControlLoRa
    # Control-adapter defaults guess the preprocessor from the file name.
    assert result.config.default_settings is not None
    assert result.config.default_settings.preprocessor == "depth_anything_image_processor"


def test_a_plain_native_krea2_lora_is_still_a_krea2_lora(tmp_path: Path) -> None:
    peft_named = {f"{key[:-2]}.lora_{key[-1]}.weight": tensor for key, tensor in _block_halves(num_blocks=2).items()}

    result = _identify(tmp_path / "krea2-lora.safetensors", peft_named)

    assert [type(match) for match in result.all_matches] == [LoRA_LyCORIS_Krea2_Config]


@pytest.mark.parametrize(
    "tensors",
    [
        pytest.param(
            {k: v for k, v in {**_block_halves(2), **_projection()}.items() if k != "blocks.1.mlp.up.B"},
            id="unpaired-half",
        ),
        pytest.param({**_block_halves(2), "first.weight": torch.zeros(6144, 128)}, id="no-projection-bias"),
        pytest.param({**_block_halves(2), **_projection(width_in=64)}, id="unwidened-projection"),
        pytest.param(
            {**_block_halves(2), **_projection(), "blocks.0.attn.norm_q.scale": torch.zeros(128)}, id="extra-key"
        ),
        pytest.param(
            {
                **_block_halves(2),
                **_projection(),
                "blocks.0.foo.A": torch.zeros(1, 8),
                "blocks.0.foo.B": torch.zeros(8, 1),
            },
            id="unknown-module",
        ),
        pytest.param(
            {**_block_halves(2), "first.weight": torch.zeros(3072, 128), "first.bias": torch.zeros(3072)},
            id="narrow-projection",
        ),
    ],
)
def test_a_file_off_the_published_layout_is_not_a_krea2_control_lora(tmp_path: Path, tensors) -> None:
    result = _identify(tmp_path / "depth-control-lora.safetensors", tensors)

    assert not any(isinstance(match, ControlLoRA_LyCORIS_Krea2_Config) for match in result.all_matches)
