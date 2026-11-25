"""SD3's MMDiT single-file layout reaches diffusers' `SD3Transformer2DModel` with its GGML tensors still packed.

Driven over real (tiny) GGUF files in both published layouts. Parity is against diffusers' own converter run on
the same file dequantized first: a cut by quantized rows has to land on exactly the values a cut of the dense
tensor gives, so the comparison is exact.
"""

from pathlib import Path

import pytest
import torch
from diffusers.loaders.single_file_utils import convert_sd3_transformer_checkpoint_to_diffusers

from invokeai.backend.model_manager.checkpoint_prefix import CheckpointPrefix
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor
from invokeai.backend.quantization.gguf.loaders import gguf_sd_loader
from invokeai.backend.sd3.checkpoint_layout import (
    convert_mmdit_to_diffusers,
    is_mmdit_transformer_key,
    is_sd3_mmdit,
    sd3_transformer_config,
)
from tests.backend.model_manager.load.sd3_gguf_fixture import TINY_CONFIG, mmdit_shapes, write_sd3_gguf


def _dense(value: torch.Tensor) -> torch.Tensor:
    return value.get_dequantized_tensor() if isinstance(value, GGMLTensor) else value


@pytest.mark.parametrize("sd_cpp", [False, True], ids=["city96", "stable-diffusion.cpp"])
def test_the_packed_conversion_matches_diffusers_converter(tmp_path: Path, sd_cpp: bool) -> None:
    path = tmp_path / "sd3.gguf"
    write_sd3_gguf(path, sd_cpp=sd_cpp)
    expected = convert_sd3_transformer_checkpoint_to_diffusers(
        {key: _dense(value) for key, value in gguf_sd_loader(path, torch.float32).items()}
    )
    # diffusers keeps stable-diffusion.cpp's batchless `pos_embed` as it is; the model's buffer has the batch dim.
    expected["pos_embed.pos_embed"] = expected["pos_embed.pos_embed"].reshape(1, -1, 64)

    read = gguf_sd_loader(path, torch.float32, keep=is_mmdit_transformer_key)
    sd = CheckpointPrefix.detect(read).strip(read, drop_foreign=True)
    config = sd3_transformer_config(sd)
    converted = convert_mmdit_to_diffusers(sd, config)

    assert config == TINY_CONFIG
    assert converted.keys() == expected.keys()
    mismatched = [key for key in converted if not torch.equal(_dense(converted[key]), expected[key])]
    assert not mismatched
    # The cuts happened on packed rows: the Q8_0 projections they produced are still Q8_0.
    for key in ("transformer_blocks.0.attn.to_k.weight", "transformer_blocks.0.attn2.to_v.weight"):
        assert isinstance(converted[key], GGMLTensor), key
        assert converted[key]._ggml_quantization_type.name == "Q8_0", key


def _meta(shapes: dict[str, tuple[int, ...]]) -> dict[str, torch.Tensor]:
    return {key: torch.empty(shape, device="meta") for key, shape in shapes.items()}


def test_the_config_follows_the_shapes_of_a_released_model() -> None:
    """SD 3.5 Large: 38 blocks of width 2432, no MMDiT-X attention, a 192-wide position grid."""
    large = {
        **TINY_CONFIG,
        "num_layers": 38,
        "attention_head_dim": 64,
        "num_attention_heads": 38,
        "caption_projection_dim": 2432,
        "in_channels": 16,
        "out_channels": 16,
        "pos_embed_max_size": 192,
        "dual_attention_layers": (),
    }

    sd = _meta(mmdit_shapes(large))

    assert is_sd3_mmdit(sd)
    assert sd3_transformer_config(sd) == large


def test_sd3_medium_without_qk_norms_gets_the_released_head_width() -> None:
    """SD 3.0 Medium has no qk norms, the only tensors that record the head width."""
    sd = _meta({key: shape for key, shape in mmdit_shapes().items() if ".ln_" not in key})

    config = sd3_transformer_config(sd)

    # The fixture's 32-wide heads are unreadable here; every released SD3 has 64-wide ones.
    assert (config["qk_norm"], config["attention_head_dim"], config["num_attention_heads"]) == (None, 64, 1)


@pytest.mark.parametrize(
    ("key", "shape"),
    [("context_embedder.weight", (64, 4352)), ("y_embedder.mlp.0.weight", (64, 1280))],
    ids=["not-t5-context", "not-clip-pooled"],
)
def test_an_mmdit_conditioned_on_other_encoders_is_not_sd3(key: str, shape: tuple[int, ...]) -> None:
    assert not is_sd3_mmdit(_meta({**mmdit_shapes(), key: shape}))
