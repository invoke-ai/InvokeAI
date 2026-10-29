"""SD3 GGUF transformers: identified by their MMDiT structure, loaded into diffusers' `SD3Transformer2DModel`.

Both are driven over real (tiny) GGUF files in the two published layouts. Identification goes through the whole
factory, because the question is which of all the candidate configs claims them. The loader is checked by its
forward, against a dense model that diffusers' own converter builds from the values the file dequantizes to.
"""

import re
from pathlib import Path

import gguf
import numpy as np
import pytest
import torch
from diffusers.loaders.single_file_utils import convert_sd3_transformer_checkpoint_to_diffusers
from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
from safetensors.torch import save_file

from invokeai.backend.model_manager.configs.factory import ModelConfigFactory
from invokeai.backend.model_manager.configs.identification_utils import InvalidMatchError
from invokeai.backend.model_manager.configs.main import Main_GGUF_SD3_Config
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
    apply_custom_layers_to_model,
)
from invokeai.backend.model_manager.load.model_loaders import sd3
from invokeai.backend.model_manager.load.model_loaders.sd3 import SD3GGUFModel
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import SubModelType
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor
from tests.backend.model_manager.load.sd3_gguf_fixture import (
    SD_CPP_PREFIX,
    TINY_CONFIG,
    tiny_mmdit_state_dict,
    write_sd3_gguf,
)
from tests.fixtures.loader_seams import Seam, prepare

SEAM = Seam(loader=SD3GGUFModel, module=sd3, entry="_load_model", casts_fp8_storage=False)
LAYOUTS = pytest.mark.parametrize("sd_cpp", [False, True], ids=["city96", "stable-diffusion.cpp"])


def _identify(path: Path):
    return ModelConfigFactory.from_model_on_disk(ModelOnDisk(path), {}, allow_unknown=True)


@pytest.mark.parametrize(
    ("filename", "sd_cpp", "steps"),
    [
        ("sd3.5_medium-Q8_0.gguf", False, 40),
        ("sd3.5_medium-Q4_0.gguf", True, 40),
        ("sd3.5_large_turbo-Q4_0.gguf", False, 4),
    ],
    ids=["city96", "stable-diffusion.cpp", "large-turbo"],
)
def test_a_published_gguf_installs_as_sd3_and_nothing_else(
    tmp_path: Path, filename: str, sd_cpp: bool, steps: int
) -> None:
    path = tmp_path / filename
    write_sd3_gguf(path, sd_cpp=sd_cpp)

    result = _identify(path)

    assert result.match_count == 1, [type(match).__name__ for match in result.all_matches]
    assert isinstance(result.config, Main_GGUF_SD3_Config)
    # Large Turbo shares Large's architecture; only the name tells the distilled model apart.
    assert result.config.default_settings.steps == steps


def _write_gguf(path: Path, sd: dict[str, torch.Tensor]) -> None:
    writer = gguf.GGUFWriter(str(path), "sd3")
    for name, tensor in sd.items():
        data = tensor.numpy()
        if name.removeprefix(SD_CPP_PREFIX).startswith("joint_blocks.") and data.ndim == 2:
            raw = gguf.quantize(data, gguf.GGMLQuantizationType.Q8_0)
            writer.add_tensor(name, raw, raw_shape=raw.shape, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
        else:
            writer.add_tensor(name, data.astype(np.float16))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _without(key: str):
    return lambda sd: {k: v for k, v in sd.items() if k != key}


def _with(key: str, shape: tuple[int, ...], prefix: str = ""):
    return lambda sd: {**{f"{prefix}{k}": v for k, v in sd.items()}, f"{prefix}{key}": torch.zeros(shape)}


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        (
            _without("joint_blocks.1.x_block.mlp.fc1.weight"),
            r"incomplete.*'joint_blocks\.1\.x_block\.mlp\.fc1\.weight'",
        ),
        # A layer-norm qk norm, say: diffusers' RMS norm has no bias, and dropping it would change the model.
        (
            _with("joint_blocks.0.x_block.attn.ln_q.bias", (32,)),
            r"no place.*'joint_blocks\.0\.x_block\.attn\.ln_q\.bias'",
        ),
        # A top-level module the MMDiT does not have is refused too, in a bare file and behind the prefix alike.
        (_with("extra_head.weight", (64, 64)), r"no place.*'extra_head\.weight'"),
        (_with("extra_head.weight", (64, 64), SD_CPP_PREFIX), r"no place.*'extra_head\.weight'"),
    ],
    ids=["incomplete", "unsupported-layout", "unknown-module", "unknown-module-prefixed"],
)
def test_an_sd3_gguf_it_cannot_build_is_refused_at_install(tmp_path: Path, change, reason: str) -> None:
    """Refused, not filed as an unknown model that fails at the first generation, and naming the tensor."""
    path = tmp_path / "sd3.gguf"
    _write_gguf(path, change(tiny_mmdit_state_dict()))

    result = _identify(path)

    assert result.config is None
    refusal = result.details[Main_GGUF_SD3_Config.__name__]
    assert isinstance(refusal, InvalidMatchError)
    assert re.search(reason, str(refusal)), str(refusal)


def test_a_safetensors_mmdit_is_not_claimed_as_gguf(tmp_path: Path) -> None:
    path = tmp_path / "sd3.5_medium.safetensors"
    save_file({key: value.half() for key, value in tiny_mmdit_state_dict().items()}, path)

    result = _identify(path)

    assert not isinstance(result.config, Main_GGUF_SD3_Config)


def _load(monkeypatch, tmp_path: Path, *, sd_cpp: bool = False):
    path = tmp_path / "sd3.5_medium-Q8_0.gguf"
    meant = write_sd3_gguf(path, sd_cpp=sd_cpp)
    config = Main_GGUF_SD3_Config.model_construct(path=str(path), name="sd3.5_medium-Q8_0")
    model = prepare(SEAM, monkeypatch).load(config, SubModelType.Transformer)
    return model, meant


def _forward(model: torch.nn.Module) -> torch.Tensor:
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        return model(
            # A 4x4 patch grid inside the 8x8 position table, so the crop is exercised too.
            hidden_states=torch.randn(1, TINY_CONFIG["in_channels"], 8, 8, generator=generator),
            encoder_hidden_states=torch.randn(1, 5, TINY_CONFIG["joint_attention_dim"], generator=generator),
            pooled_projections=torch.randn(1, TINY_CONFIG["pooled_projection_dim"], generator=generator),
            timestep=torch.tensor([500.0]),
            return_dict=False,
        )[0]


@LAYOUTS
def test_the_gguf_transformer_computes_what_its_file_means(monkeypatch, tmp_path: Path, sd_cpp: bool) -> None:
    """Not bit-exact: the torch kernels multiply codes by their fp16 block scale in fp16, the reference reader
    in float32. A swapped shift/scale, a q/k/v cut in the wrong place or a dropped norm is off by far more."""
    model, meant = _load(monkeypatch, tmp_path, sd_cpp=sd_cpp)
    apply_custom_layers_to_model(model)
    reference = SD3Transformer2DModel(**TINY_CONFIG)
    reference.load_state_dict(convert_sd3_transformer_checkpoint_to_diffusers(meant), strict=True)

    output, expected = _forward(model), _forward(reference)

    assert expected.abs().mean() > 0.1
    assert torch.allclose(output, expected, atol=2e-3, rtol=2e-3)


@LAYOUTS
def test_the_large_linears_stay_packed_and_what_runs_outside_a_matmul_does_not(
    monkeypatch, tmp_path: Path, sd_cpp: bool
) -> None:
    """stable-diffusion.cpp's qk norms are quantized, and still arrive unpacked."""
    model, _ = _load(monkeypatch, tmp_path, sd_cpp=sd_cpp)

    # The modulation Linears too: diffusers' own skip patterns (`norm`) would unpack every `norm1.linear`.
    for name in (
        "transformer_blocks.0.norm1.linear.weight",
        "transformer_blocks.2.norm1_context.linear.weight",
        "transformer_blocks.0.attn.to_q.weight",
        "transformer_blocks.0.attn2.to_out.0.weight",
    ):
        assert isinstance(model.get_parameter(name).data, GGMLTensor), name
    for name in ("pos_embed.proj.weight", "transformer_blocks.0.attn.norm_q.weight", "context_embedder.weight"):
        assert type(model.get_parameter(name).data) is torch.Tensor, name
    assert type(model.get_buffer("pos_embed.pos_embed")) is torch.Tensor


def test_a_bundled_vae_is_never_read(monkeypatch, tmp_path: Path) -> None:
    """stable-diffusion.cpp's files carry the VAE, and some the text encoders: reading them would cost their size in
    RAM for tensors the transformer drops."""
    read: list[dict[str, torch.Tensor]] = []
    real_reader = sd3.gguf_sd_loader
    monkeypatch.setattr(
        sd3, "gguf_sd_loader", lambda *args, **kwargs: read.append(real_reader(*args, **kwargs)) or read[-1]
    )

    _load(monkeypatch, tmp_path, sd_cpp=True)

    assert read and all(key.startswith("model.diffusion_model.") for key in read[0])
