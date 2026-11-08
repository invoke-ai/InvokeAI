"""Anima GGUF transformers, driven over real (tiny) GGUF files through the loader and the module's forward.

The files are laid out the way the published conversions are (see `anima_gguf_fixture`). The loader is checked
by its forward, in float32, against a dense transformer holding the values the file dequantizes to -- so a dropped
norm, a misread 1-D Q8_0 tensor, a wrong prefix strip or a model built at the wrong depth shows as a different
output, not only as a missing key.
"""

from pathlib import Path

import pytest
import torch

from invokeai.backend.model_manager.configs.main import Main_Checkpoint_Anima_Config, Main_GGUF_Anima_Config
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
    apply_custom_layers_to_model,
)
from invokeai.backend.model_manager.load.model_loaders import anima
from invokeai.backend.model_manager.load.model_loaders.anima import AnimaCheckpointModel, AnimaGGUFModel
from invokeai.backend.model_manager.taxonomy import AnimaVariantType, SubModelType
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor
from tests.backend.model_manager.load.anima_gguf_fixture import (
    TINY_LLM_ADAPTER,
    TINY_TRANSFORMER_CONFIG,
    install_tiny_anima_geometry,
    tiny_anima_transformer,
    write_anima_gguf,
)
from tests.fixtures.loader_seams import Seam, prepare

SEAM = Seam(
    loader=AnimaGGUFModel,
    module=anima,
    entry="_load_model",
    patches_device=True,
    sets_torch_dtype=False,
    casts_fp8_storage=False,
)

# What stable-diffusion.cpp bundles beside the transformer in an all-in-one file.
_AIO_COMPONENTS = {
    "text_encoders.llm.model.layers.0.self_attn.q_proj.weight": torch.randn(64, 64),
    "first_stage_model.decoder.conv_in.weight": torch.randn(8, 4, 3, 3).to(torch.float16),
}


@pytest.fixture(autouse=True)
def _tiny_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    install_tiny_anima_geometry(monkeypatch)


def _load(monkeypatch: pytest.MonkeyPatch, path: Path) -> torch.nn.Module:
    config = Main_GGUF_Anima_Config.model_construct(path=str(path), name=path.stem, variant=AnimaVariantType.Qwen3)
    return prepare(SEAM, monkeypatch).load(config, SubModelType.Transformer)


def _forward(model: torch.nn.Module) -> torch.Tensor:
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        return model(
            torch.randn(1, TINY_TRANSFORMER_CONFIG["in_channels"], 1, 4, 4, generator=generator),
            torch.tensor([0.4]),
            torch.randn(1, 3, TINY_LLM_ADAPTER["dim"], generator=generator),
            t5xxl_ids=torch.randint(0, TINY_LLM_ADAPTER["vocab_size"], (1, 3), generator=generator),
        )


def _assert_computes_what_the_file_means(model: torch.nn.Module, meant: dict[str, torch.Tensor]) -> None:
    """Not bit-exact: the torch kernels multiply codes by their fp16 block scale in fp16, the reference reader
    in float32. A wrong tensor anywhere is off by far more."""
    num_blocks = len(model.blocks)
    reference = tiny_anima_transformer(num_blocks)
    reference.load_state_dict(meant)
    apply_custom_layers_to_model(model)

    expected = _forward(reference)
    assert expected.std() > 0.1, "a vacuous reference: the output does not depend on the weights"
    torch.testing.assert_close(_forward(model), expected, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("prefix", ["", "net.", "model.diffusion_model."], ids=["bare", "net", "all-in-one"])
def test_the_gguf_transformer_computes_what_its_file_means(monkeypatch, tmp_path, prefix) -> None:
    path = tmp_path / "anima.gguf"
    meant = write_anima_gguf(path, prefix=prefix, extra=_AIO_COMPONENTS if prefix else None)

    _assert_computes_what_the_file_means(_load(monkeypatch, path), meant)


def test_an_all_in_one_files_encoder_and_vae_are_never_read(monkeypatch, tmp_path) -> None:
    """They are the size of a second model; the loader node supplies both, so only the transformer is copied."""
    path = tmp_path / "anima-aio.gguf"
    write_anima_gguf(path, prefix="model.diffusion_model.", extra=_AIO_COMPONENTS)
    read: list[str] = []
    real_reader = anima.gguf_sd_loader

    def recording_reader(*args, **kwargs):
        sd = real_reader(*args, **kwargs)
        read.extend(sd)
        return sd

    monkeypatch.setattr(anima, "gguf_sd_loader", recording_reader)
    _load(monkeypatch, path)

    assert read and all(key.startswith("model.diffusion_model.") for key in read)


def test_an_all_in_one_file_is_sized_by_its_transformer_alone(monkeypatch, tmp_path) -> None:
    """What the cache makes room for before the load: the encoder and VAE beside it are never read."""
    aio, bare = tmp_path / "anima-aio.gguf", tmp_path / "anima.gguf"
    write_anima_gguf(aio, prefix="model.diffusion_model.", extra=_AIO_COMPONENTS)
    write_anima_gguf(bare)
    loader = prepare(SEAM, monkeypatch).loader

    assert loader.get_size_fs(None, aio) == loader.get_size_fs(None, bare) > 0


def test_a_depth_expanded_export_is_built_at_its_own_depth(monkeypatch, tmp_path) -> None:
    """Like Anima-2.9B: more blocks than the release, and the RoPE buffers its export serializes beside them."""
    path = tmp_path / "anima-deep.gguf"
    buffers = {
        "net.pos_embedder.seq": torch.arange(16, dtype=torch.float32),
        "net.pos_embedder.dim_spatial_range": torch.rand(5),
        "net.pos_embedder.dim_temporal_range": torch.rand(6),
    }
    meant = write_anima_gguf(path, prefix="net.", num_blocks=3, extra=buffers)

    model = _load(monkeypatch, path)

    assert len(model.blocks) == 3
    _assert_computes_what_the_file_means(model, meant)


def test_only_the_quantized_linears_stay_packed(monkeypatch, tmp_path) -> None:
    path = tmp_path / "anima.gguf"
    write_anima_gguf(path)

    model = _load(monkeypatch, path)

    for name in ("blocks.0.mlp.layer1.weight", "blocks.1.cross_attn.k_proj.weight", "llm_adapter.out_proj.weight"):
        assert isinstance(model.get_parameter(name).data, GGMLTensor), name
    # The embedding and the norms (one of them a 1-D Q8_0) are read outside a matmul; the BF16 Linear and the
    # F32 bias are no larger unpacked.
    for name in (
        "llm_adapter.embed.weight",
        "llm_adapter.norm.weight",
        "blocks.1.self_attn.k_norm.weight",
        "t_embedder.1.linear_1.weight",
        "llm_adapter.out_proj.bias",
    ):
        assert type(model.get_parameter(name).data) is torch.Tensor, name


def test_an_old_checkpoint_record_of_a_gguf_file_asks_for_re_identification(monkeypatch, tmp_path) -> None:
    """Releases before the GGUF config recorded these files as checkpoints; read as safetensors they fail on a
    header error that names neither the cause nor the fix."""
    path = tmp_path / "anima-turbo-v1.1-Q8_0.gguf"
    write_anima_gguf(path)
    config = Main_Checkpoint_Anima_Config.model_construct(path=str(path), variant=AnimaVariantType.Qwen3)
    seam = Seam(loader=AnimaCheckpointModel, module=anima, patches_device=True)

    with pytest.raises(ValueError, match="Re-identify model"):
        prepare(seam, monkeypatch).load(config)
