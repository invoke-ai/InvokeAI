import logging
from pathlib import Path
from types import SimpleNamespace

import gguf
import numpy as np
import pytest

from invokeai.backend.model_manager.configs.t5_encoder import (
    T5Encoder_BnBLLMint8_Config,
    T5Encoder_SDNQ_Config,
    T5Encoder_T5Encoder_Config,
)
from invokeai.backend.model_manager.load.model_cache import model_cache as model_cache_module
from invokeai.backend.model_manager.load.model_cache.model_cache import ModelCache
from invokeai.backend.model_manager.load.model_cache.shared_cpu_weights import SharedCpuWeightsStore
from invokeai.backend.model_manager.load.model_loaders.flux import (
    BnbQuantizedLlmInt8bCheckpointModel,
    T5EncoderCheckpointModel,
    T5EncoderGGUFModel,
    T5EncoderSDNQLoader,
    T5EncoderSingleFileLoader,
)
from invokeai.backend.model_manager.load.model_loaders.gemma2_encoder import (
    Gemma2EncoderGGUFLoader,
    Gemma2EncoderLoader,
)
from invokeai.backend.model_manager.load.model_loaders.mistral_encoder import (
    MistralEncoderCheckpointLoader,
    MistralEncoderGGUFLoader,
)
from invokeai.backend.model_manager.load.model_loaders.z_image import (
    Qwen3EncoderCheckpointLoader,
    Qwen3EncoderGGUFLoader,
    Qwen3EncoderSDNQLoader,
)
from invokeai.backend.model_manager.load.model_util import calc_tokenizer_size_by_fs
from invokeai.backend.model_manager.taxonomy import SubModelType
from invokeai.backend.qwen3.qwen3_tokenizer import _TOKENIZER_DIR
from invokeai.backend.t5.t5_tokenizer import _TOKENIZER_DIR as _T5_TOKENIZER_DIR


def _file_of_size(path: Path, size: int) -> Path:
    path.write_bytes(b"\x00" * size)
    return path


def _loader(cls: type):
    return object.__new__(cls)


@pytest.mark.parametrize(
    ("loader_cls", "checkpoint_name"),
    [
        (MistralEncoderCheckpointLoader, "mistral.safetensors"),
        (MistralEncoderGGUFLoader, "mistral.gguf"),
        (Gemma2EncoderGGUFLoader, "gemma2.gguf"),
    ],
)
def test_tokenizer_only_single_file_does_not_size_checkpoint(loader_cls: type, checkpoint_name: str, tmp_path: Path):
    checkpoint = _file_of_size(tmp_path / checkpoint_name, 4096)
    loader = _loader(loader_cls)

    assert loader.get_size_fs(SimpleNamespace(), checkpoint, SubModelType.Tokenizer) == 0
    assert loader.get_size_fs(SimpleNamespace(), checkpoint, SubModelType.TextEncoder) == checkpoint.stat().st_size


@pytest.mark.parametrize("loader_cls", [Qwen3EncoderCheckpointLoader, Qwen3EncoderGGUFLoader, Qwen3EncoderSDNQLoader])
def test_qwen3_tokenizer_sizes_bundled_compressed_assets(loader_cls: type, tmp_path: Path):
    tokenizer_bytes = calc_tokenizer_size_by_fs(_TOKENIZER_DIR, include_runtime_overhead=True)
    checkpoint = _file_of_size(tmp_path / "qwen3.safetensors", 4096)
    loader = _loader(loader_cls)

    assert tokenizer_bytes > 0
    assert tokenizer_bytes > checkpoint.stat().st_size
    assert loader.get_size_fs(SimpleNamespace(), checkpoint, SubModelType.Tokenizer) == tokenizer_bytes


def test_bundled_qwen3_tokenizer_fits_bounded_host_admission(monkeypatch: pytest.MonkeyPatch):
    tokenizer_bytes = calc_tokenizer_size_by_fs(_TOKENIZER_DIR, include_runtime_overhead=True)
    monkeypatch.setattr(model_cache_module, "_host_memory_available", lambda: tokenizer_bytes)
    cache = ModelCache(
        execution_device_working_mem_gb=1.0,
        enable_partial_loading=False,
        keep_ram_copy_of_weights=True,
        execution_device="cpu",
        storage_device="cpu",
        logger=logging.getLogger("test-tokenizer-size-admission"),
        shared_cpu_weights=SharedCpuWeightsStore(),
    )
    try:
        result = cache.make_room(tokenizer_bytes)
        assert result.ram_headroom_satisfied
        assert not cache.make_room(tokenizer_bytes + 1).ram_headroom_satisfied
    finally:
        cache.shutdown()


def test_compressed_tokenizer_size_can_include_cold_parse_overhead(tmp_path: Path):
    _file_of_size(tmp_path / "tokenizer.json.gz", 128)
    _file_of_size(tmp_path / "tokenizer_config.json", 16)

    on_disk = calc_tokenizer_size_by_fs(tmp_path)
    cold = calc_tokenizer_size_by_fs(tmp_path, include_runtime_overhead=True)

    assert on_disk == 144
    assert cold == on_disk + 130 * 2**20


def test_mistral_tokenizer_sizes_local_assets_not_model_file(tmp_path: Path):
    from invokeai.backend.model_manager.load.model_loaders.mistral_encoder import MistralEncoderDiffusersLoader

    (tmp_path / "text_encoder").mkdir()
    checkpoint = _file_of_size(tmp_path / "text_encoder" / "model.safetensors", 4096)
    (tmp_path / "tokenizer").mkdir()
    (tmp_path / "tokenizer" / "tekken.json").write_bytes(b"tekken")
    (tmp_path / "tokenizer" / "README.md").write_bytes(b"not loaded")
    loader = _loader(MistralEncoderDiffusersLoader)

    assert loader.get_size_fs(SimpleNamespace(), tmp_path, SubModelType.Tokenizer) == len(b"tekken")
    assert loader.get_size_fs(SimpleNamespace(), tmp_path, SubModelType.TextEncoder) == checkpoint.stat().st_size


def test_mistral_embedded_gguf_tokenizer_size_uses_tensor_metadata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from invokeai.backend.model_manager.load.model_loaders import mistral_encoder

    checkpoint = tmp_path / "mistral.gguf"
    writer = gguf.GGUFWriter(str(checkpoint), "llama")
    writer.add_tensor("tekken_model", np.arange(17, dtype=np.float16))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    monkeypatch.setattr(mistral_encoder, "_extract_tekken_bytes", lambda _path: (_ for _ in ()).throw(AssertionError()))
    loader = _loader(MistralEncoderGGUFLoader)

    assert loader.get_size_fs(SimpleNamespace(), checkpoint, SubModelType.Tokenizer) == 17 * 8


def test_mistral_embedded_safetensors_tokenizer_size_uses_tensor_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    import torch
    from safetensors.torch import save_file

    from invokeai.backend.model_manager.load.model_loaders import mistral_encoder

    checkpoint = tmp_path / "mistral.safetensors"
    save_file({"tekken_model": torch.zeros(17, dtype=torch.uint8)}, str(checkpoint))

    monkeypatch.setattr(mistral_encoder, "_extract_tekken_bytes", lambda _path: (_ for _ in ()).throw(AssertionError()))
    loader = _loader(MistralEncoderCheckpointLoader)

    assert loader.get_size_fs(SimpleNamespace(), checkpoint, SubModelType.Tokenizer) == 17 * 8


def test_mistral_tokenizer_size_follows_root_tekken_fallback_with_empty_sibling(tmp_path: Path):
    from invokeai.backend.model_manager.load.model_loaders.mistral_encoder import MistralEncoderDiffusersLoader

    (tmp_path / "tokenizer").mkdir()
    (tmp_path / "tokenizer" / "tokenizer.json").write_bytes(b"sibling-hf-tokenizer")
    (tmp_path / "tekken.json").write_bytes(b"root-tekken")
    loader = _loader(MistralEncoderDiffusersLoader)

    assert loader.get_size_fs(SimpleNamespace(), tmp_path, SubModelType.Tokenizer) == len(b"root-tekken")


def test_mistral_tokenizer_size_skips_metadata_only_sibling_for_root_fallback(tmp_path: Path):
    from invokeai.backend.model_manager.load.model_loaders.mistral_encoder import MistralEncoderDiffusersLoader

    sibling = tmp_path / "tokenizer"
    sibling.mkdir()
    (sibling / "chat_template.json").write_bytes(b"{}")
    (tmp_path / "tokenizer.json").write_bytes(b"root-tokenizer" * 62)
    loader = _loader(MistralEncoderDiffusersLoader)

    assert loader.get_size_fs(SimpleNamespace(), tmp_path, SubModelType.Tokenizer) == (len(b"root-tokenizer") * 62)


def test_gemma_tokenizer_sizes_local_assets_not_model_file(tmp_path: Path):
    checkpoint = _file_of_size(tmp_path / "model.safetensors", 4096)
    (tmp_path / "config.json").write_text('{"architectures": ["Gemma2ForCausalLM"]}')
    (tmp_path / "tokenizer.json").write_bytes(b"tokenizer")
    (tmp_path / "tokenizer_config.json").write_bytes(b"config")
    loader = _loader(Gemma2EncoderLoader)

    assert loader.get_size_fs(SimpleNamespace(), tmp_path, SubModelType.Tokenizer) == len(b"tokenizer") + len(b"config")
    assert loader.get_size_fs(SimpleNamespace(), tmp_path, SubModelType.TextEncoder) == checkpoint.stat().st_size


@pytest.mark.parametrize("submodel_type", [SubModelType.Tokenizer2, SubModelType.Tokenizer3])
def test_t5_folder_tokenizer_sizes_its_local_assets(tmp_path: Path, submodel_type: SubModelType):
    tokenizer_dir = tmp_path / "tokenizer_2"
    tokenizer_dir.mkdir()
    (tokenizer_dir / "tokenizer.json").write_bytes(b"tokenizer")
    (tokenizer_dir / "tokenizer_config.json").write_bytes(b"config")
    loader = _loader(T5EncoderCheckpointModel)
    config = T5Encoder_T5Encoder_Config.model_construct(path=str(tmp_path), name="t5")

    assert loader.get_size_fs(config, tmp_path, submodel_type) == len(b"tokenizer") + len(b"config")


@pytest.mark.parametrize("submodel_type", [SubModelType.Tokenizer2, SubModelType.Tokenizer3])
def test_t5_folder_tokenizer_sizes_sentencepiece_vocab(tmp_path: Path, submodel_type: SubModelType):
    tokenizer_dir = tmp_path / "tokenizer_2"
    tokenizer_dir.mkdir()
    (tokenizer_dir / "spiece.model").write_bytes(b"sentencepiece-vocab")
    loader = _loader(T5EncoderCheckpointModel)
    config = T5Encoder_T5Encoder_Config.model_construct(path=str(tmp_path), name="t5")

    assert loader.get_size_fs(config, tmp_path, submodel_type) == len(b"sentencepiece-vocab")


@pytest.mark.parametrize("loader_cls", [BnbQuantizedLlmInt8bCheckpointModel])
@pytest.mark.parametrize("submodel_type", [SubModelType.Tokenizer2, SubModelType.Tokenizer3])
def test_bnb_t5_tokenizer_sizes_its_local_assets(loader_cls: type, submodel_type: SubModelType, tmp_path: Path):
    tokenizer_dir = tmp_path / "tokenizer_2"
    tokenizer_dir.mkdir()
    (tokenizer_dir / "tokenizer.json").write_bytes(b"tokenizer")
    loader = _loader(loader_cls)
    config = T5Encoder_BnBLLMint8_Config.model_construct(path=str(tmp_path), name="t5")

    assert loader.get_size_fs(config, tmp_path, submodel_type) == len(b"tokenizer")


@pytest.mark.parametrize("loader_cls", [T5EncoderSingleFileLoader, T5EncoderGGUFModel])
@pytest.mark.parametrize("submodel_type", [SubModelType.Tokenizer2, SubModelType.Tokenizer3])
def test_bundled_t5_tokenizer_sizes_its_assets(loader_cls: type, submodel_type: SubModelType, tmp_path: Path):
    loader = _loader(loader_cls)
    size = calc_tokenizer_size_by_fs(_T5_TOKENIZER_DIR)
    checkpoint = _file_of_size(tmp_path / "t5.safetensors", 4096)

    assert size > 0
    assert size > checkpoint.stat().st_size
    assert loader.get_size_fs(SimpleNamespace(path=str(checkpoint)), checkpoint, submodel_type) == size


def test_bundled_t5_tokenizer_size_reaches_ram_admission(monkeypatch: pytest.MonkeyPatch):
    size = calc_tokenizer_size_by_fs(_T5_TOKENIZER_DIR)
    monkeypatch.setattr(model_cache_module, "_host_memory_available", lambda: size)
    cache = ModelCache(
        execution_device_working_mem_gb=1.0,
        enable_partial_loading=False,
        keep_ram_copy_of_weights=True,
        execution_device="cpu",
        storage_device="cpu",
        logger=logging.getLogger("test-t5-tokenizer-size-admission"),
        shared_cpu_weights=SharedCpuWeightsStore(),
    )
    try:
        assert cache.make_room(size).ram_headroom_satisfied
        assert not cache.make_room(size + 1).ram_headroom_satisfied
    finally:
        cache.shutdown()


@pytest.mark.parametrize("submodel_type", [SubModelType.Tokenizer2, SubModelType.Tokenizer3])
@pytest.mark.parametrize("inline", [False, True], ids=["standalone", "inline"])
def test_sdnq_t5_tokenizer_sizes_sibling_assets(tmp_path: Path, submodel_type: SubModelType, inline: bool):
    text_encoder_dir = tmp_path / "text_encoder_2"
    text_encoder_dir.mkdir()
    (text_encoder_dir / "config.json").write_text("{}")
    tokenizer_dir = tmp_path / "tokenizer_2"
    tokenizer_dir.mkdir()
    (tokenizer_dir / "tokenizer.json").write_bytes(b"tokenizer")
    loader = _loader(T5EncoderSDNQLoader)
    config = T5Encoder_SDNQ_Config.model_construct(path=str(text_encoder_dir), name="t5")

    model_path = text_encoder_dir if inline else tmp_path
    assert loader.get_size_fs(config, model_path, submodel_type) == len(b"tokenizer")
