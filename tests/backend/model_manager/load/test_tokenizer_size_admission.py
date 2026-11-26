import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import gguf
import numpy as np
import pytest
import torch

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
    loader = object.__new__(cls)
    loader._torch_dtype = torch.float32
    loader._torch_device = torch.device("cpu")
    return loader


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
def test_qwen3_tokenizer_sizes_bundled_compressed_assets(
    loader_cls: type, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    from invokeai.backend.qwen3.qwen3_tokenizer import load_bundled_qwen3_tokenizer

    monkeypatch.setattr(load_bundled_qwen3_tokenizer, "cache_info", lambda: SimpleNamespace(currsize=0))
    tokenizer_bytes = calc_tokenizer_size_by_fs(_TOKENIZER_DIR)
    checkpoint = _file_of_size(tmp_path / "qwen3.safetensors", 4096)
    loader = _loader(loader_cls)

    assert tokenizer_bytes > 0
    assert tokenizer_bytes > checkpoint.stat().st_size
    assert loader.get_size_fs(SimpleNamespace(), checkpoint, SubModelType.Tokenizer) == tokenizer_bytes
    assert loader._get_host_bytes_needed(
        SimpleNamespace(), checkpoint, SubModelType.Tokenizer
    ) == calc_tokenizer_size_by_fs(_TOKENIZER_DIR, include_runtime_overhead=True)


def test_qwen3_memoized_tokenizer_is_not_admitted_twice(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from invokeai.backend.qwen3.qwen3_tokenizer import load_bundled_qwen3_tokenizer

    monkeypatch.setattr(load_bundled_qwen3_tokenizer, "cache_info", lambda: SimpleNamespace(currsize=1))
    checkpoint = _file_of_size(tmp_path / "qwen3.safetensors", 4096)
    loader = _loader(Qwen3EncoderCheckpointLoader)

    assert loader.get_size_fs(SimpleNamespace(), checkpoint, SubModelType.Tokenizer) == 0
    assert loader._get_host_bytes_needed(SimpleNamespace(), checkpoint, SubModelType.Tokenizer) == 0


def test_qwen3_vl_processor_sizes_its_cold_tokenizer_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from invokeai.backend.model_manager.configs.qwen3_vl_encoder import Qwen3VLEncoder_GGUF_Config
    from invokeai.backend.model_manager.load.model_loaders.krea2 import Qwen3VLEncoderGGUFLoader
    from invokeai.backend.qwen3_vl.qwen3_vl_assets import (
        _TOKENIZER_DIR as qwen3vl_tokenizer_dir,
    )
    from invokeai.backend.qwen3_vl.qwen3_vl_assets import (
        load_bundled_qwen3_vl_tokenizer,
    )

    monkeypatch.setattr(load_bundled_qwen3_vl_tokenizer, "cache_info", lambda: SimpleNamespace(currsize=0))

    checkpoint = _file_of_size(tmp_path / "qwen3vl.gguf", 4096)
    loader = _loader(Qwen3VLEncoderGGUFLoader)
    config = Qwen3VLEncoder_GGUF_Config.model_construct(path=str(checkpoint), name="qwen3vl")

    assert loader.get_size_fs(config, checkpoint, SubModelType.Processor) == calc_tokenizer_size_by_fs(
        qwen3vl_tokenizer_dir
    )
    assert loader._get_host_bytes_needed(config, checkpoint, SubModelType.Processor) == calc_tokenizer_size_by_fs(
        qwen3vl_tokenizer_dir, include_runtime_overhead=True
    )


def test_qwen3_vl_memoized_processor_does_not_recharge_tokenizer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from invokeai.backend.model_manager.load.model_loaders.krea2 import Qwen3VLEncoderGGUFLoader
    from invokeai.backend.qwen3_vl.qwen3_vl_assets import load_bundled_qwen3_vl_tokenizer

    monkeypatch.setattr(load_bundled_qwen3_vl_tokenizer, "cache_info", lambda: SimpleNamespace(currsize=1))
    checkpoint = _file_of_size(tmp_path / "qwen3vl.gguf", 4096)
    loader = _loader(Qwen3VLEncoderGGUFLoader)

    assert loader.get_size_fs(SimpleNamespace(), checkpoint, SubModelType.Processor) == 0
    assert loader._get_host_bytes_needed(SimpleNamespace(), checkpoint, SubModelType.Processor) == 0


def test_qwen3_vl_flat_processor_sizes_tokenizer_not_encoder(tmp_path: Path):
    from invokeai.backend.model_manager.configs.qwen3_vl_encoder import Qwen3VLEncoder_Qwen3VLEncoder_Config
    from invokeai.backend.model_manager.load.model_loaders.krea2 import Qwen3VLEncoderLoader

    (tmp_path / "config.json").write_text("{}")
    _file_of_size(tmp_path / "model.safetensors", 4096)
    (tmp_path / "tokenizer.json").write_bytes(b"selected tokenizer")
    loader = _loader(Qwen3VLEncoderLoader)
    config = Qwen3VLEncoder_Qwen3VLEncoder_Config.model_construct(path=str(tmp_path), name="qwen3vl")

    assert loader.get_size_fs(config, tmp_path, SubModelType.Processor) == len(b"selected tokenizer")


def test_qwen3_5_checkpoint_sizes_tokenizer_and_processor_assets(tmp_path: Path):
    from invokeai.backend.model_manager.configs.text_llm import TextLLM_Checkpoint_Config
    from invokeai.backend.model_manager.load.model_loaders.text_llm import TextLLMCheckpointLoader
    from invokeai.backend.qwen3_5.qwen3_5_encoder import _TOKENIZER_DIR as qwen35_tokenizer_dir

    checkpoint = _file_of_size(tmp_path / "qwen35.safetensors", 4096)
    loader = _loader(TextLLMCheckpointLoader)
    config = TextLLM_Checkpoint_Config.model_construct(path=str(checkpoint), name="qwen35")
    expected = calc_tokenizer_size_by_fs(qwen35_tokenizer_dir)

    assert loader.get_size_fs(config, checkpoint, SubModelType.Tokenizer) == expected
    assert loader.get_size_fs(config, checkpoint, SubModelType.Processor) == expected
    assert loader._get_host_bytes_needed(config, checkpoint, SubModelType.Processor) == calc_tokenizer_size_by_fs(
        qwen35_tokenizer_dir, include_runtime_overhead=True
    )


@pytest.mark.parametrize("location", ["tokenizer/tekken.json", "tekken.json"])
def test_mistral_local_tekken_admission_covers_parser_peak(tmp_path: Path, location: str):
    from invokeai.backend.model_manager.load.model_loaders.mistral_encoder import MistralEncoderDiffusersLoader

    tekken = tmp_path / location
    tekken.parent.mkdir(parents=True, exist_ok=True)
    _file_of_size(tekken, 18 * 2**20)
    loader = _loader(MistralEncoderDiffusersLoader)

    assert loader.get_size_fs(SimpleNamespace(), tmp_path, SubModelType.Tokenizer) == 18 * 2**20
    assert loader._get_host_bytes_needed(SimpleNamespace(), tmp_path, SubModelType.Tokenizer) == 102 * 2**20


def test_qwen3_5_encoder_cached_tokenizer_has_no_second_cold_charge(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from invokeai.backend.model_manager.configs.qwen3_5_encoder import Qwen35Encoder_Checkpoint_Config
    from invokeai.backend.model_manager.load.model_loaders.qwen3_5_encoder import Qwen35EncoderCheckpointLoader
    from invokeai.backend.qwen3_5.qwen3_5_encoder import load_bundled_qwen3_5_tokenizer

    monkeypatch.setattr(load_bundled_qwen3_5_tokenizer, "cache_info", lambda: SimpleNamespace(currsize=1))
    checkpoint = _file_of_size(tmp_path / "qwen35.safetensors", 4096)
    config = Qwen35Encoder_Checkpoint_Config.model_construct(path=str(checkpoint), name="qwen35")
    loader = _loader(Qwen35EncoderCheckpointLoader)

    assert loader.get_size_fs(config, checkpoint, SubModelType.Tokenizer) == 0
    assert loader._get_host_bytes_needed(config, checkpoint, SubModelType.Tokenizer) == 0


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

    assert loader.get_size_fs(SimpleNamespace(), checkpoint, SubModelType.Tokenizer) == 17
    assert loader._get_host_bytes_needed(SimpleNamespace(), checkpoint, SubModelType.Tokenizer) == 17 * 8


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

    assert loader.get_size_fs(SimpleNamespace(), checkpoint, SubModelType.Tokenizer) == 17
    assert loader._get_host_bytes_needed(SimpleNamespace(), checkpoint, SubModelType.Tokenizer) == 17 * 8


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
    assert loader._get_host_bytes_needed(
        SimpleNamespace(), tmp_path, SubModelType.Tokenizer
    ) == calc_tokenizer_size_by_fs(tmp_path, include_runtime_overhead=True)


def test_mistral_tokenizer_host_estimate_follows_sibling_hf_selection(tmp_path: Path):
    from invokeai.backend.model_manager.load.model_loaders.mistral_encoder import MistralEncoderDiffusersLoader

    sibling = tmp_path / "tokenizer"
    sibling.mkdir()
    (sibling / "tokenizer.json").write_bytes(b"selected-sibling-vocabulary" * 20)
    (tmp_path / "chat_template.json").write_bytes(b"{}")
    loader = _loader(MistralEncoderDiffusersLoader)
    expected = calc_tokenizer_size_by_fs(sibling, include_runtime_overhead=True)

    assert loader.get_size_fs(SimpleNamespace(), tmp_path, SubModelType.Tokenizer) == expected
    assert loader._get_host_bytes_needed(SimpleNamespace(), tmp_path, SubModelType.Tokenizer) == expected


def test_mistral_checkpoint_without_embedded_tokenizer_admits_hf_fallback_before_parser(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from safetensors.torch import save_file

    from invokeai.backend.model_manager.configs.mistral_encoder import MistralEncoder_Checkpoint_Config
    from invokeai.backend.model_manager.load.model_loaders import mistral_encoder
    from invokeai.backend.model_manager.taxonomy import MistralVariantType

    checkpoint = tmp_path / "mistral_without_tekken.safetensors"
    save_file({"model.embed_tokens.weight": torch.ones(1)}, checkpoint)
    config = MistralEncoder_Checkpoint_Config.model_construct(
        path=str(checkpoint), name="mistral", key="mistral", variant=MistralVariantType.Cow
    )
    loader = _loader(MistralEncoderCheckpointLoader)
    loader._app_config = SimpleNamespace(log_memory_usage=False)
    loader._logger = MagicMock()
    loader.config_check = None
    loader._get_model_path = lambda model_config: Path(model_config.path)
    hf_fallback_parser = MagicMock(return_value=object())
    monkeypatch.setattr(mistral_encoder, "_load_tokenizer_from_hf", hf_fallback_parser)

    def make_room(_size: int, *, host_bytes_needed: int = 0):
        return SimpleNamespace(ram_headroom_satisfied=host_bytes_needed == 0)

    loader._ram_cache = SimpleNamespace(
        get_with_first_use_claim=MagicMock(side_effect=[IndexError, IndexError, (object(), None)]),
        shared_cpu_weights=None,
        make_room=MagicMock(side_effect=make_room),
        put=MagicMock(return_value=None),
    )

    assert loader.get_size_fs(config, checkpoint, SubModelType.Tokenizer) == 0

    # The real model-path wrapper must recognize this valid checkpoint as having no embedded
    # Tekken payload and dispatch to the canonical HuggingFace fallback.
    fallback_tokenizer = object()
    hf_fallback_parser.return_value = fallback_tokenizer
    assert (
        mistral_encoder._load_tokenizer_for_model(checkpoint, loader._logger, MistralVariantType.Cow)
        is fallback_tokenizer
    )
    hf_fallback_parser.assert_called_once()
    hf_fallback_parser.reset_mock()

    with pytest.raises(MemoryError, match="host memory"):
        loader._load_and_cache(config, SubModelType.Tokenizer)

    assert loader._ram_cache.make_room.call_args.kwargs["host_bytes_needed"] > 0
    hf_fallback_parser.assert_not_called()


def test_mistral_diffusers_without_local_vocabulary_admits_hf_fallback_before_parser(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from safetensors.torch import save_file

    from invokeai.backend.model_manager.configs.mistral_encoder import MistralEncoder_Diffusers_Config
    from invokeai.backend.model_manager.load.model_loaders import mistral_encoder
    from invokeai.backend.model_manager.load.model_loaders.mistral_encoder import MistralEncoderDiffusersLoader
    from invokeai.backend.model_manager.taxonomy import MistralVariantType

    model_dir = tmp_path / "mistral"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"architectures": ["MistralModel"], "model_type": "mistral"}')
    (model_dir / "chat_template.json").write_bytes(b"metadata-only")
    save_file({"model.embed_tokens.weight": torch.ones(1)}, model_dir / "model.safetensors")
    config = MistralEncoder_Diffusers_Config.model_construct(
        path=str(model_dir), name="mistral", key="mistral", variant=MistralVariantType.Cow
    )
    loader = _loader(MistralEncoderDiffusersLoader)
    loader._app_config = SimpleNamespace(log_memory_usage=False)
    loader._logger = MagicMock()
    loader.config_check = None
    loader._get_model_path = lambda model_config: Path(model_config.path)
    hf_fallback_parser = MagicMock(return_value=object())
    monkeypatch.setattr(mistral_encoder, "_load_tokenizer_from_hf", hf_fallback_parser)

    # The actual wrapper has no embedded or local tokenizer asset to select, so it falls through to
    # the HF parser. The parser is stubbed because this test covers admission, not network loading.
    fallback_tokenizer = mistral_encoder._load_tokenizer_for_model(model_dir, loader._logger, MistralVariantType.Cow)
    assert fallback_tokenizer is hf_fallback_parser.return_value
    hf_fallback_parser.assert_called_once()
    hf_fallback_parser.reset_mock()

    expected_cold_fallback_bytes = max(calc_tokenizer_size_by_fs(model_dir, include_runtime_overhead=True), 102 * 2**20)
    assert loader._get_host_bytes_needed(config, model_dir, SubModelType.Tokenizer) == expected_cold_fallback_bytes

    def make_room(_size: int, *, host_bytes_needed: int = 0):
        return SimpleNamespace(ram_headroom_satisfied=host_bytes_needed == 0)

    loader._ram_cache = SimpleNamespace(
        get_with_first_use_claim=MagicMock(side_effect=[IndexError, IndexError, (object(), None)]),
        shared_cpu_weights=None,
        make_room=MagicMock(side_effect=make_room),
        put=MagicMock(return_value=None),
    )

    with pytest.raises(MemoryError, match="host memory"):
        loader._load_and_cache(config, SubModelType.Tokenizer)

    assert loader._ram_cache.make_room.call_args.kwargs["host_bytes_needed"] > 0
    hf_fallback_parser.assert_not_called()


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
