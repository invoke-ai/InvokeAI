from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from invokeai.backend.model_manager.configs.main import (
    Main_Checkpoint_Krea2_Config,
    Main_Diffusers_Krea2_Config,
    Main_GGUF_Krea2_Config,
)
from invokeai.backend.model_manager.configs.qwen3_vl_encoder import Qwen3VLEncoder_Qwen3VLEncoder_Config
from invokeai.backend.model_manager.load.model_loaders.krea2 import (
    Krea2CheckpointModel,
    Krea2DiffusersModel,
    Krea2GGUFCheckpointModel,
    Qwen3VLEncoderLoader,
)
from invokeai.backend.model_manager.taxonomy import Krea2VariantType, SubModelType


class _TinyKrea2Transformer(torch.nn.Module):
    def __init__(self, **_kwargs) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(2, 2))


def test_single_file_loader_constructs_and_materializes_model(monkeypatch, tmp_path) -> None:
    import diffusers
    import safetensors.torch

    checkpoint_path = tmp_path / "krea2.safetensors"
    checkpoint_path.touch()
    config = Main_Checkpoint_Krea2_Config.model_construct(
        path=str(checkpoint_path), variant=Krea2VariantType.Turbo, fp8_storage=None
    )
    ram_cache = SimpleNamespace(make_room=MagicMock())
    loader = object.__new__(Krea2CheckpointModel)
    loader._ram_cache = ram_cache
    loader._apply_fp8_layerwise_casting = lambda model, _config, _submodel: model

    monkeypatch.setattr(diffusers, "Krea2Transformer2DModel", _TinyKrea2Transformer, raising=False)
    monkeypatch.setattr(safetensors.torch, "load_file", lambda _path: {"weight": torch.ones(2, 2)})
    monkeypatch.setattr(
        "invokeai.backend.model_manager.load.model_loaders.krea2.TorchDevice.choose_torch_device",
        lambda: torch.device("cpu"),
    )
    monkeypatch.setattr(
        "invokeai.backend.model_manager.load.model_loaders.krea2.TorchDevice.choose_bfloat16_safe_dtype",
        lambda _device: torch.float32,
    )

    model = loader._load_from_singlefile(config)

    assert isinstance(model, _TinyKrea2Transformer)
    assert model.weight.device.type == "cpu"
    assert torch.equal(model.weight, torch.ones(2, 2))
    ram_cache.make_room.assert_called_once()


def test_diffusers_loader_reaches_transformer_from_pretrained(monkeypatch, tmp_path) -> None:
    config = Main_Diffusers_Krea2_Config.model_construct(path=str(tmp_path), repo_variant=None)
    loader = object.__new__(Krea2DiffusersModel)
    loaded_model = object()
    load_class = SimpleNamespace(from_pretrained=MagicMock(return_value=loaded_model))
    loader.get_hf_load_class = lambda _path, _submodel: load_class
    loader._apply_fp8_layerwise_casting = lambda model, _config, _submodel: model
    monkeypatch.setattr(
        "invokeai.backend.model_manager.load.model_loaders.krea2.TorchDevice.choose_torch_device",
        lambda: torch.device("cpu"),
    )
    monkeypatch.setattr(
        "invokeai.backend.model_manager.load.model_loaders.krea2.TorchDevice.choose_bfloat16_safe_dtype",
        lambda _device: torch.float32,
    )

    model = loader._load_model(config, SubModelType.Transformer)

    assert model is loaded_model
    load_class.from_pretrained.assert_called_once_with(
        tmp_path / "transformer", torch_dtype=torch.float32, variant=None
    )


def test_gguf_loader_constructs_and_materializes_model(monkeypatch, tmp_path) -> None:
    import diffusers

    checkpoint_path = tmp_path / "krea2.gguf"
    checkpoint_path.touch()
    config = Main_GGUF_Krea2_Config.model_construct(path=str(checkpoint_path), variant=Krea2VariantType.Turbo)
    loader = object.__new__(Krea2GGUFCheckpointModel)

    monkeypatch.setattr(diffusers, "Krea2Transformer2DModel", _TinyKrea2Transformer, raising=False)
    monkeypatch.setattr(
        "invokeai.backend.model_manager.load.model_loaders.krea2.gguf_sd_loader",
        lambda _path, *, compute_dtype: {"weight": torch.ones(2, 2, dtype=compute_dtype)},
    )
    monkeypatch.setattr(
        "invokeai.backend.model_manager.load.model_loaders.krea2.TorchDevice.choose_torch_device",
        lambda: torch.device("cpu"),
    )
    monkeypatch.setattr(
        "invokeai.backend.model_manager.load.model_loaders.krea2.TorchDevice.choose_bfloat16_safe_dtype",
        lambda _device: torch.float32,
    )

    model = loader._load_from_gguf(config)

    assert isinstance(model, _TinyKrea2Transformer)
    assert model.weight.device.type == "cpu"
    assert torch.equal(model.weight, torch.ones(2, 2))


def test_directory_encoder_loader_reaches_transformers_from_pretrained(monkeypatch, tmp_path) -> None:
    import transformers

    (tmp_path / "config.json").write_text("{}")
    config = Qwen3VLEncoder_Qwen3VLEncoder_Config.model_construct(path=str(tmp_path))
    loader = object.__new__(Qwen3VLEncoderLoader)
    text_config = SimpleNamespace(rope_parameters={"rope_type": "default"}, rope_scaling=None)
    encoder_config = SimpleNamespace(text_config=text_config)
    loaded_model = object()
    from_pretrained = MagicMock(return_value=loaded_model)

    monkeypatch.setattr(
        "invokeai.backend.model_manager.load.model_loaders.krea2.AutoConfig.from_pretrained",
        lambda *_args, **_kwargs: encoder_config,
    )
    monkeypatch.setattr(transformers.Qwen3VLModel, "from_pretrained", from_pretrained)
    monkeypatch.setattr(
        "invokeai.backend.model_manager.load.model_loaders.krea2.TorchDevice.choose_torch_device",
        lambda: torch.device("cpu"),
    )
    monkeypatch.setattr(
        "invokeai.backend.model_manager.load.model_loaders.krea2.TorchDevice.choose_bfloat16_safe_dtype",
        lambda _device: torch.float32,
    )

    model = loader._load_model(config, SubModelType.TextEncoder)

    assert model is loaded_model
    assert text_config.rope_scaling == text_config.rope_parameters
    from_pretrained.assert_called_once_with(
        tmp_path,
        config=encoder_config,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
