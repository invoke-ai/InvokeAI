"""A Qwen3 causal-LM folder identified as a Text LLM also loads as the Z-Image / FLUX.2 Klein / Anima Qwen3 encoder:
its view in that role goes through `Qwen3EncoderLoader` and yields the same language model and tokenizer."""

from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerBase, Qwen3Config, Qwen3ForCausalLM

from invokeai.backend.model_manager.configs.qwen3_encoder import Qwen3Encoder_Qwen3Encoder_Config
from invokeai.backend.model_manager.configs.roles import view_as
from invokeai.backend.model_manager.configs.text_llm import TextLLM_Diffusers_Config
from invokeai.backend.model_manager.load.model_loaders import z_image as loader_module
from invokeai.backend.model_manager.load.model_loaders.z_image import Qwen3EncoderLoader
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import ModelType, Qwen3VariantType, SubModelType
from tests.fixtures.model_roles import text_llm_qwen3_folder

# Identification reads the hidden size off config.json and needs a real Qwen3 width (1024 is the 0.6B); everything
# else is as small as the architecture allows, so the folder is a few hundred kilobytes.
_HIDDEN_SIZE = 1024


def _qwen3_folder(root: Path) -> Qwen3ForCausalLM:
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=32,
            hidden_size=_HIDDEN_SIZE,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            max_position_embeddings=64,
        )
    ).eval()
    model.save_pretrained(root)
    vocab = {"<unk>": 0, **{f"w{i}": i for i in range(1, 32)}}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(root / "tokenizer.json"))
    return model


@pytest.fixture
def loader(monkeypatch: pytest.MonkeyPatch) -> Qwen3EncoderLoader:
    monkeypatch.setattr(loader_module.TorchDevice, "choose_torch_device", staticmethod(lambda: torch.device("cpu")))
    monkeypatch.setattr(
        loader_module.TorchDevice, "choose_bfloat16_safe_dtype", staticmethod(lambda _device: torch.float32)
    )
    return object.__new__(Qwen3EncoderLoader)


def test_a_qwen3_folder_identified_as_a_text_llm_loads_as_the_qwen3_encoder_of_its_size(
    tmp_path: Path, loader: Qwen3EncoderLoader
) -> None:
    reference = _qwen3_folder(tmp_path)
    fields = {
        "key": "k",
        "hash": "blake3:x",
        "path": str(tmp_path),
        "file_size": 1,
        "name": "qwen3 folder",
        "source": "test",
        "source_type": "path",
    }
    record = TextLLM_Diffusers_Config.from_model_on_disk(ModelOnDisk(tmp_path), fields)

    view = view_as(record, ModelType.Qwen3Encoder)

    assert isinstance(view, Qwen3Encoder_Qwen3Encoder_Config)
    assert view.variant is Qwen3VariantType.Qwen3_06B
    model = loader._load_model(view, SubModelType.TextEncoder)
    assert isinstance(model, Qwen3ForCausalLM)
    for name, tensor in reference.state_dict().items():
        assert torch.equal(model.state_dict()[name], tensor), name
    assert isinstance(loader._load_model(view, SubModelType.Tokenizer), PreTrainedTokenizerBase)


def test_the_tokenizer_of_a_root_layout_folder_is_not_sized_as_the_whole_folder(
    tmp_path: Path, loader: Qwen3EncoderLoader
) -> None:
    _qwen3_folder(tmp_path)
    record = text_llm_qwen3_folder("k", Qwen3VariantType.Qwen3_06B).model_copy(update={"path": str(tmp_path)})
    view = view_as(record, ModelType.Qwen3Encoder)

    assert loader.get_size_fs(view, tmp_path, SubModelType.Tokenizer) == 0
    assert loader.get_size_fs(view, tmp_path, SubModelType.TextEncoder) > 0
