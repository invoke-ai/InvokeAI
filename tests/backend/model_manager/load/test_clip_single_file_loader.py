"""Loading a CLIP-L or CLIP-G text encoder from one safetensors file in ComfyUI's layout.

The loader builds the variant's vendored architecture; a tiny stand-in replaces it here, so the files are written
at that size. CLIP-L's file carries the `text_model.` prefix that transformers' flattened `CLIPTextModel` no longer
has, plus the `logit_scale` and projection fine-tuned exports keep; CLIP-G's is `CLIPTextModelWithProjection`'s own
state dict. Forwards run in float32: CPU bf16 matmuls fault on part of the hosted Windows fleet.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from safetensors.torch import save_file
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

import invokeai.backend.model_manager.load.model_loaders.flux as loader_module
from invokeai.backend.clip.clip_text_encoder import load_bundled_clip_tokenizer
from invokeai.backend.model_manager.configs.clip_embed import (
    CLIPEmbed_Checkpoint_G_Config,
    CLIPEmbed_Checkpoint_L_Config,
)
from invokeai.backend.model_manager.load.load_default import put_in_eval_mode
from invokeai.backend.model_manager.load.model_loaders.flux import CLIPSingleFileLoader
from invokeai.backend.model_manager.taxonomy import SubModelType

TINY = CLIPTextConfig(
    vocab_size=49408,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    max_position_embeddings=77,
    projection_dim=32,
    hidden_act="quick_gelu",
)


@pytest.fixture
def loader(monkeypatch: pytest.MonkeyPatch) -> CLIPSingleFileLoader:
    monkeypatch.setattr(loader_module, "clip_text_config", lambda _variant: TINY)
    loader = object.__new__(CLIPSingleFileLoader)
    loader._ram_cache = MagicMock()
    loader._torch_dtype = torch.float32
    return loader


def _reference(model_class: type) -> torch.nn.Module:
    torch.manual_seed(0)
    model = model_class(TINY).eval()
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(torch.randn_like(param) * 0.2)
    return model


def _input_ids() -> torch.Tensor:
    return load_bundled_clip_tokenizer()(
        ["a cat on a sofa"], padding="max_length", max_length=77, return_tensors="pt"
    ).input_ids


def test_clip_l_loads_from_the_prefixed_file_into_a_plain_text_model(
    loader: CLIPSingleFileLoader, tmp_path: Path
) -> None:
    reference = _reference(CLIPTextModel)
    file_sd = {f"text_model.{key}": value.half() for key, value in reference.state_dict().items()}
    file_sd["logit_scale"] = torch.tensor(4.6)
    file_sd["text_projection"] = torch.randn(32, 32).half()
    path = tmp_path / "clip_l.safetensors"
    save_file(file_sd, str(path))
    config = CLIPEmbed_Checkpoint_L_Config.model_construct(path=str(path), name="clip_l", cpu_only=None)

    model = put_in_eval_mode(loader._load_model(config, SubModelType.TextEncoder))

    # FLUX.1 asserts a plain CLIPTextModel and reads its pooled output.
    assert type(model) is CLIPTextModel
    expected = CLIPTextModel(TINY).eval()
    expected.load_state_dict({key: value.half().float() for key, value in reference.state_dict().items()})
    with torch.no_grad():
        torch.testing.assert_close(
            model(input_ids=_input_ids()).pooler_output, expected(input_ids=_input_ids()).pooler_output
        )
    loader._ram_cache.make_room.assert_called_once()


def test_clip_g_loads_with_its_projection(loader: CLIPSingleFileLoader, tmp_path: Path) -> None:
    reference = _reference(CLIPTextModelWithProjection)
    path = tmp_path / "clip_g.safetensors"
    save_file({key: value.half() for key, value in reference.state_dict().items()}, str(path))
    config = CLIPEmbed_Checkpoint_G_Config.model_construct(path=str(path), name="clip_g", cpu_only=None)

    model = put_in_eval_mode(loader._load_model(config, SubModelType.TextEncoder))

    # SD 3 conditions on the projected pooled output.
    assert type(model) is CLIPTextModelWithProjection
    expected = CLIPTextModelWithProjection(TINY).eval()
    expected.load_state_dict({key: value.half().float() for key, value in reference.state_dict().items()})
    with torch.no_grad():
        torch.testing.assert_close(
            model(input_ids=_input_ids()).text_embeds, expected(input_ids=_input_ids()).text_embeds
        )


def test_the_tokenizer_is_the_bundled_one_and_reserves_nothing(loader: CLIPSingleFileLoader, tmp_path: Path) -> None:
    path = tmp_path / "clip_l.safetensors"
    config = CLIPEmbed_Checkpoint_L_Config.model_construct(path=str(path), name="clip_l", cpu_only=None)

    tokenizer = loader._load_model(config, SubModelType.Tokenizer)

    assert isinstance(tokenizer, CLIPTokenizer)
    assert tokenizer.model_max_length == 77
    assert loader.get_size_fs(config, path, SubModelType.Tokenizer) == 0
