"""Loading Ministral 3B, the single-file encoder ERNIE-Image encodes prompts with.

The variant decides the model class. Ministral 3B is not a smaller Mistral Small 3: it applies
YaRN RoPE and a position-dependent attention scale that only ``Ministral3Model`` implements, and
its weights fit ``MistralModel`` well enough to load — so getting the class wrong produces an
encoder that loads without complaint and encodes garbage. These drive the loader itself, off real
safetensors files, and check what comes out the other end.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from safetensors.torch import save_file
from transformers import Ministral3Config, Ministral3Model, MistralConfig, MistralModel

import invokeai.backend.model_manager.load.model_loaders.mistral_encoder as loader_module
from invokeai.backend.model_manager.configs.mistral_encoder import MistralEncoder_Checkpoint_Config
from invokeai.backend.model_manager.load.model_loaders.mistral_encoder import (
    MistralEncoderCheckpointLoader,
    _build_ministral3_config,
)
from invokeai.backend.model_manager.taxonomy import MistralVariantType

# The released RoPE settings, from `baidu/ERNIE-Image`'s `text_encoder/config.json`. The loader
# leaves these to `Ministral3Config`'s defaults rather than restating them; this is the copy that
# notices if a transformers upgrade moves them.
RELEASED_ROPE = {
    "rope_type": "yarn",
    "rope_theta": 1000000.0,
    "factor": 16.0,
    "original_max_position_embeddings": 16384,
    "beta_fast": 32.0,
    "beta_slow": 1.0,
    "mscale": 1.0,
    "mscale_all_dim": 1.0,
    "llama_4_scaling_beta": 0.1,
}
RELEASED_MAX_POSITION_EMBEDDINGS = 262144

# Toy geometry. `head_dim` must stay 128 because the loader reads the head counts off the
# projections against exactly that width -- it is the one number a Mistral-family checkpoint does
# not reveal.
_HEAD_DIM = 128
_GEOMETRY = {
    "vocab_size": 512,
    "hidden_size": 128,
    "intermediate_size": 256,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": _HEAD_DIM,
}


@pytest.fixture(autouse=True)
def _cpu_float32(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the loader to CPU/float32 so the round-trip below is exact on any machine."""
    monkeypatch.setattr(loader_module.TorchDevice, "choose_torch_device", staticmethod(lambda: torch.device("cpu")))
    monkeypatch.setattr(
        loader_module.TorchDevice, "choose_bfloat16_safe_dtype", staticmethod(lambda _device: torch.float32)
    )


def _write_checkpoint(path: Path, reference: torch.nn.Module, *, with_vision: bool = False) -> Path:
    """Save a model the way a redistribution ships it: `model.`-prefixed keys, extras alongside."""
    state_dict = {f"model.{key}": value.contiguous() for key, value in reference.state_dict().items()}
    if with_vision:
        # Comfy-Org's Ministral file carries the whole multimodal stack; the encoder needs none of it.
        state_dict["vision_tower.transformer.layers.0.attention.q_proj.weight"] = torch.zeros(4, 4)
        state_dict["vision_tower.patch_conv.weight"] = torch.zeros(4, 4)
        state_dict["multi_modal_projector.linear_1.weight"] = torch.zeros(4, 4)
    save_file(state_dict, str(path))
    return path


def _load(path: Path, variant: MistralVariantType) -> torch.nn.Module:
    config = MistralEncoder_Checkpoint_Config.model_construct(path=str(path), name=path.stem, variant=variant)
    loader = object.__new__(MistralEncoderCheckpointLoader)
    # The loader reserves before it folds or casts. Nothing here is about the size it asks for, but the
    # reservation is sized from the model it just built, so a variant the prediction cannot walk fails here.
    loader._ram_cache = SimpleNamespace(make_room=MagicMock())
    return loader._load_text_encoder(config)


def test_ministral_3b_loads_as_a_ministral_model_with_the_released_rope(tmp_path: Path) -> None:
    torch.manual_seed(0)
    reference = Ministral3Model(Ministral3Config(**_GEOMETRY)).eval()
    path = _write_checkpoint(tmp_path / "ministral-3-3b.safetensors", reference, with_vision=True)

    model = _load(path, MistralVariantType.Ministral3B)

    assert isinstance(model, Ministral3Model)
    assert model.config.max_position_embeddings == RELEASED_MAX_POSITION_EMBEDDINGS
    assert {key: model.config.rope_parameters[key] for key in RELEASED_ROPE} == RELEASED_ROPE
    # The YaRN ratio is derived, not configured: it has to come out at the configured factor.
    assert (
        model.config.max_position_embeddings / model.config.rope_parameters["original_max_position_embeddings"]
        == model.config.rope_parameters["factor"]
    )


def test_the_weights_survive_the_load_and_the_encoder_reproduces_the_reference(tmp_path: Path) -> None:
    """A model that loads is not the point; one that encodes the same thing is."""
    torch.manual_seed(0)
    reference = Ministral3Model(Ministral3Config(**_GEOMETRY)).eval()
    path = _write_checkpoint(tmp_path / "ministral-3-3b.safetensors", reference, with_vision=True)

    model = _load(path, MistralVariantType.Ministral3B).eval()

    assert [name for name, param in model.named_parameters() if param.is_meta] == []
    input_ids = torch.tensor([[1, 7, 23, 99, 4]])
    with torch.no_grad():
        expected = reference(input_ids=input_ids, output_hidden_states=True).hidden_states[-2]
        actual = model(input_ids=input_ids, output_hidden_states=True).hidden_states[-2]
    assert torch.equal(actual, expected)


def test_the_vision_tower_is_left_behind(tmp_path: Path) -> None:
    """222 tensors and ~0.9GB of the released file are a Pixtral vision tower the encoder-only
    model has no slot for. Carrying them through the cast would cost real memory and then report
    them as unexpected keys."""
    torch.manual_seed(0)
    reference = Ministral3Model(Ministral3Config(**_GEOMETRY))
    path = _write_checkpoint(tmp_path / "ministral-3-3b.safetensors", reference, with_vision=True)

    model = _load(path, MistralVariantType.Ministral3B)

    assert getattr(model, "vision_tower", None) is None
    assert getattr(model, "multi_modal_projector", None) is None
    assert not [name for name, _ in model.named_parameters() if "vision" in name or "multi_modal" in name]


def test_a_flux2_encoder_still_loads_as_a_mistral_model(tmp_path: Path) -> None:
    """The branch must not capture the encoders that were already installing."""
    torch.manual_seed(0)
    reference = MistralModel(MistralConfig(**_GEOMETRY))
    path = _write_checkpoint(tmp_path / "cow-mistral3-small.safetensors", reference)

    model = _load(path, MistralVariantType.Cow)

    assert isinstance(model, MistralModel)
    assert not isinstance(model, Ministral3Model)


def test_geometry_that_cannot_be_read_is_refused_rather_than_guessed() -> None:
    """`_build_mistral_config` substitutes nominal geometry when a projection is missing, which
    suits the GGUFs it also serves. Here a guess would build a model that loads and encodes
    garbage, so the shapes are required."""
    torch.manual_seed(0)
    complete = {
        f"model.{key}": value for key, value in Ministral3Model(Ministral3Config(**_GEOMETRY)).state_dict().items()
    }
    without_gate = {key: value for key, value in complete.items() if "mlp.gate_proj" not in key}

    with pytest.raises(ValueError, match="mlp.gate_proj.weight"):
        _build_ministral3_config(without_gate, torch_dtype=torch.float32)
