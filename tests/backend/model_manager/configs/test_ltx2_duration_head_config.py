"""Identification tests for the LTX-2 duration head.

The head ships as a bare 19-tensor safetensors file with no config beside it and no marker in its
metadata, so identification rests entirely on its key set. Two things have to hold: a real head is
recognised, and a file that carries only part of one is not -- a partially matched head would load
with uninitialised weights and answer every prompt with a confident, meaningless duration rather
than fail.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch
from safetensors.torch import save_file

from invokeai.backend.ltx2.duration_head_state_dict_utils import is_state_dict_likely_ltx2_duration_head
from invokeai.backend.model_manager.configs.factory import ModelConfigFactory
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType

# The published head, at its real widths: video connector 4096, audio connector 2048, pooled 256.
_POOLED, _VIDEO, _AUDIO = 256, 4096, 2048


def _duration_head_state_dict() -> dict[str, torch.Tensor]:
    sd: dict[str, torch.Tensor] = {
        "video_input_proj.weight": torch.zeros(_POOLED, _VIDEO),
        "video_input_proj.bias": torch.zeros(_POOLED),
        "video_modality_emb": torch.zeros(_POOLED),
        "audio_input_proj.weight": torch.zeros(_POOLED, _AUDIO),
        "audio_input_proj.bias": torch.zeros(_POOLED),
        "audio_modality_emb": torch.zeros(_POOLED),
        "attention_pooler.query_tokens": torch.zeros(1, _POOLED),
        "mlp_hidden.weight": torch.zeros(_POOLED, _POOLED),
        "mlp_hidden.bias": torch.zeros(_POOLED),
        "mlp_out.weight": torch.zeros(1, _POOLED),
        "mlp_out.bias": torch.zeros(1),
    }
    for projection in ("to_q", "to_k", "to_v", "to_out"):
        sd[f"attention_pooler.{projection}.weight"] = torch.zeros(_POOLED, _POOLED)
        sd[f"attention_pooler.{projection}.bias"] = torch.zeros(_POOLED)
    return sd


def _identify(sd: dict[str, torch.Tensor]) -> object:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "ltx-2.5-duration-head.safetensors"
        save_file(sd, path)
        return ModelConfigFactory.from_model_on_disk(path, {}).config


def test_a_duration_head_is_identified_as_one() -> None:
    config = _identify(_duration_head_state_dict())

    assert config.type is ModelType.LTX2DurationHead
    assert config.base is BaseModelType.LTX2
    assert config.format is ModelFormat.Checkpoint


@pytest.mark.parametrize(
    "dropped",
    ["mlp_out.weight", "attention_pooler.query_tokens", "audio_input_proj.weight", "video_modality_emb"],
)
def test_a_head_missing_any_of_its_tensors_is_not_identified_as_one(dropped: str) -> None:
    """A truncated or edited file must not be installable: the missing tensor would stay random."""
    sd = _duration_head_state_dict()
    del sd[dropped]

    assert not is_state_dict_likely_ltx2_duration_head(sd)
    assert _identify(sd).type is not ModelType.LTX2DurationHead


def test_a_head_carrying_an_unknown_tensor_is_refused() -> None:
    """A later head with a tensor this version cannot place must be refused at install.

    The loader is strict both ways, so admitting it here would install cleanly and then fail at the
    first generation with "Unexpected keys" -- after a download and a model pick.
    """
    sd = _duration_head_state_dict()
    # Deliberately not a `.scale` suffix: that makes the file look SDNQ-quantized, and the
    # state-dict reader folds those away before identification ever sees them.
    sd["second_pooler.query_tokens"] = torch.zeros(1, _POOLED)

    assert not is_state_dict_likely_ltx2_duration_head(sd)
    assert _identify(sd).type is not ModelType.LTX2DurationHead


def test_the_pooler_alone_is_not_a_duration_head() -> None:
    """Both modality projections are what distinguish the head; an attention pooler on its own is not one."""
    sd = {k: v for k, v in _duration_head_state_dict().items() if k.startswith("attention_pooler.")}

    assert not is_state_dict_likely_ltx2_duration_head(sd)


@pytest.mark.parametrize(
    ("key", "shape"),
    [
        ("video_input_proj.weight", (_POOLED, 3840)),
        ("audio_input_proj.weight", (_POOLED, 1024)),
        ("mlp_out.weight", (2, _POOLED)),
    ],
)
def test_a_head_at_other_widths_is_not_identified(key: str, shape: tuple[int, int]) -> None:
    """Same keys at widths the loader does not build: refuse at install, not at the first generation."""
    sd = _duration_head_state_dict()
    sd[key] = torch.zeros(*shape)

    assert not is_state_dict_likely_ltx2_duration_head(sd)
    assert _identify(sd).type is not ModelType.LTX2DurationHead
