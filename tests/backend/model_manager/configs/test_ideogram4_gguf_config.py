"""Identification of community Ideogram 4 GGUF transformers.

Run end to end through `ModelConfigFactory` over real (tiny) files, because the two questions that
matter are about the whole candidate set rather than one class: that exactly one config claims a
GGUF -- the safetensors config sees the same four fingerprint keys -- and which branch it records.

No published GGUF carries metadata, so the branch comes from the filename alone. The names below are
the releases' own, one pair per naming scheme in use.
"""

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from invokeai.backend.model_manager.configs.factory import ModelConfigFactory
from invokeai.backend.model_manager.configs.main import Main_Checkpoint_Ideogram4_Config, Main_GGUF_Ideogram4_Config
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from tests.backend.model_manager.load.ideogram4_gguf_fixture import packs_block_linears, write_ideogram4_gguf


def _identify(path: Path):
    return ModelConfigFactory.from_model_on_disk(ModelOnDisk(path), {}, allow_unknown=True)


@pytest.mark.parametrize(
    ("filename", "branch"),
    [
        # molbal
        ("ideogram4-transformer-q4_0.gguf", "conditional"),
        ("ideogram4-unconditional_transformer-q4_0.gguf", "unconditional"),
        # leejet
        ("ideogram4-Q4_0.gguf", "conditional"),
        ("ideogram4_uncond-Q4_0.gguf", "unconditional"),
        # stduhpf
        ("ideogram4-Q4_K.gguf", "conditional"),
        ("ideogram4_unconditional-Q8_0.gguf", "unconditional"),
        # rectangleworm
        ("ideogram4_Q5_K.gguf", "conditional"),
        ("ideogram4_unconditional_Q5_K.gguf", "unconditional"),
    ],
)
def test_a_published_gguf_installs_as_the_branch_its_name_says(tmp_path: Path, filename: str, branch: str) -> None:
    path = tmp_path / filename
    write_ideogram4_gguf(path, packs_block_linears)

    result = _identify(path)

    assert result.match_count == 1, [type(match).__name__ for match in result.all_matches]
    assert isinstance(result.config, Main_GGUF_Ideogram4_Config)
    assert result.config.branch == branch


def test_a_safetensors_single_file_is_not_claimed_as_gguf(tmp_path: Path) -> None:
    """The other half of the exclusion: the GGUF config must not take Comfy-Org's files."""
    reference = write_ideogram4_gguf(tmp_path / "unused.gguf", packs_block_linears)
    path = tmp_path / "ideogram4_fp8_scaled.safetensors"
    save_file({key: value.to(torch.bfloat16) for key, value in reference.items()}, path)

    result = _identify(path)

    assert result.match_count == 1, [type(match).__name__ for match in result.all_matches]
    assert isinstance(result.config, Main_Checkpoint_Ideogram4_Config)
