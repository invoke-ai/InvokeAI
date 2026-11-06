"""Identification of Anima GGUF transformers: which config claims them, and which are refused with a reason.

The GGUF config compares every tensor's shape with the full-size transformer the loader builds, so these files
carry the real key layouts at their real extents (the Anima-2.9B and Anima-3.8B fixtures), as stripped GGUF state
dicts -- every tensor a `GGMLTensor` holding only its shape, the way `tests/model_identification` stores the
published files. Those published files, the accepted layouts, are pinned by their stripped headers there.
"""

import re
from pathlib import Path

import gguf
import pytest
import torch
from safetensors.torch import save_file

from invokeai.backend.model_manager.configs.factory import ModelClassificationResult, ModelConfigFactory
from invokeai.backend.model_manager.configs.identification_utils import InvalidMatchError, NotAMatchError
from invokeai.backend.model_manager.configs.main import Main_Checkpoint_Anima_Config, Main_GGUF_Anima_Config
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import AnimaVariantType
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor
from tests.backend.model_manager.load.state_dicts.anima_2_9b_keys import ANIMA_2_9B_NUM_BLOCKS
from tests.backend.model_manager.load.state_dicts.anima_2_9b_keys import state_dict_keys as anima_2_9b_keys
from tests.backend.model_manager.load.state_dicts.anima_3_8b_connector_keys import (
    ANIMA_3_8B_NUM_BLOCKS,
    connector_keys,
)
from tests.backend.model_manager.load.state_dicts.anima_3_8b_connector_keys import metadata as anima_3_8b_metadata

_BLOCK_KEY = re.compile(r"^net\.blocks\.(\d+)\.(.+)$")
_PATCH_EMBED = "net.x_embedder.proj.1.weight"


def _layout(num_blocks: int, *, connector: bool = False) -> dict[str, list[int]]:
    """The Anima-2.9B export's keys and shapes at `num_blocks` blocks, its `pos_embedder` buffers included."""
    block = {
        m.group(2): shape for key, shape in anima_2_9b_keys.items() if (m := _BLOCK_KEY.match(key)) and m[1] == "0"
    }
    shapes = {key: shape for key, shape in anima_2_9b_keys.items() if not _BLOCK_KEY.match(key)}
    shapes.update({f"net.blocks.{i}.{suffix}": shape for i in range(num_blocks) for suffix, shape in block.items()})
    if connector:
        shapes.update(connector_keys)
    return shapes


def _as_gguf(shapes: dict[str, list[int]], *, doubled: bool = False) -> dict[str, GGMLTensor]:
    """Store it as the converters do: 2-D weights Q8_0, the rest and the 68-wide patch embedding unquantized.

    `doubled` writes every quantized tensor with twice its columns, as vanes430's Anima-2.9B conversion stores
    them (read from its Q4_0 header: all 707 quantized tensors, and none of the unquantized ones).
    """

    def stored(key: str, shape: list[int]) -> GGMLTensor:
        quantized = len(shape) == 2 and key != _PATCH_EMBED
        if quantized and doubled:
            shape = [*shape[:-1], 2 * shape[-1]]
        qtype = gguf.GGMLQuantizationType.Q8_0 if quantized else gguf.GGMLQuantizationType.BF16
        return GGMLTensor(torch.empty(0, dtype=torch.uint8, device="meta"), qtype, torch.Size(shape), torch.float32)

    return {key: stored(key, shape) for key, shape in shapes.items()}


class _StrippedGGUF(ModelOnDisk):
    """A GGUF as identification reads it, its state dict and string metadata, held in memory.

    `StrippedModelOnDisk` would serve the same from a JSON file, but parses it again for every candidate config:
    three seconds per test at these extents.
    """

    def __init__(self, path: Path, sd: dict[str, torch.Tensor], metadata: dict[str, str]):
        super().__init__(path)
        self._sd = sd
        self._header = metadata

    def load_state_dict(self, path: Path | None = None) -> dict[str | int, torch.Tensor]:
        return dict(self._sd)

    def metadata(self, path: Path | None = None) -> dict[str, str]:
        return self._header


def _identify(tmp_path: Path, sd: dict[str, GGMLTensor], metadata: dict[str, str] | None = None):
    path = tmp_path / "anima.gguf"
    path.touch()
    # `allow_unknown` as the installer defaults to: a refusal must not fall back to an unknown model.
    return ModelConfigFactory.from_model_on_disk(_StrippedGGUF(path, sd, metadata or {}), allow_unknown=True)


def _refusal(result: ModelClassificationResult) -> str:
    assert result.config is None, f"installed as {type(result.config).__name__}"
    refusal = result.details[Main_GGUF_Anima_Config.__name__]
    assert isinstance(refusal, InvalidMatchError), refusal
    return str(refusal)


def test_the_2_9b_depth_installs_as_a_gguf_main_model(tmp_path) -> None:
    result = _identify(tmp_path, _as_gguf(_layout(ANIMA_2_9B_NUM_BLOCKS)))

    assert result.match_count == 1, [type(match).__name__ for match in result.all_matches]
    assert isinstance(result.config, Main_GGUF_Anima_Config)
    assert result.config.variant is AnimaVariantType.Qwen3
    # Before the GGUF config existed, the checkpoint config claimed these files and the load crashed.
    assert isinstance(result.details[Main_Checkpoint_Anima_Config.__name__], NotAMatchError)


def test_a_safetensors_checkpoint_is_not_claimed_as_gguf(tmp_path) -> None:
    path = tmp_path / "anima.safetensors"
    save_file({key: torch.zeros(1) for key in _layout(1)}, path)

    result = ModelConfigFactory.from_model_on_disk(path, allow_unknown=False)

    assert isinstance(result.config, Main_Checkpoint_Anima_Config)
    assert isinstance(result.details[Main_GGUF_Anima_Config.__name__], NotAMatchError)


def test_the_byte_quantized_conversion_is_refused_with_its_cause(tmp_path) -> None:
    """Its patch embedding is right and every quantized tensor twice as wide: the bf16 bytes were quantized as
    8-bit numbers. Installed, it would fail at the first render or draw noise."""
    reason = _refusal(_identify(tmp_path, _as_gguf(_layout(ANIMA_2_9B_NUM_BLOCKS), doubled=True)))

    assert "twice the columns" in reason
    assert "`t_embedder.1.linear_1.weight` is (2048, 4096) where the model's is (2048, 2048)" in reason
    assert "safetensors release" in reason


@pytest.mark.parametrize(
    "mismatched",
    [
        {"net.blocks.1.mlp.layer1.weight": [8192, 3072]},
        # One tensor with the byte-quantization signature does not make it the cause of the other.
        {"net.blocks.0.mlp.layer1.weight": [8192, 4096], "net.blocks.1.mlp.layer1.weight": [8192, 3072]},
    ],
    ids=["alone", "beside-a-doubled-one"],
)
def test_a_mismatch_of_another_kind_names_no_cause(tmp_path, mismatched) -> None:
    """The byte-quantization cause is claimed only where every mismatch has its signature."""
    sd = {**_as_gguf(_layout(2)), **_as_gguf(mismatched)}

    reason = _refusal(_identify(tmp_path, sd))

    assert f"{len(mismatched)} of its tensors" in reason
    assert "twice the columns" not in reason


def test_a_transformer_of_another_width_is_refused(tmp_path) -> None:
    sd = _as_gguf(_layout(2))
    sd[_PATCH_EMBED] = _as_gguf({_PATCH_EMBED: [3072, 68]})[_PATCH_EMBED]

    reason = _refusal(_identify(tmp_path, sd))

    assert "a size this version of Invoke cannot build" in reason


def test_an_incomplete_transformer_is_refused(tmp_path) -> None:
    sd = _as_gguf(_layout(2))
    del sd["net.t_embedding_norm.weight"]

    reason = _refusal(_identify(tmp_path, sd))

    assert "lacks 1 tensor(s)" in reason and "t_embedding_norm.weight" in reason


def test_a_52_block_gguf_without_the_connector_is_refused(tmp_path) -> None:
    """The Anima-3.8B v1.0 transformer is 52 blocks without a connector; the checkpoint config tells it apart by
    a header key no GGUF converter carries over, so a GGUF of that shape cannot be installed safely."""
    reason = _refusal(_identify(tmp_path, _as_gguf(_layout(ANIMA_3_8B_NUM_BLOCKS))))

    assert "Anima-3.8B v1.0" in reason and "Anima-3.8B-v1.1.safetensors" in reason


def test_anima_3_8b_with_its_connector_is_refused(tmp_path) -> None:
    """No GGUF of Anima-3.8B exists to verify the connector against, so none is installed."""
    sd = _as_gguf(_layout(ANIMA_3_8B_NUM_BLOCKS, connector=True))

    reason = _refusal(_identify(tmp_path, sd, anima_3_8b_metadata))

    assert "No GGUF of Anima-3.8B has been verified" in reason and "Anima-3.8B (v1.1) starter model" in reason
