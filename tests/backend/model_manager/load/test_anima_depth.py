"""The Anima loader builds the transformer at the depth of the checkpoint it loads.

Anima-2.9B (40 blocks) and Anima-3.8B (52 blocks) are depth-expanded finetunes of the official
28-block release. A transformer built at the official depth loads the first 28 blocks of such a
checkpoint and drops the rest as unexpected keys under `strict=False` -- logged at DEBUG only, so the
failure is a degraded image, not an error. These tests pin the depth detection to the real key
layouts and check that a model built from it takes every key.
"""

import re

import accelerate
import pytest
import torch

from invokeai.backend.anima.anima_transformer import AnimaTransformer
from invokeai.backend.model_manager.load.model_loaders.anima import (
    ANIMA_TRANSFORMER_CONFIG,
    _filter_non_model_keys,
    _strip_anima_bundle_prefix,
    anima_transformer_config,
    count_anima_dit_blocks,
)
from tests.backend.model_manager.load.state_dicts.anima_2_9b_keys import ANIMA_2_9B_NUM_BLOCKS
from tests.backend.model_manager.load.state_dicts.anima_2_9b_keys import state_dict_keys as anima_2_9b_keys
from tests.backend.model_manager.load.state_dicts.anima_comfyui_keys import state_dict_keys as anima_keys

_BLOCK_KEY = re.compile(r"^net\.blocks\.(\d+)\.(.+)$")


def _full_depth_meta_state_dict(fixture: dict[str, list[int]], num_blocks: int) -> dict[str, torch.Tensor]:
    """Expand a fixture's block-0 keys to `num_blocks` blocks, as meta tensors.

    Meta, because the real extents of 40 blocks are billions of elements and nothing here reads a
    value.
    """
    block_suffixes = {
        m.group(2): shape for key, shape in fixture.items() if (m := _BLOCK_KEY.match(key)) and m.group(1) == "0"
    }
    sd = {key: torch.empty(shape, device="meta") for key, shape in fixture.items() if not _BLOCK_KEY.match(key)}
    for index in range(num_blocks):
        for suffix, shape in block_suffixes.items():
            sd[f"net.blocks.{index}.{suffix}"] = torch.empty(shape, device="meta")
    return sd


def _prepare(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The two key passes the loader runs before it builds the model."""
    return _filter_non_model_keys(_strip_anima_bundle_prefix(sd))


class TestCountAnimaDitBlocks:
    def test_official_release_has_28_blocks(self) -> None:
        sd = _prepare(_full_depth_meta_state_dict(anima_keys, 28))
        assert count_anima_dit_blocks(sd) == 28 == ANIMA_TRANSFORMER_CONFIG["num_blocks"]

    def test_anima_2_9b_has_40_blocks(self) -> None:
        sd = _prepare(_full_depth_meta_state_dict(anima_2_9b_keys, ANIMA_2_9B_NUM_BLOCKS))
        assert count_anima_dit_blocks(sd) == 40

    def test_llm_adapter_blocks_are_not_counted(self) -> None:
        # `llm_adapter.blocks.<n>` shares the `blocks.` segment; only a key that starts with it counts.
        sd = {"blocks.0.mlp.layer1.weight": torch.empty(0), "llm_adapter.blocks.5.mlp.0.weight": torch.empty(0)}
        assert count_anima_dit_blocks(sd) == 1

    def test_gap_in_block_indices_is_refused(self) -> None:
        sd = {f"blocks.{i}.mlp.layer1.weight": torch.empty(0) for i in (0, 1, 3)}
        with pytest.raises(ValueError, match=r"gaps .*\[2\]"):
            count_anima_dit_blocks(sd)

    def test_state_dict_without_blocks_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no DiT blocks"):
            count_anima_dit_blocks({"final_layer.linear.weight": torch.empty(0)})


class TestRealAnima29BLayout:
    """What the captured 2.9B header says, beyond the block count."""

    def test_last_block_has_the_same_layout_as_the_first(self) -> None:
        first = {m.group(2): s for k, s in anima_2_9b_keys.items() if (m := _BLOCK_KEY.match(k)) and m.group(1) == "0"}
        last = {
            m.group(2): s
            for k, s in anima_2_9b_keys.items()
            if (m := _BLOCK_KEY.match(k)) and m.group(1) == str(ANIMA_2_9B_NUM_BLOCKS - 1)
        }
        assert first and first == last

    def test_every_non_block_tensor_matches_the_official_release(self) -> None:
        # Only the depth changed: the LLM adapter, embedders and final layer are the official shapes.
        official = {k: s for k, s in anima_keys.items() if not _BLOCK_KEY.match(k)}
        expanded = {k: s for k, s in anima_2_9b_keys.items() if not _BLOCK_KEY.match(k)}
        assert {k: s for k, s in expanded.items() if not k.startswith("net.pos_embedder.")} == official


@pytest.mark.parametrize(
    ("fixture", "num_blocks"),
    [(anima_keys, 28), (anima_2_9b_keys, ANIMA_2_9B_NUM_BLOCKS)],
    ids=["official-28", "anima-2.9b-40"],
)
def test_model_built_at_detected_depth_takes_every_key(fixture: dict[str, list[int]], num_blocks: int) -> None:
    sd = _prepare(_full_depth_meta_state_dict(fixture, num_blocks))

    with accelerate.init_empty_weights():
        model = AnimaTransformer(**anima_transformer_config(sd))
    result = model.load_state_dict(sd, strict=False, assign=True)

    assert len(model.blocks) == num_blocks
    assert result.unexpected_keys == []
    assert result.missing_keys == []
