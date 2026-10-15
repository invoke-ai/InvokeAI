"""Where the depth-expanded Anima finetunes keep the blocks of the model they grew from."""

import pytest

from invokeai.backend.anima import block_layout
from invokeai.backend.anima.block_layout import adapter_block_positions

BASE_TO_2_9B = block_layout._POSITIONS[(28, 40)]
FROM_2_9B_TO_3_8B = block_layout._POSITIONS[(40, 52)]
BASE_TO_3_8B = block_layout._POSITIONS[(28, 52)]


@pytest.mark.parametrize(("source", "target"), [(28, 40), (40, 52), (28, 52)])
def test_every_source_block_has_its_own_target_block_in_order(source: int, target: int) -> None:
    positions = block_layout._POSITIONS[(source, target)]
    assert len(positions) == source
    assert list(positions) == sorted(set(positions))
    assert positions[0] == 0 and positions[-1] < target


def test_the_new_blocks_sit_where_the_checkpoints_have_them() -> None:
    """Measured: in 2.9B, the blocks that are not bit-identical to an Anima base block; in 3.8B, the blocks at a cosine
    similarity of about 0.6 to every 2.9B block."""
    assert sorted(set(range(40)) - set(BASE_TO_2_9B)) == [2, 5, 8, 11, 14, 17, 21, 24, 27, 30, 33, 36]
    assert sorted(set(range(52)) - set(FROM_2_9B_TO_3_8B)) == list(range(3, 48, 4))


def test_the_base_blocks_still_bit_identical_in_3_8b_are_where_the_composed_table_puts_them() -> None:
    """Ten blocks of Anima base-v1.0 are bit-identical in Anima-3.8B v1.1 -- an independent check of the composition."""
    measured = {5: 9, 7: 13, 10: 20, 14: 26, 15: 29, 19: 37, 21: 41, 22: 42, 23: 45, 27: 51}
    assert {base: BASE_TO_3_8B[base] for base in measured} == measured


@pytest.mark.parametrize(
    ("max_block_index", "target_depth", "expected"),
    [
        (27, 40, BASE_TO_2_9B),
        (27, 52, BASE_TO_3_8B),
        (39, 52, FROM_2_9B_TO_3_8B),
        # A base LoRA that leaves out the last blocks still comes from a 28-block model.
        (20, 40, BASE_TO_2_9B),
    ],
    ids=["base-on-2.9B", "base-on-3.8B", "2.9B-on-3.8B", "partial-base-on-2.9B"],
)
def test_a_shallower_adapter_moves_to_the_blocks_it_was_trained_on(max_block_index, target_depth, expected) -> None:
    assert adapter_block_positions(max_block_index, target_depth) == expected


@pytest.mark.parametrize(
    ("max_block_index", "target_depth"),
    [(27, 28), (39, 40), (51, 52), (39, 28), (51, 40), (60, 52), (27, 30)],
    ids=[
        "base-on-base",
        "2.9B-on-2.9B",
        "3.8B-on-3.8B",
        "deeper-adapter",
        "3.8B-on-2.9B",
        "unknown-depth",
        "no-layout",
    ],
)
def test_everything_else_addresses_blocks_by_index(max_block_index, target_depth) -> None:
    assert adapter_block_positions(max_block_index, target_depth) is None
