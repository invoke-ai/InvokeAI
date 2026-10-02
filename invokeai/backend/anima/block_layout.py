"""Where the depth-expanded Anima finetunes keep the blocks of the model they grew from.

Anima-2.9B and Anima-3.8B insert new DiT blocks between the existing ones (LLaMA-Pro style interleaved expansion),
so an adapter that addresses blocks by index -- a LoRA, a ControlNet-LLLite -- trained on the shallower model would,
from the third block on, patch different blocks than the ones it was trained on. These tables send it to the right
ones. Measured from the released checkpoints:

- Anima-2.9B-preview-v1 (40 blocks) holds all 28 blocks of Anima base-v1.0 bit for bit: its card says only the new
  layers were trained. The 12 new blocks sit at 2, 5, 8, 11, 14, 17, 21, 24, 27, 30, 33 and 36.
- Anima-3.8B v1.1 (52 blocks) holds the 40 blocks of Anima-2.9B nearly unchanged: each has a mean cosine similarity of
  at least 0.9997 to its 2.9B block, and 11 are bit-identical. The 12 new blocks, at 3, 7, 11, ..., 47, are at about
  0.6 to the neighbor they were copied from. Ten blocks of Anima base-v1.0 are still bit-identical in it, all at the
  positions below.

Keyed by depth: these are the only depth-expanded Anima models. A future finetune with one of these depths but
another layout would need its own entry.
"""

from typing import Optional

ANIMA_BASE_DEPTH = 28

# (source depth, target depth) -> the target position of each source block.
_POSITIONS: dict[tuple[int, int], tuple[int, ...]] = {
    (28, 40): (0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 20, 22, 23, 25, 26, 28, 29, 31, 32, 34, 35, 37, 38, 39),
    (40, 52): (
        0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25,
        26, 28, 29, 30, 32, 33, 34, 36, 37, 38, 40, 41, 42, 44, 45, 46, 48, 49, 50, 51,
    ),
}  # fmt: skip
_POSITIONS[(28, 52)] = tuple(_POSITIONS[(40, 52)][i] for i in _POSITIONS[(28, 40)])

KNOWN_DEPTHS = (28, 40, 52)


def adapter_block_positions(max_block_index: int, target_depth: int) -> Optional[tuple[int, ...]]:
    """Where, in a `target_depth`-block model, the blocks of the model an adapter was trained on sit.

    The adapter's own depth is read off the highest block it addresses: the smallest known depth that has that block.
    An adapter trained on Anima base touches block 27; one trained on Anima-2.9B, a block past 27. Returns None when
    the adapter addresses blocks by index as before -- same depth, a deeper adapter than the model, or no layout for
    the pair.
    """
    source_depth = next((depth for depth in KNOWN_DEPTHS if depth > max_block_index), None)
    if source_depth is None or source_depth >= target_depth:
        return None
    return _POSITIONS.get((source_depth, target_depth))
