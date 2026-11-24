"""Key layout of an SD 1.5 UNet's attentions under SimpleTuner's LyCORIS flat spelling.

Not a captured file: diffusers' own module names for `UNet2DConditionModel` at the released SD 1.5
config (`block_out_channels` 320/640/1280/1280, `layers_per_block` 2, `cross_attention_dim` 768,
`attention_head_dim` 8, `use_linear_projection`), flattened behind `lycoris_`. SD 1.x and SD 2.x put
a single transformer block inside every attention, so their flattened names land in the very
templates SDXL shares with them
(`lycoris_down_blocks_2_attentions_0_transformer_blocks_0_attn1_to_q`). The transformer-block index
is all that separates the families, so no config may claim a file of these names.

`state_dict_keys` holds the eight attention projections of every attention block (128 layers);
`attention_and_ff_state_dict_keys` adds the feed-forward pair (160 layers). Values are
`(shape, dtype)`; the widths only set the LoKR factor shapes, since detection reads names.
"""

# (flattened prefix, attention block indices, block width, its feed-forward width), in the UNet's own
# order: `up_blocks.0` carries no attention, and `mid_block` holds a single one.
_ATTENTION_BLOCKS: tuple[tuple[str, tuple[int, ...], int, int], ...] = (
    ("down_blocks_0_attentions", (0, 1), 320, 2560),
    ("down_blocks_1_attentions", (0, 1), 640, 5120),
    ("down_blocks_2_attentions", (0, 1), 1280, 10240),
    ("up_blocks_1_attentions", (0, 1, 2), 1280, 10240),
    ("up_blocks_2_attentions", (0, 1, 2), 640, 5120),
    ("up_blocks_3_attentions", (0, 1, 2), 320, 2560),
    ("mid_block_attentions", (0,), 1280, 10240),
)


def build_state_dict_keys(cross_attention_dim: int, *, include_feed_forward: bool) -> dict[str, tuple[list[int], str]]:
    """The flat key layout of an SD 1.x/SD 2.x UNet's attentions at a given text-encoding width."""
    keys: dict[str, tuple[list[int], str]] = {}

    def layer(name: str, in_dim: int, out_dim: int) -> None:
        factor = in_dim // 16
        keys[f"lycoris_{name}.alpha"] = ([], "BF16")
        keys[f"lycoris_{name}.lokr_w1"] = ([16, 16], "BF16")
        keys[f"lycoris_{name}.lokr_w2"] = ([out_dim // 16, factor], "BF16")

    for prefix, indices, width, ff_width in _ATTENTION_BLOCKS:
        for index in indices:
            block = f"{prefix}_{index}_transformer_blocks_0"
            for tail in ("attn1_to_q", "attn1_to_k", "attn1_to_v", "attn1_to_out_0", "attn2_to_q"):
                layer(f"{block}_{tail}", width, width)
            # The cross-attention projections read the text encoding's width, not the block's.
            for tail in ("attn2_to_k", "attn2_to_v"):
                layer(f"{block}_{tail}", cross_attention_dim, width)
            layer(f"{block}_attn2_to_out_0", width, width)
            if include_feed_forward:
                layer(f"{block}_ff_net_0_proj", width, ff_width)
                layer(f"{block}_ff_net_2", ff_width, width)
    return keys


state_dict_keys = build_state_dict_keys(768, include_feed_forward=False)
attention_and_ff_state_dict_keys = build_state_dict_keys(768, include_feed_forward=True)
