"""Key layout of SimpleTuner's LyCORIS LoKR over Wan (`uxoah/wan2.1_lokr_simpletuner`).

Captured on 2026-11-11 with a header-only read of `erkcga_lokr_simpletuner.safetensors` of the
repo, the issue #10174 Wan 2.1 row. Subsetting rule: block 0 in full -- the Attention preset's
eight projections, every module kind of this file. Values are `(shape, dtype)`.

The Wan transformer's own module names flattened behind `lycoris_`
(`lycoris_blocks_0_attn1_to_q`, `lycoris_blocks_0_attn2_to_out_0`) -- the same tails kohya's
`lora_unet_blocks_` naming uses. 960 keys in the file, 320 layers (40 of 40 blocks)."""

state_dict_keys: dict[str, tuple[list[int], str]] = {
    "lycoris_blocks_0_attn1_to_k.alpha": ([], "BF16"),
    "lycoris_blocks_0_attn1_to_k.lokr_w1": ([16, 16], "BF16"),
    "lycoris_blocks_0_attn1_to_k.lokr_w2": ([320, 320], "BF16"),
    "lycoris_blocks_0_attn1_to_out_0.alpha": ([], "BF16"),
    "lycoris_blocks_0_attn1_to_out_0.lokr_w1": ([16, 16], "BF16"),
    "lycoris_blocks_0_attn1_to_out_0.lokr_w2": ([320, 320], "BF16"),
    "lycoris_blocks_0_attn1_to_q.alpha": ([], "BF16"),
    "lycoris_blocks_0_attn1_to_q.lokr_w1": ([16, 16], "BF16"),
    "lycoris_blocks_0_attn1_to_q.lokr_w2": ([320, 320], "BF16"),
    "lycoris_blocks_0_attn1_to_v.alpha": ([], "BF16"),
    "lycoris_blocks_0_attn1_to_v.lokr_w1": ([16, 16], "BF16"),
    "lycoris_blocks_0_attn1_to_v.lokr_w2": ([320, 320], "BF16"),
    "lycoris_blocks_0_attn2_to_k.alpha": ([], "BF16"),
    "lycoris_blocks_0_attn2_to_k.lokr_w1": ([16, 16], "BF16"),
    "lycoris_blocks_0_attn2_to_k.lokr_w2": ([320, 320], "BF16"),
    "lycoris_blocks_0_attn2_to_out_0.alpha": ([], "BF16"),
    "lycoris_blocks_0_attn2_to_out_0.lokr_w1": ([16, 16], "BF16"),
    "lycoris_blocks_0_attn2_to_out_0.lokr_w2": ([320, 320], "BF16"),
    "lycoris_blocks_0_attn2_to_q.alpha": ([], "BF16"),
    "lycoris_blocks_0_attn2_to_q.lokr_w1": ([16, 16], "BF16"),
    "lycoris_blocks_0_attn2_to_q.lokr_w2": ([320, 320], "BF16"),
    "lycoris_blocks_0_attn2_to_v.alpha": ([], "BF16"),
    "lycoris_blocks_0_attn2_to_v.lokr_w1": ([16, 16], "BF16"),
    "lycoris_blocks_0_attn2_to_v.lokr_w2": ([320, 320], "BF16"),
}
