"""FLUX.3 Image Klein Base (undistilled), BFL's native bf16 layout -- `model.safetensors` of `black-forest-labs/flux-3-image-klein-base`.

Captured on 2026-10-16 with `scripts/capture_state_dict_fixture.py`,
which reads the header only, from
`black-forest-labs/flux-3-image-klein-base/model.safetensors`.

Subsetting rule: `content_mode_blocks.image` index 0, `content_mode_blocks.image_cond` index 0, `single_blocks` index 0, `txt_mode_blocks` index 0, plus every key outside a stack. Values are `(shape, dtype)`.

The same layout as Klein without `ref_time_in.*`: that embedder is what tells a flow-map checkpoint from the Base model, and
what the variant detection and the transformer config read.
"""

state_dict_keys: dict[str, tuple[list[int], str]] = {
    "content_mode_blocks.image.0.attn_out.weight": ([3072, 3072], "BF16"),
    "content_mode_blocks.image.0.k_proj.weight": ([3072, 3072], "BF16"),
    "content_mode_blocks.image.0.mlp_in.weight": ([18432, 3072], "BF16"),
    "content_mode_blocks.image.0.mlp_out.weight": ([3072, 9216], "BF16"),
    "content_mode_blocks.image.0.norm.key_norm.scale": ([128], "BF16"),
    "content_mode_blocks.image.0.norm.query_norm.scale": ([128], "BF16"),
    "content_mode_blocks.image.0.q_proj.weight": ([3072, 3072], "BF16"),
    "content_mode_blocks.image.0.v_proj.weight": ([3072, 3072], "BF16"),
    "content_mode_blocks.image_cond.0.attn_out.weight": ([3072, 3072], "BF16"),
    "content_mode_blocks.image_cond.0.k_proj.weight": ([3072, 3072], "BF16"),
    "content_mode_blocks.image_cond.0.mlp_in.weight": ([18432, 3072], "BF16"),
    "content_mode_blocks.image_cond.0.mlp_out.weight": ([3072, 9216], "BF16"),
    "content_mode_blocks.image_cond.0.norm.key_norm.scale": ([128], "BF16"),
    "content_mode_blocks.image_cond.0.norm.query_norm.scale": ([128], "BF16"),
    "content_mode_blocks.image_cond.0.q_proj.weight": ([3072, 3072], "BF16"),
    "content_mode_blocks.image_cond.0.v_proj.weight": ([3072, 3072], "BF16"),
    "early_stream_modulations.image.lin.weight": ([9216, 3072], "BF16"),
    "early_stream_modulations.image_cond.lin.weight": ([9216, 3072], "BF16"),
    "early_stream_modulations.txt.lin.weight": ([9216, 3072], "BF16"),
    "emb_in.image.weight": ([3072, 128], "BF16"),
    "emb_in.image_cond.weight": ([3072, 128], "BF16"),
    "final_layer.image.adaLN_modulation.1.weight": ([6144, 3072], "BF16"),
    "final_layer.image.linear.weight": ([128, 3072], "BF16"),
    "single_blocks.0.attn_out.weight": ([3072, 3072], "BF16"),
    "single_blocks.0.k_proj.weight": ([3072, 3072], "BF16"),
    "single_blocks.0.mlp_in.weight": ([18432, 3072], "BF16"),
    "single_blocks.0.mlp_out.weight": ([3072, 9216], "BF16"),
    "single_blocks.0.norm.key_norm.scale": ([128], "BF16"),
    "single_blocks.0.norm.query_norm.scale": ([128], "BF16"),
    "single_blocks.0.q_proj.weight": ([3072, 3072], "BF16"),
    "single_blocks.0.v_proj.weight": ([3072, 3072], "BF16"),
    "single_stream_modulations.image.lin.weight": ([9216, 3072], "BF16"),
    "single_stream_modulations.image_cond.lin.weight": ([9216, 3072], "BF16"),
    "single_stream_modulations.txt.lin.weight": ([9216, 3072], "BF16"),
    "time_in.in_layer.weight": ([3072, 256], "BF16"),
    "time_in.out_layer.weight": ([3072, 3072], "BF16"),
    "txt_in.weight": ([3072, 20480], "BF16"),
    "txt_mode_blocks.0.attn_out.weight": ([3072, 3072], "BF16"),
    "txt_mode_blocks.0.k_proj.weight": ([3072, 3072], "BF16"),
    "txt_mode_blocks.0.mlp_in.weight": ([18432, 3072], "BF16"),
    "txt_mode_blocks.0.mlp_out.weight": ([3072, 9216], "BF16"),
    "txt_mode_blocks.0.norm.key_norm.scale": ([128], "BF16"),
    "txt_mode_blocks.0.norm.query_norm.scale": ([128], "BF16"),
    "txt_mode_blocks.0.q_proj.weight": ([3072, 3072], "BF16"),
    "txt_mode_blocks.0.v_proj.weight": ([3072, 3072], "BF16"),
}
