"""Key layout of SimpleTuner's LyCORIS LoKR over FLUX.2 Klein 9B
(`gattaplayer/besch-flux2-klein-9b-lokr-lion-5e-6-bs2-ga2-v01`).

Captured on 2026-11-11 with a header-only read of the repo's `pytorch_lora_weights.safetensors`,
the issue #10174 FLUX.2 Klein row. Subsetting rule: single block 0 and double block 0 in full --
every module kind of the file's presets (Flux2Attention, Flux2FeedForward, Flux2ParallelSelfAttention).
Values are `(shape, dtype)`.

Flux2Transformer2DModel's own module names flattened behind `lycoris_`: the fused
`single_transformer_blocks.N.attn.to_qkv_mlp_proj` and `attn.to_out`, and the double blocks'
`ff.linear_in`/`ff.linear_out` (not FLUX.1's `ff.net.0.proj`). 432 keys in the file, 144 layers."""

state_dict_keys: dict[str, tuple[list[int], str]] = {
    "lycoris_single_transformer_blocks_0_attn_to_out.alpha": ([], "BF16"),
    "lycoris_single_transformer_blocks_0_attn_to_out.lokr_w1": ([16, 16], "BF16"),
    "lycoris_single_transformer_blocks_0_attn_to_out.lokr_w2": ([256, 1024], "BF16"),
    "lycoris_single_transformer_blocks_0_attn_to_qkv_mlp_proj.alpha": ([], "BF16"),
    "lycoris_single_transformer_blocks_0_attn_to_qkv_mlp_proj.lokr_w1": ([16, 16], "BF16"),
    "lycoris_single_transformer_blocks_0_attn_to_qkv_mlp_proj.lokr_w2": ([2304, 256], "BF16"),
    "lycoris_transformer_blocks_0_attn_add_k_proj.alpha": ([], "BF16"),
    "lycoris_transformer_blocks_0_attn_add_k_proj.lokr_w1": ([16, 16], "BF16"),
    "lycoris_transformer_blocks_0_attn_add_k_proj.lokr_w2": ([256, 256], "BF16"),
    "lycoris_transformer_blocks_0_attn_add_q_proj.alpha": ([], "BF16"),
    "lycoris_transformer_blocks_0_attn_add_q_proj.lokr_w1": ([16, 16], "BF16"),
    "lycoris_transformer_blocks_0_attn_add_q_proj.lokr_w2": ([256, 256], "BF16"),
    "lycoris_transformer_blocks_0_attn_add_v_proj.alpha": ([], "BF16"),
    "lycoris_transformer_blocks_0_attn_add_v_proj.lokr_w1": ([16, 16], "BF16"),
    "lycoris_transformer_blocks_0_attn_add_v_proj.lokr_w2": ([256, 256], "BF16"),
    "lycoris_transformer_blocks_0_attn_to_add_out.alpha": ([], "BF16"),
    "lycoris_transformer_blocks_0_attn_to_add_out.lokr_w1": ([16, 16], "BF16"),
    "lycoris_transformer_blocks_0_attn_to_add_out.lokr_w2": ([256, 256], "BF16"),
    "lycoris_transformer_blocks_0_attn_to_k.alpha": ([], "BF16"),
    "lycoris_transformer_blocks_0_attn_to_k.lokr_w1": ([16, 16], "BF16"),
    "lycoris_transformer_blocks_0_attn_to_k.lokr_w2": ([256, 256], "BF16"),
    "lycoris_transformer_blocks_0_attn_to_out_0.alpha": ([], "BF16"),
    "lycoris_transformer_blocks_0_attn_to_out_0.lokr_w1": ([16, 16], "BF16"),
    "lycoris_transformer_blocks_0_attn_to_out_0.lokr_w2": ([256, 256], "BF16"),
    "lycoris_transformer_blocks_0_attn_to_q.alpha": ([], "BF16"),
    "lycoris_transformer_blocks_0_attn_to_q.lokr_w1": ([16, 16], "BF16"),
    "lycoris_transformer_blocks_0_attn_to_q.lokr_w2": ([256, 256], "BF16"),
    "lycoris_transformer_blocks_0_attn_to_v.alpha": ([], "BF16"),
    "lycoris_transformer_blocks_0_attn_to_v.lokr_w1": ([16, 16], "BF16"),
    "lycoris_transformer_blocks_0_attn_to_v.lokr_w2": ([256, 256], "BF16"),
    "lycoris_transformer_blocks_0_ff_context_linear_in.alpha": ([], "BF16"),
    "lycoris_transformer_blocks_0_ff_context_linear_in.lokr_w1": ([16, 16], "BF16"),
    "lycoris_transformer_blocks_0_ff_context_linear_in.lokr_w2": ([1536, 256], "BF16"),
    "lycoris_transformer_blocks_0_ff_context_linear_out.alpha": ([], "BF16"),
    "lycoris_transformer_blocks_0_ff_context_linear_out.lokr_w1": ([16, 16], "BF16"),
    "lycoris_transformer_blocks_0_ff_context_linear_out.lokr_w2": ([256, 768], "BF16"),
    "lycoris_transformer_blocks_0_ff_linear_in.alpha": ([], "BF16"),
    "lycoris_transformer_blocks_0_ff_linear_in.lokr_w1": ([16, 16], "BF16"),
    "lycoris_transformer_blocks_0_ff_linear_in.lokr_w2": ([1536, 256], "BF16"),
    "lycoris_transformer_blocks_0_ff_linear_out.alpha": ([], "BF16"),
    "lycoris_transformer_blocks_0_ff_linear_out.lokr_w1": ([16, 16], "BF16"),
    "lycoris_transformer_blocks_0_ff_linear_out.lokr_w2": ([256, 768], "BF16"),
}
