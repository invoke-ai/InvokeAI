"""Key layout of SimpleTuner's LyCORIS LoKR over FLUX.1 (`toxicwind/glassobject-lokr-flux`).

Captured on 2026-11-11 with a header-only read (HTTP range request) of
`toxicwind/glassobject-lokr-flux/pytorch_lora_weights.safetensors`, the issue #10174 repro file.
Subsetting rule: single block 0's attention and double block 0 in full -- one block of each kind,
every module kind the file trains (Attention + FeedForward presets). Values are `(shape, dtype)`.

diffusers' `FluxTransformer2DModel` module names flattened behind `lycoris_` with every `.` turned
into `_` (`lycoris_single_transformer_blocks_0_attn_to_k`). Module names contain underscores of
their own (`to_k`, `add_k_proj`), so a flattened name is only resolvable against the whole module
list. LoKR with `alpha`, a full `lokr_w1` and a full `lokr_w2`; the patch weight's shape is the
Kronecker product. 1026 keys in the file, 342 layers."""

state_dict_keys: dict[str, tuple[list[int], str]] = {
    "lycoris_single_transformer_blocks_0_attn_to_k.alpha": ([], "BF16"),
    "lycoris_single_transformer_blocks_0_attn_to_k.lokr_w1": ([16, 16], "BF16"),
    "lycoris_single_transformer_blocks_0_attn_to_k.lokr_w2": ([192, 192], "BF16"),
    "lycoris_single_transformer_blocks_0_attn_to_q.alpha": ([], "BF16"),
    "lycoris_single_transformer_blocks_0_attn_to_q.lokr_w1": ([16, 16], "BF16"),
    "lycoris_single_transformer_blocks_0_attn_to_q.lokr_w2": ([192, 192], "BF16"),
    "lycoris_single_transformer_blocks_0_attn_to_v.alpha": ([], "BF16"),
    "lycoris_single_transformer_blocks_0_attn_to_v.lokr_w1": ([16, 16], "BF16"),
    "lycoris_single_transformer_blocks_0_attn_to_v.lokr_w2": ([192, 192], "BF16"),
    "lycoris_transformer_blocks_0_attn_add_k_proj.alpha": ([], "BF16"),
    "lycoris_transformer_blocks_0_attn_add_k_proj.lokr_w1": ([16, 16], "BF16"),
    "lycoris_transformer_blocks_0_attn_add_k_proj.lokr_w2": ([192, 192], "BF16"),
    "lycoris_transformer_blocks_0_attn_add_q_proj.alpha": ([], "BF16"),
    "lycoris_transformer_blocks_0_attn_add_q_proj.lokr_w1": ([16, 16], "BF16"),
    "lycoris_transformer_blocks_0_attn_add_q_proj.lokr_w2": ([192, 192], "BF16"),
    "lycoris_transformer_blocks_0_attn_add_v_proj.alpha": ([], "BF16"),
    "lycoris_transformer_blocks_0_attn_add_v_proj.lokr_w1": ([16, 16], "BF16"),
    "lycoris_transformer_blocks_0_attn_add_v_proj.lokr_w2": ([192, 192], "BF16"),
    "lycoris_transformer_blocks_0_attn_to_add_out.alpha": ([], "BF16"),
    "lycoris_transformer_blocks_0_attn_to_add_out.lokr_w1": ([16, 16], "BF16"),
    "lycoris_transformer_blocks_0_attn_to_add_out.lokr_w2": ([192, 192], "BF16"),
    "lycoris_transformer_blocks_0_attn_to_k.alpha": ([], "BF16"),
    "lycoris_transformer_blocks_0_attn_to_k.lokr_w1": ([16, 16], "BF16"),
    "lycoris_transformer_blocks_0_attn_to_k.lokr_w2": ([192, 192], "BF16"),
    "lycoris_transformer_blocks_0_attn_to_out_0.alpha": ([], "BF16"),
    "lycoris_transformer_blocks_0_attn_to_out_0.lokr_w1": ([16, 16], "BF16"),
    "lycoris_transformer_blocks_0_attn_to_out_0.lokr_w2": ([192, 192], "BF16"),
    "lycoris_transformer_blocks_0_attn_to_q.alpha": ([], "BF16"),
    "lycoris_transformer_blocks_0_attn_to_q.lokr_w1": ([16, 16], "BF16"),
    "lycoris_transformer_blocks_0_attn_to_q.lokr_w2": ([192, 192], "BF16"),
    "lycoris_transformer_blocks_0_attn_to_v.alpha": ([], "BF16"),
    "lycoris_transformer_blocks_0_attn_to_v.lokr_w1": ([16, 16], "BF16"),
    "lycoris_transformer_blocks_0_attn_to_v.lokr_w2": ([192, 192], "BF16"),
    "lycoris_transformer_blocks_0_ff_context_net_0_proj.alpha": ([], "BF16"),
    "lycoris_transformer_blocks_0_ff_context_net_0_proj.lokr_w1": ([8, 8], "BF16"),
    "lycoris_transformer_blocks_0_ff_context_net_0_proj.lokr_w2": ([1536, 384], "BF16"),
    "lycoris_transformer_blocks_0_ff_context_net_2.alpha": ([], "BF16"),
    "lycoris_transformer_blocks_0_ff_context_net_2.lokr_w1": ([8, 8], "BF16"),
    "lycoris_transformer_blocks_0_ff_context_net_2.lokr_w2": ([384, 1536], "BF16"),
    "lycoris_transformer_blocks_0_ff_net_0_proj.alpha": ([], "BF16"),
    "lycoris_transformer_blocks_0_ff_net_0_proj.lokr_w1": ([8, 8], "BF16"),
    "lycoris_transformer_blocks_0_ff_net_0_proj.lokr_w2": ([1536, 384], "BF16"),
    "lycoris_transformer_blocks_0_ff_net_2.alpha": ([], "BF16"),
    "lycoris_transformer_blocks_0_ff_net_2.lokr_w1": ([8, 8], "BF16"),
    "lycoris_transformer_blocks_0_ff_net_2.lokr_w2": ([384, 1536], "BF16"),
}
