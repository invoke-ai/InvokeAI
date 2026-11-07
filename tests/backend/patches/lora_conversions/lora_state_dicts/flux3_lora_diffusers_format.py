"""A diffusers/PEFT FLUX.3 LoRA, akhaliq/flux-3-image-klein-angles-lora (diffusers' DreamBooth trainer, rank 16).

Captured on 2026-10-31 with `scripts/capture_state_dict_fixture.py`,
which reads the header only, from
`akhaliq/flux-3-image-klein-angles-lora/pytorch_lora_weights.safetensors`.

Subsetting rule: `transformer.mode_blocks.image` index 0, `transformer.mode_blocks.image_cond` index 0, `transformer.single_transformer_blocks` index 0, `transformer.txt_mode_blocks` index 0, plus every key outside a stack. Values are `(shape, dtype)`.

Keys are `transformer.` + diffusers' module path + `lora_A`/`lora_B`, with no `alpha`: the header's
`lora_adapter_metadata` (not recorded here) declares `r` 16 and `lora_alpha` 16, so the rank default scales it by 1.
The file's 516 tensors cover the attention and feed-forward Linears of every block. The conversion keeps these names; a
re-upload that renamed a module would show up as a key that resolves to no Linear.
"""

state_dict_keys: dict[str, tuple[list[int], str]] = {
    "transformer.mode_blocks.image.0.attn.to_k.lora_A.weight": ([16, 3072], "BF16"),
    "transformer.mode_blocks.image.0.attn.to_k.lora_B.weight": ([3072, 16], "BF16"),
    "transformer.mode_blocks.image.0.attn.to_out.0.lora_A.weight": ([16, 3072], "BF16"),
    "transformer.mode_blocks.image.0.attn.to_out.0.lora_B.weight": ([3072, 16], "BF16"),
    "transformer.mode_blocks.image.0.attn.to_q.lora_A.weight": ([16, 3072], "BF16"),
    "transformer.mode_blocks.image.0.attn.to_q.lora_B.weight": ([3072, 16], "BF16"),
    "transformer.mode_blocks.image.0.attn.to_v.lora_A.weight": ([16, 3072], "BF16"),
    "transformer.mode_blocks.image.0.attn.to_v.lora_B.weight": ([3072, 16], "BF16"),
    "transformer.mode_blocks.image.0.ff.linear_in.lora_A.weight": ([16, 3072], "BF16"),
    "transformer.mode_blocks.image.0.ff.linear_in.lora_B.weight": ([18432, 16], "BF16"),
    "transformer.mode_blocks.image.0.ff.linear_out.lora_A.weight": ([16, 9216], "BF16"),
    "transformer.mode_blocks.image.0.ff.linear_out.lora_B.weight": ([3072, 16], "BF16"),
    "transformer.mode_blocks.image_cond.0.attn.to_k.lora_A.weight": ([16, 3072], "BF16"),
    "transformer.mode_blocks.image_cond.0.attn.to_k.lora_B.weight": ([3072, 16], "BF16"),
    "transformer.mode_blocks.image_cond.0.attn.to_out.0.lora_A.weight": ([16, 3072], "BF16"),
    "transformer.mode_blocks.image_cond.0.attn.to_out.0.lora_B.weight": ([3072, 16], "BF16"),
    "transformer.mode_blocks.image_cond.0.attn.to_q.lora_A.weight": ([16, 3072], "BF16"),
    "transformer.mode_blocks.image_cond.0.attn.to_q.lora_B.weight": ([3072, 16], "BF16"),
    "transformer.mode_blocks.image_cond.0.attn.to_v.lora_A.weight": ([16, 3072], "BF16"),
    "transformer.mode_blocks.image_cond.0.attn.to_v.lora_B.weight": ([3072, 16], "BF16"),
    "transformer.mode_blocks.image_cond.0.ff.linear_in.lora_A.weight": ([16, 3072], "BF16"),
    "transformer.mode_blocks.image_cond.0.ff.linear_in.lora_B.weight": ([18432, 16], "BF16"),
    "transformer.mode_blocks.image_cond.0.ff.linear_out.lora_A.weight": ([16, 9216], "BF16"),
    "transformer.mode_blocks.image_cond.0.ff.linear_out.lora_B.weight": ([3072, 16], "BF16"),
    "transformer.single_transformer_blocks.0.attn.to_k.lora_A.weight": ([16, 3072], "BF16"),
    "transformer.single_transformer_blocks.0.attn.to_k.lora_B.weight": ([3072, 16], "BF16"),
    "transformer.single_transformer_blocks.0.attn.to_out.0.lora_A.weight": ([16, 3072], "BF16"),
    "transformer.single_transformer_blocks.0.attn.to_out.0.lora_B.weight": ([3072, 16], "BF16"),
    "transformer.single_transformer_blocks.0.attn.to_q.lora_A.weight": ([16, 3072], "BF16"),
    "transformer.single_transformer_blocks.0.attn.to_q.lora_B.weight": ([3072, 16], "BF16"),
    "transformer.single_transformer_blocks.0.attn.to_v.lora_A.weight": ([16, 3072], "BF16"),
    "transformer.single_transformer_blocks.0.attn.to_v.lora_B.weight": ([3072, 16], "BF16"),
    "transformer.single_transformer_blocks.0.ff.linear_in.lora_A.weight": ([16, 3072], "BF16"),
    "transformer.single_transformer_blocks.0.ff.linear_in.lora_B.weight": ([18432, 16], "BF16"),
    "transformer.single_transformer_blocks.0.ff.linear_out.lora_A.weight": ([16, 9216], "BF16"),
    "transformer.single_transformer_blocks.0.ff.linear_out.lora_B.weight": ([3072, 16], "BF16"),
    "transformer.txt_mode_blocks.0.attn.to_k.lora_A.weight": ([16, 3072], "BF16"),
    "transformer.txt_mode_blocks.0.attn.to_k.lora_B.weight": ([3072, 16], "BF16"),
    "transformer.txt_mode_blocks.0.attn.to_out.0.lora_A.weight": ([16, 3072], "BF16"),
    "transformer.txt_mode_blocks.0.attn.to_out.0.lora_B.weight": ([3072, 16], "BF16"),
    "transformer.txt_mode_blocks.0.attn.to_q.lora_A.weight": ([16, 3072], "BF16"),
    "transformer.txt_mode_blocks.0.attn.to_q.lora_B.weight": ([3072, 16], "BF16"),
    "transformer.txt_mode_blocks.0.attn.to_v.lora_A.weight": ([16, 3072], "BF16"),
    "transformer.txt_mode_blocks.0.attn.to_v.lora_B.weight": ([3072, 16], "BF16"),
    "transformer.txt_mode_blocks.0.ff.linear_in.lora_A.weight": ([16, 3072], "BF16"),
    "transformer.txt_mode_blocks.0.ff.linear_in.lora_B.weight": ([18432, 16], "BF16"),
    "transformer.txt_mode_blocks.0.ff.linear_out.lora_A.weight": ([16, 9216], "BF16"),
    "transformer.txt_mode_blocks.0.ff.linear_out.lora_B.weight": ([3072, 16], "BF16"),
}
