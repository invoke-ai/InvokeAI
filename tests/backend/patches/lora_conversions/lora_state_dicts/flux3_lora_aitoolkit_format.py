"""An ai-toolkit FLUX.3 LoRA, AIImageStudio/RadianceChromeVoluptuous_Flux3Klein_v1.0 (rank 32).

Captured on 2026-10-31 with `scripts/capture_state_dict_fixture.py`,
which reads the header only, from
`AIImageStudio/RadianceChromeVoluptuous_Flux3Klein_v1.0/RadianceChromeVoluptuous_Flux3Klein_v1.0.safetensors`.

Subsetting rule: `diffusion_model.content_mode_blocks.image` index 0, `diffusion_model.content_mode_blocks.image_cond` index 0, `diffusion_model.single_blocks` index 0, `diffusion_model.txt_mode_blocks` index 0, plus every key outside a stack. Values are `(shape, dtype)`.

Keys are `diffusion_model.` + Black Forest Labs' native module path + `lora_A`/`lora_B`, with no `alpha`; the
file's 516 tensors cover q/k/v, attn_out, mlp_in and mlp_out of every block. `metadata` holds the header entries the
FLUX.1 ai-toolkit probe reads, copied by hand (the capture records none). That probe claimed this file as a FLUX.1 LoRA
until FLUX.3 had its own: `diffusion_model.single_blocks.N.` is FLUX.1's native layout too.
"""

state_dict_keys: dict[str, tuple[list[int], str]] = {
    "diffusion_model.content_mode_blocks.image.0.attn_out.lora_A.weight": ([32, 3072], "BF16"),
    "diffusion_model.content_mode_blocks.image.0.attn_out.lora_B.weight": ([3072, 32], "BF16"),
    "diffusion_model.content_mode_blocks.image.0.k_proj.lora_A.weight": ([32, 3072], "BF16"),
    "diffusion_model.content_mode_blocks.image.0.k_proj.lora_B.weight": ([3072, 32], "BF16"),
    "diffusion_model.content_mode_blocks.image.0.mlp_in.lora_A.weight": ([32, 3072], "BF16"),
    "diffusion_model.content_mode_blocks.image.0.mlp_in.lora_B.weight": ([18432, 32], "BF16"),
    "diffusion_model.content_mode_blocks.image.0.mlp_out.lora_A.weight": ([32, 9216], "BF16"),
    "diffusion_model.content_mode_blocks.image.0.mlp_out.lora_B.weight": ([3072, 32], "BF16"),
    "diffusion_model.content_mode_blocks.image.0.q_proj.lora_A.weight": ([32, 3072], "BF16"),
    "diffusion_model.content_mode_blocks.image.0.q_proj.lora_B.weight": ([3072, 32], "BF16"),
    "diffusion_model.content_mode_blocks.image.0.v_proj.lora_A.weight": ([32, 3072], "BF16"),
    "diffusion_model.content_mode_blocks.image.0.v_proj.lora_B.weight": ([3072, 32], "BF16"),
    "diffusion_model.content_mode_blocks.image_cond.0.attn_out.lora_A.weight": ([32, 3072], "BF16"),
    "diffusion_model.content_mode_blocks.image_cond.0.attn_out.lora_B.weight": ([3072, 32], "BF16"),
    "diffusion_model.content_mode_blocks.image_cond.0.k_proj.lora_A.weight": ([32, 3072], "BF16"),
    "diffusion_model.content_mode_blocks.image_cond.0.k_proj.lora_B.weight": ([3072, 32], "BF16"),
    "diffusion_model.content_mode_blocks.image_cond.0.mlp_in.lora_A.weight": ([32, 3072], "BF16"),
    "diffusion_model.content_mode_blocks.image_cond.0.mlp_in.lora_B.weight": ([18432, 32], "BF16"),
    "diffusion_model.content_mode_blocks.image_cond.0.mlp_out.lora_A.weight": ([32, 9216], "BF16"),
    "diffusion_model.content_mode_blocks.image_cond.0.mlp_out.lora_B.weight": ([3072, 32], "BF16"),
    "diffusion_model.content_mode_blocks.image_cond.0.q_proj.lora_A.weight": ([32, 3072], "BF16"),
    "diffusion_model.content_mode_blocks.image_cond.0.q_proj.lora_B.weight": ([3072, 32], "BF16"),
    "diffusion_model.content_mode_blocks.image_cond.0.v_proj.lora_A.weight": ([32, 3072], "BF16"),
    "diffusion_model.content_mode_blocks.image_cond.0.v_proj.lora_B.weight": ([3072, 32], "BF16"),
    "diffusion_model.single_blocks.0.attn_out.lora_A.weight": ([32, 3072], "BF16"),
    "diffusion_model.single_blocks.0.attn_out.lora_B.weight": ([3072, 32], "BF16"),
    "diffusion_model.single_blocks.0.k_proj.lora_A.weight": ([32, 3072], "BF16"),
    "diffusion_model.single_blocks.0.k_proj.lora_B.weight": ([3072, 32], "BF16"),
    "diffusion_model.single_blocks.0.mlp_in.lora_A.weight": ([32, 3072], "BF16"),
    "diffusion_model.single_blocks.0.mlp_in.lora_B.weight": ([18432, 32], "BF16"),
    "diffusion_model.single_blocks.0.mlp_out.lora_A.weight": ([32, 9216], "BF16"),
    "diffusion_model.single_blocks.0.mlp_out.lora_B.weight": ([3072, 32], "BF16"),
    "diffusion_model.single_blocks.0.q_proj.lora_A.weight": ([32, 3072], "BF16"),
    "diffusion_model.single_blocks.0.q_proj.lora_B.weight": ([3072, 32], "BF16"),
    "diffusion_model.single_blocks.0.v_proj.lora_A.weight": ([32, 3072], "BF16"),
    "diffusion_model.single_blocks.0.v_proj.lora_B.weight": ([3072, 32], "BF16"),
    "diffusion_model.txt_mode_blocks.0.attn_out.lora_A.weight": ([32, 3072], "BF16"),
    "diffusion_model.txt_mode_blocks.0.attn_out.lora_B.weight": ([3072, 32], "BF16"),
    "diffusion_model.txt_mode_blocks.0.k_proj.lora_A.weight": ([32, 3072], "BF16"),
    "diffusion_model.txt_mode_blocks.0.k_proj.lora_B.weight": ([3072, 32], "BF16"),
    "diffusion_model.txt_mode_blocks.0.mlp_in.lora_A.weight": ([32, 3072], "BF16"),
    "diffusion_model.txt_mode_blocks.0.mlp_in.lora_B.weight": ([18432, 32], "BF16"),
    "diffusion_model.txt_mode_blocks.0.mlp_out.lora_A.weight": ([32, 9216], "BF16"),
    "diffusion_model.txt_mode_blocks.0.mlp_out.lora_B.weight": ([3072, 32], "BF16"),
    "diffusion_model.txt_mode_blocks.0.q_proj.lora_A.weight": ([32, 3072], "BF16"),
    "diffusion_model.txt_mode_blocks.0.q_proj.lora_B.weight": ([3072, 32], "BF16"),
    "diffusion_model.txt_mode_blocks.0.v_proj.lora_A.weight": ([32, 3072], "BF16"),
    "diffusion_model.txt_mode_blocks.0.v_proj.lora_B.weight": ([3072, 32], "BF16"),
}

# Two of the header's entries, verbatim: the trainer and the base model it names.
metadata: dict[str, str] = {
    "software": '{"name": "ai-toolkit", "repo": "https://github.com/ostris/ai-toolkit", "version": "0.13.26"}',
    "ss_base_model_version": "flux3",
}
