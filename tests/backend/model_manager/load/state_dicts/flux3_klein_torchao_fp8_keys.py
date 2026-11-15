"""unsloth's torchao FP8 build of FLUX.3 Image Klein, `flux-3-image-klein-FP8.safetensors`.

Captured on 2026-11-06 with `scripts/capture_state_dict_fixture.py`, which reads the header only, from a local copy of
unsloth's `flux-3-image-klein-FP8.safetensors`.

Subsetting rule: `mode_blocks.image` index 0, `mode_blocks.image_cond` index 0, `single_transformer_blocks` index 0,
`txt_mode_blocks` index 0, plus every key outside a stack. Values are `(shape, dtype)`.

The only single-file FLUX.3 build under diffusers' key names rather than Black Forest Labs'. Every block linear but the
text blocks' (`attn.to_{q,k,v}`, `attn.to_out.0`, `ff.linear_in`, `ff.linear_out`) is torchao's `Float8Tensor`: e4m3
codes under `<layer>._weight_qdata` and one float32 scale per output row, shaped `[out, 1]`, under
`<layer>._weight_scale`, with no `<layer>.weight`. The modulations, embedders, `norm_out.linear` and `proj_out` are
dense BF16, the q/k norms F32. It has the flow-map embedder (`target_timestep_embedder`), so it is Klein. A re-upload
as INT8 or with block-wise scales keeps these names and changes the payload dtype or the scale shape.
"""

state_dict_keys: dict[str, tuple[list[int], str]] = {
    "context_embedder.weight": ([3072, 20480], "BF16"),
    "early_stream_modulations.image.linear.weight": ([9216, 3072], "BF16"),
    "early_stream_modulations.image_cond.linear.weight": ([9216, 3072], "BF16"),
    "early_stream_modulations.txt.linear.weight": ([9216, 3072], "BF16"),
    "mode_blocks.image.0.attn.norm_k.weight": ([128], "F32"),
    "mode_blocks.image.0.attn.norm_q.weight": ([128], "F32"),
    "mode_blocks.image.0.attn.to_k._weight_qdata": ([3072, 3072], "F8_E4M3"),
    "mode_blocks.image.0.attn.to_k._weight_scale": ([3072, 1], "F32"),
    "mode_blocks.image.0.attn.to_out.0._weight_qdata": ([3072, 3072], "F8_E4M3"),
    "mode_blocks.image.0.attn.to_out.0._weight_scale": ([3072, 1], "F32"),
    "mode_blocks.image.0.attn.to_q._weight_qdata": ([3072, 3072], "F8_E4M3"),
    "mode_blocks.image.0.attn.to_q._weight_scale": ([3072, 1], "F32"),
    "mode_blocks.image.0.attn.to_v._weight_qdata": ([3072, 3072], "F8_E4M3"),
    "mode_blocks.image.0.attn.to_v._weight_scale": ([3072, 1], "F32"),
    "mode_blocks.image.0.ff.linear_in._weight_qdata": ([18432, 3072], "F8_E4M3"),
    "mode_blocks.image.0.ff.linear_in._weight_scale": ([18432, 1], "F32"),
    "mode_blocks.image.0.ff.linear_out._weight_qdata": ([3072, 9216], "F8_E4M3"),
    "mode_blocks.image.0.ff.linear_out._weight_scale": ([3072, 1], "F32"),
    "mode_blocks.image_cond.0.attn.norm_k.weight": ([128], "F32"),
    "mode_blocks.image_cond.0.attn.norm_q.weight": ([128], "F32"),
    "mode_blocks.image_cond.0.attn.to_k._weight_qdata": ([3072, 3072], "F8_E4M3"),
    "mode_blocks.image_cond.0.attn.to_k._weight_scale": ([3072, 1], "F32"),
    "mode_blocks.image_cond.0.attn.to_out.0._weight_qdata": ([3072, 3072], "F8_E4M3"),
    "mode_blocks.image_cond.0.attn.to_out.0._weight_scale": ([3072, 1], "F32"),
    "mode_blocks.image_cond.0.attn.to_q._weight_qdata": ([3072, 3072], "F8_E4M3"),
    "mode_blocks.image_cond.0.attn.to_q._weight_scale": ([3072, 1], "F32"),
    "mode_blocks.image_cond.0.attn.to_v._weight_qdata": ([3072, 3072], "F8_E4M3"),
    "mode_blocks.image_cond.0.attn.to_v._weight_scale": ([3072, 1], "F32"),
    "mode_blocks.image_cond.0.ff.linear_in._weight_qdata": ([18432, 3072], "F8_E4M3"),
    "mode_blocks.image_cond.0.ff.linear_in._weight_scale": ([18432, 1], "F32"),
    "mode_blocks.image_cond.0.ff.linear_out._weight_qdata": ([3072, 9216], "F8_E4M3"),
    "mode_blocks.image_cond.0.ff.linear_out._weight_scale": ([3072, 1], "F32"),
    "norm_out.linear.weight": ([6144, 3072], "BF16"),
    "proj_out.weight": ([128, 3072], "BF16"),
    "single_stream_modulations.image.linear.weight": ([9216, 3072], "BF16"),
    "single_stream_modulations.image_cond.linear.weight": ([9216, 3072], "BF16"),
    "single_stream_modulations.txt.linear.weight": ([9216, 3072], "BF16"),
    "single_transformer_blocks.0.attn.norm_k.weight": ([128], "F32"),
    "single_transformer_blocks.0.attn.norm_q.weight": ([128], "F32"),
    "single_transformer_blocks.0.attn.to_k._weight_qdata": ([3072, 3072], "F8_E4M3"),
    "single_transformer_blocks.0.attn.to_k._weight_scale": ([3072, 1], "F32"),
    "single_transformer_blocks.0.attn.to_out.0._weight_qdata": ([3072, 3072], "F8_E4M3"),
    "single_transformer_blocks.0.attn.to_out.0._weight_scale": ([3072, 1], "F32"),
    "single_transformer_blocks.0.attn.to_q._weight_qdata": ([3072, 3072], "F8_E4M3"),
    "single_transformer_blocks.0.attn.to_q._weight_scale": ([3072, 1], "F32"),
    "single_transformer_blocks.0.attn.to_v._weight_qdata": ([3072, 3072], "F8_E4M3"),
    "single_transformer_blocks.0.attn.to_v._weight_scale": ([3072, 1], "F32"),
    "single_transformer_blocks.0.ff.linear_in._weight_qdata": ([18432, 3072], "F8_E4M3"),
    "single_transformer_blocks.0.ff.linear_in._weight_scale": ([18432, 1], "F32"),
    "single_transformer_blocks.0.ff.linear_out._weight_qdata": ([3072, 9216], "F8_E4M3"),
    "single_transformer_blocks.0.ff.linear_out._weight_scale": ([3072, 1], "F32"),
    "target_timestep_embedder.linear_1.weight": ([3072, 256], "BF16"),
    "target_timestep_embedder.linear_2.weight": ([3072, 3072], "BF16"),
    "timestep_embedder.linear_1.weight": ([3072, 256], "BF16"),
    "timestep_embedder.linear_2.weight": ([3072, 3072], "BF16"),
    "txt_mode_blocks.0.attn.norm_k.weight": ([128], "F32"),
    "txt_mode_blocks.0.attn.norm_q.weight": ([128], "F32"),
    "txt_mode_blocks.0.attn.to_k.weight": ([3072, 3072], "BF16"),
    "txt_mode_blocks.0.attn.to_out.0.weight": ([3072, 3072], "BF16"),
    "txt_mode_blocks.0.attn.to_q.weight": ([3072, 3072], "BF16"),
    "txt_mode_blocks.0.attn.to_v.weight": ([3072, 3072], "BF16"),
    "txt_mode_blocks.0.ff.linear_in.weight": ([18432, 3072], "BF16"),
    "txt_mode_blocks.0.ff.linear_out.weight": ([3072, 9216], "BF16"),
    "x_embedders.image.weight": ([3072, 128], "BF16"),
    "x_embedders.image_cond.weight": ([3072, 128], "BF16"),
}

# The header's `__metadata__` is one JSON blob per tensor (362 entries); only the file-level entries are kept, the
# second one shortened.
metadata: dict[str, str] = {
    "unsloth_format": "unsloth_prequant_transformer_state_dict_v1",
    "unsloth_metadata": '{"family": "flux.3-klein", "scheme": "fp8"}',
}
