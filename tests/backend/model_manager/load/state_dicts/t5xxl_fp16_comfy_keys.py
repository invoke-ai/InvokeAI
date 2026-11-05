"""Key layout of ComfyUI's fp16 T5-XXL encoder (`t5xxl_fp16`), as FLUX.1 and SD 3 users bring it.

Captured on 2026-10-11 with `scripts/capture_state_dict_fixture.py`,
which reads the header only, from
`comfyanonymous/flux_text_encoders/t5xxl_fp16.safetensors`.

Subsetting rule: `encoder.block` index 0, plus every key outside a stack. Values are `(shape, dtype)`.

It is `T5EncoderModel`'s own state dict, key for key: no prefix, no config, no tokenizer. The token
embedding is written twice, as `shared.weight` and `encoder.embed_tokens.weight`, which the model ties
to one parameter -- the loader drops the copy rather than hold 256 MiB twice. Only block 0 carries a
`relative_attention_bias`; UMT5, which shares every other name, has one per block, and that is what
keeps the two apart.
"""

state_dict_keys: dict[str, tuple[list[int], str]] = {
    "encoder.block.0.layer.0.SelfAttention.k.weight": ([4096, 4096], "F16"),
    "encoder.block.0.layer.0.SelfAttention.o.weight": ([4096, 4096], "F16"),
    "encoder.block.0.layer.0.SelfAttention.q.weight": ([4096, 4096], "F16"),
    "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight": ([32, 64], "F16"),
    "encoder.block.0.layer.0.SelfAttention.v.weight": ([4096, 4096], "F16"),
    "encoder.block.0.layer.0.layer_norm.weight": ([4096], "F16"),
    "encoder.block.0.layer.1.DenseReluDense.wi_0.weight": ([10240, 4096], "F16"),
    "encoder.block.0.layer.1.DenseReluDense.wi_1.weight": ([10240, 4096], "F16"),
    "encoder.block.0.layer.1.DenseReluDense.wo.weight": ([4096, 10240], "F16"),
    "encoder.block.0.layer.1.layer_norm.weight": ([4096], "F16"),
    "encoder.embed_tokens.weight": ([32128, 4096], "F16"),
    "encoder.final_layer_norm.weight": ([4096], "F16"),
    "shared.weight": ([32128, 4096], "F16"),
}
