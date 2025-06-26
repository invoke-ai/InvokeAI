from invokeai.backend.model_manager.omi.vendor.convert.lora.convert_clip import map_clip
from invokeai.backend.model_manager.omi.vendor.convert.lora.convert_lora_util import (
    LoraConversionKeySet,
    map_prefix_range,
)
from invokeai.backend.model_manager.omi.vendor.convert.lora.convert_t5 import map_t5


def __map_double_transformer_block(key_prefix: LoraConversionKeySet) -> list[LoraConversionKeySet]:
    keys = []

    keys += [LoraConversionKeySet("img_attn.qkv.0", "attn.to_q", parent=key_prefix)]
    keys += [LoraConversionKeySet("img_attn.qkv.1", "attn.to_k", parent=key_prefix)]
    keys += [LoraConversionKeySet("img_attn.qkv.2", "attn.to_v", parent=key_prefix)]

    keys += [LoraConversionKeySet("txt_attn.qkv.0", "attn.add_q_proj", parent=key_prefix)]
    keys += [LoraConversionKeySet("txt_attn.qkv.1", "attn.add_k_proj", parent=key_prefix)]
    keys += [LoraConversionKeySet("txt_attn.qkv.2", "attn.add_v_proj", parent=key_prefix)]

    keys += [LoraConversionKeySet("img_attn.proj", "attn.to_out.0", parent=key_prefix)]
    keys += [LoraConversionKeySet("img_mlp.0", "ff.net.0.proj", parent=key_prefix)]
    keys += [LoraConversionKeySet("img_mlp.2", "ff.net.2", parent=key_prefix)]
    keys += [LoraConversionKeySet("img_mod.lin", "norm1.linear", parent=key_prefix)]

    keys += [LoraConversionKeySet("txt_attn.proj", "attn.to_add_out", parent=key_prefix)]
    keys += [LoraConversionKeySet("txt_mlp.0", "ff_context.net.0.proj", parent=key_prefix)]
    keys += [LoraConversionKeySet("txt_mlp.2", "ff_context.net.2", parent=key_prefix)]
    keys += [LoraConversionKeySet("txt_mod.lin", "norm1_context.linear", parent=key_prefix)]

    return keys


def __map_single_transformer_block(key_prefix: LoraConversionKeySet) -> list[LoraConversionKeySet]:
    keys = []

    keys += [LoraConversionKeySet("linear1.0", "attn.to_q", parent=key_prefix)]
    keys += [LoraConversionKeySet("linear1.1", "attn.to_k", parent=key_prefix)]
    keys += [LoraConversionKeySet("linear1.2", "attn.to_v", parent=key_prefix)]
    keys += [LoraConversionKeySet("linear1.3", "proj_mlp", parent=key_prefix)]

    keys += [LoraConversionKeySet("linear2", "proj_out", parent=key_prefix)]
    keys += [LoraConversionKeySet("modulation.lin", "norm.linear", parent=key_prefix)]

    return keys


def __map_transformer(key_prefix: LoraConversionKeySet) -> list[LoraConversionKeySet]:
    keys = []

    keys += [LoraConversionKeySet("txt_in", "context_embedder", parent=key_prefix)]
    keys += [
        LoraConversionKeySet("final_layer.adaLN_modulation.1", "norm_out.linear", parent=key_prefix, swap_chunks=True)
    ]
    keys += [LoraConversionKeySet("final_layer.linear", "proj_out", parent=key_prefix)]
    keys += [
        LoraConversionKeySet("guidance_in.in_layer", "time_text_embed.guidance_embedder.linear_1", parent=key_prefix)
    ]
    keys += [
        LoraConversionKeySet("guidance_in.out_layer", "time_text_embed.guidance_embedder.linear_2", parent=key_prefix)
    ]
    keys += [LoraConversionKeySet("vector_in.in_layer", "time_text_embed.text_embedder.linear_1", parent=key_prefix)]
    keys += [LoraConversionKeySet("vector_in.out_layer", "time_text_embed.text_embedder.linear_2", parent=key_prefix)]
    keys += [LoraConversionKeySet("time_in.in_layer", "time_text_embed.timestep_embedder.linear_1", parent=key_prefix)]
    keys += [LoraConversionKeySet("time_in.out_layer", "time_text_embed.timestep_embedder.linear_2", parent=key_prefix)]
    keys += [LoraConversionKeySet("img_in.proj", "x_embedder", parent=key_prefix)]

    for k in map_prefix_range("double_blocks", "transformer_blocks", parent=key_prefix):
        keys += __map_double_transformer_block(k)

    for k in map_prefix_range("single_blocks", "single_transformer_blocks", parent=key_prefix):
        keys += __map_single_transformer_block(k)

    return keys


def convert_flux_lora_key_sets() -> list[LoraConversionKeySet]:
    keys = []

    keys += [LoraConversionKeySet("bundle_emb", "bundle_emb")]
    keys += __map_transformer(LoraConversionKeySet("transformer", "lora_transformer"))
    keys += map_clip(LoraConversionKeySet("clip_l", "lora_te1"))
    keys += map_t5(LoraConversionKeySet("t5", "lora_te2"))

    return keys
