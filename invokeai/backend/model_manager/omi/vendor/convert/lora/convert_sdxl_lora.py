from invokeai.backend.model_manager.omi.vendor.convert.lora.convert_clip import map_clip
from invokeai.backend.model_manager.omi.vendor.convert.lora.convert_lora_util import (
    LoraConversionKeySet,
    map_prefix_range,
)


def __map_unet_resnet_block(key_prefix: LoraConversionKeySet) -> list[LoraConversionKeySet]:
    keys = []

    keys += [LoraConversionKeySet("emb_layers.1", "time_emb_proj", parent=key_prefix)]
    keys += [LoraConversionKeySet("in_layers.2", "conv1", parent=key_prefix)]
    keys += [LoraConversionKeySet("out_layers.3", "conv2", parent=key_prefix)]
    keys += [LoraConversionKeySet("skip_connection", "conv_shortcut", parent=key_prefix)]

    return keys


def __map_unet_attention_block(key_prefix: LoraConversionKeySet) -> list[LoraConversionKeySet]:
    keys = []

    keys += [LoraConversionKeySet("proj_in", "proj_in", parent=key_prefix)]
    keys += [LoraConversionKeySet("proj_out", "proj_out", parent=key_prefix)]
    for k in map_prefix_range("transformer_blocks", "transformer_blocks", parent=key_prefix):
        keys += [LoraConversionKeySet("attn1.to_q", "attn1.to_q", parent=k)]
        keys += [LoraConversionKeySet("attn1.to_k", "attn1.to_k", parent=k)]
        keys += [LoraConversionKeySet("attn1.to_v", "attn1.to_v", parent=k)]
        keys += [LoraConversionKeySet("attn1.to_out.0", "attn1.to_out.0", parent=k)]
        keys += [LoraConversionKeySet("attn2.to_q", "attn2.to_q", parent=k)]
        keys += [LoraConversionKeySet("attn2.to_k", "attn2.to_k", parent=k)]
        keys += [LoraConversionKeySet("attn2.to_v", "attn2.to_v", parent=k)]
        keys += [LoraConversionKeySet("attn2.to_out.0", "attn2.to_out.0", parent=k)]
        keys += [LoraConversionKeySet("ff.net.0.proj", "ff.net.0.proj", parent=k)]
        keys += [LoraConversionKeySet("ff.net.2", "ff.net.2", parent=k)]

    return keys


def __map_unet_down_blocks(key_prefix: LoraConversionKeySet) -> list[LoraConversionKeySet]:
    keys = []

    keys += __map_unet_resnet_block(LoraConversionKeySet("1.0", "0.resnets.0", parent=key_prefix))
    keys += __map_unet_resnet_block(LoraConversionKeySet("2.0", "0.resnets.1", parent=key_prefix))
    keys += [LoraConversionKeySet("3.0.op", "0.downsamplers.0.conv", parent=key_prefix)]

    keys += __map_unet_resnet_block(LoraConversionKeySet("4.0", "1.resnets.0", parent=key_prefix))
    keys += __map_unet_attention_block(LoraConversionKeySet("4.1", "1.attentions.0", parent=key_prefix))
    keys += __map_unet_resnet_block(LoraConversionKeySet("5.0", "1.resnets.1", parent=key_prefix))
    keys += __map_unet_attention_block(LoraConversionKeySet("5.1", "1.attentions.1", parent=key_prefix))
    keys += [LoraConversionKeySet("6.0.op", "1.downsamplers.0.conv", parent=key_prefix)]

    keys += __map_unet_resnet_block(LoraConversionKeySet("7.0", "2.resnets.0", parent=key_prefix))
    keys += __map_unet_attention_block(LoraConversionKeySet("7.1", "2.attentions.0", parent=key_prefix))
    keys += __map_unet_resnet_block(LoraConversionKeySet("8.0", "2.resnets.1", parent=key_prefix))
    keys += __map_unet_attention_block(LoraConversionKeySet("8.1", "2.attentions.1", parent=key_prefix))

    return keys


def __map_unet_mid_block(key_prefix: LoraConversionKeySet) -> list[LoraConversionKeySet]:
    keys = []

    keys += __map_unet_resnet_block(LoraConversionKeySet("0", "resnets.0", parent=key_prefix))
    keys += __map_unet_attention_block(LoraConversionKeySet("1", "attentions.0", parent=key_prefix))
    keys += __map_unet_resnet_block(LoraConversionKeySet("2", "resnets.1", parent=key_prefix))

    return keys


def __map_unet_up_block(key_prefix: LoraConversionKeySet) -> list[LoraConversionKeySet]:
    keys = []

    keys += __map_unet_resnet_block(LoraConversionKeySet("0.0", "0.resnets.0", parent=key_prefix))
    keys += __map_unet_attention_block(LoraConversionKeySet("0.1", "0.attentions.0", parent=key_prefix))
    keys += __map_unet_resnet_block(LoraConversionKeySet("1.0", "0.resnets.1", parent=key_prefix))
    keys += __map_unet_attention_block(LoraConversionKeySet("1.1", "0.attentions.1", parent=key_prefix))
    keys += __map_unet_resnet_block(LoraConversionKeySet("2.0", "0.resnets.2", parent=key_prefix))
    keys += __map_unet_attention_block(LoraConversionKeySet("2.1", "0.attentions.2", parent=key_prefix))
    keys += [LoraConversionKeySet("2.2.conv", "0.upsamplers.0.conv", parent=key_prefix)]

    keys += __map_unet_resnet_block(LoraConversionKeySet("3.0", "1.resnets.0", parent=key_prefix))
    keys += __map_unet_attention_block(LoraConversionKeySet("3.1", "1.attentions.0", parent=key_prefix))
    keys += __map_unet_resnet_block(LoraConversionKeySet("4.0", "1.resnets.1", parent=key_prefix))
    keys += __map_unet_attention_block(LoraConversionKeySet("4.1", "1.attentions.1", parent=key_prefix))
    keys += __map_unet_resnet_block(LoraConversionKeySet("5.0", "1.resnets.2", parent=key_prefix))
    keys += __map_unet_attention_block(LoraConversionKeySet("5.1", "1.attentions.2", parent=key_prefix))
    keys += [LoraConversionKeySet("5.2.conv", "1.upsamplers.0.conv", parent=key_prefix)]

    keys += __map_unet_resnet_block(LoraConversionKeySet("6.0", "2.resnets.0", parent=key_prefix))
    keys += __map_unet_resnet_block(LoraConversionKeySet("7.0", "2.resnets.1", parent=key_prefix))
    keys += __map_unet_resnet_block(LoraConversionKeySet("8.0", "2.resnets.2", parent=key_prefix))

    return keys


def __map_unet(key_prefix: LoraConversionKeySet) -> list[LoraConversionKeySet]:
    keys = []

    keys += [LoraConversionKeySet("input_blocks.0.0", "conv_in", parent=key_prefix)]

    keys += [LoraConversionKeySet("time_embed.0", "time_embedding.linear_1", parent=key_prefix)]
    keys += [LoraConversionKeySet("time_embed.2", "time_embedding.linear_2", parent=key_prefix)]

    keys += [LoraConversionKeySet("label_emb.0.0", "add_embedding.linear_1", parent=key_prefix)]
    keys += [LoraConversionKeySet("label_emb.0.2", "add_embedding.linear_2", parent=key_prefix)]

    keys += __map_unet_down_blocks(LoraConversionKeySet("input_blocks", "down_blocks", parent=key_prefix))
    keys += __map_unet_mid_block(LoraConversionKeySet("middle_block", "mid_block", parent=key_prefix))
    keys += __map_unet_up_block(LoraConversionKeySet("output_blocks", "up_blocks", parent=key_prefix))

    keys += [LoraConversionKeySet("out.0", "conv_norm_out", parent=key_prefix)]
    keys += [LoraConversionKeySet("out.2", "conv_out", parent=key_prefix)]

    return keys


def convert_sdxl_lora_key_sets() -> list[LoraConversionKeySet]:
    keys = []

    keys += [LoraConversionKeySet("bundle_emb", "bundle_emb")]
    keys += __map_unet(LoraConversionKeySet("unet", "lora_unet"))
    keys += map_clip(LoraConversionKeySet("clip_l", "lora_te1"))
    keys += map_clip(LoraConversionKeySet("clip_g", "lora_te2"))

    return keys
