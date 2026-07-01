from invokeai.backend.model_manager.omi.vendor.convert.lora.convert_lora_util import (
    LoraConversionKeySet,
    map_prefix_range,
)


def map_clip(key_prefix: LoraConversionKeySet) -> list[LoraConversionKeySet]:
    keys = []

    keys += [LoraConversionKeySet("text_projection", "text_projection", parent=key_prefix)]

    for k in map_prefix_range("text_model.encoder.layers", "text_model.encoder.layers", parent=key_prefix):
        keys += [LoraConversionKeySet("mlp.fc1", "mlp.fc1", parent=k)]
        keys += [LoraConversionKeySet("mlp.fc2", "mlp.fc2", parent=k)]
        keys += [LoraConversionKeySet("self_attn.k_proj", "self_attn.k_proj", parent=k)]
        keys += [LoraConversionKeySet("self_attn.out_proj", "self_attn.out_proj", parent=k)]
        keys += [LoraConversionKeySet("self_attn.q_proj", "self_attn.q_proj", parent=k)]
        keys += [LoraConversionKeySet("self_attn.v_proj", "self_attn.v_proj", parent=k)]

    return keys
