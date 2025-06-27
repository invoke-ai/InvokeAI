from invokeai.backend.model_manager.omi.vendor.convert.lora.convert_lora_util import (
    LoraConversionKeySet,
    map_prefix_range,
)


def map_t5(key_prefix: LoraConversionKeySet) -> list[LoraConversionKeySet]:
    keys = []

    for k in map_prefix_range("encoder.block", "encoder.block", parent=key_prefix):
        keys += [LoraConversionKeySet("layer.0.SelfAttention.k", "layer.0.SelfAttention.k", parent=k)]
        keys += [LoraConversionKeySet("layer.0.SelfAttention.o", "layer.0.SelfAttention.o", parent=k)]
        keys += [LoraConversionKeySet("layer.0.SelfAttention.q", "layer.0.SelfAttention.q", parent=k)]
        keys += [LoraConversionKeySet("layer.0.SelfAttention.v", "layer.0.SelfAttention.v", parent=k)]
        keys += [LoraConversionKeySet("layer.1.DenseReluDense.wi_0", "layer.1.DenseReluDense.wi_0", parent=k)]
        keys += [LoraConversionKeySet("layer.1.DenseReluDense.wi_1", "layer.1.DenseReluDense.wi_1", parent=k)]
        keys += [LoraConversionKeySet("layer.1.DenseReluDense.wo", "layer.1.DenseReluDense.wo", parent=k)]

    return keys
