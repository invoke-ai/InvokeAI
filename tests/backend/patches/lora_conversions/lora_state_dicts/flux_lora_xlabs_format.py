# A sample state dict in the xlabs FLUX LoRA format.
# The xlabs format uses:
# - lora1 for image attention stream (img_attn)
# - lora2 for text attention stream (txt_attn)
# - qkv for query/key/value projection
# - proj for output projection
state_dict_keys = {
    "double_blocks.0.processor.proj_lora1.down.weight": [16, 3072],
    "double_blocks.0.processor.proj_lora1.up.weight": [3072, 16],
    "double_blocks.0.processor.proj_lora2.down.weight": [16, 3072],
    "double_blocks.0.processor.proj_lora2.up.weight": [3072, 16],
    "double_blocks.0.processor.qkv_lora1.down.weight": [16, 3072],
    "double_blocks.0.processor.qkv_lora1.up.weight": [9216, 16],
    "double_blocks.0.processor.qkv_lora2.down.weight": [16, 3072],
    "double_blocks.0.processor.qkv_lora2.up.weight": [9216, 16],
    "double_blocks.1.processor.proj_lora1.down.weight": [16, 3072],
    "double_blocks.1.processor.proj_lora1.up.weight": [3072, 16],
    "double_blocks.1.processor.proj_lora2.down.weight": [16, 3072],
    "double_blocks.1.processor.proj_lora2.up.weight": [3072, 16],
    "double_blocks.1.processor.qkv_lora1.down.weight": [16, 3072],
    "double_blocks.1.processor.qkv_lora1.up.weight": [9216, 16],
    "double_blocks.1.processor.qkv_lora2.down.weight": [16, 3072],
    "double_blocks.1.processor.qkv_lora2.up.weight": [9216, 16],
    "double_blocks.10.processor.proj_lora1.down.weight": [16, 3072],
    "double_blocks.10.processor.proj_lora1.up.weight": [3072, 16],
    "double_blocks.10.processor.proj_lora2.down.weight": [16, 3072],
    "double_blocks.10.processor.proj_lora2.up.weight": [3072, 16],
    "double_blocks.10.processor.qkv_lora1.down.weight": [16, 3072],
    "double_blocks.10.processor.qkv_lora1.up.weight": [9216, 16],
    "double_blocks.10.processor.qkv_lora2.down.weight": [16, 3072],
    "double_blocks.10.processor.qkv_lora2.up.weight": [9216, 16],
}
