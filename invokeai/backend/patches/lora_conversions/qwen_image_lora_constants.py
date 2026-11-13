import re

# Qwen Image Edit LoRA prefix constants
# These prefixes are used for key mapping when applying LoRA patches to Qwen Image Edit models

# Prefix for Qwen Image Edit transformer LoRA layers
QWEN_IMAGE_EDIT_LORA_TRANSFORMER_PREFIX = "lora_transformer-"

# Modules of Qwen Image's dual-stream blocks that no other family naming `transformer_blocks.N` has: its per-stream
# modulation and MLPs, dotted or Kohya-flattened. A LoRA naming one of them is Qwen Image's alone.
QWEN_IMAGE_ONLY_MODULE_RE = re.compile(r"transformer_blocks[._]\d+[._](?:img_mod|txt_mod|img_mlp[._]net|txt_mlp)[._]")

# Prefix for Qwen2.5-VL text encoder LoRA layers
QWEN_IMAGE_LORA_QWEN_VL_PREFIX = "lora_qwen_vl-"
