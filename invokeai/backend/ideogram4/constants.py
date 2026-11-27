SEQUENCE_PADDING_INDICATOR = -1

OUTPUT_IMAGE_INDICATOR = 2
LLM_TOKEN_INDICATOR = 3

# Image grid coordinates start at this offset so they never collide with text token indices
# (text positions start at 0 and never exceed max_text_tokens, which is well below this).
IMAGE_POSITION_OFFSET = 65536

# Layers of Qwen3-VL whose hidden states are concatenated and fed to the transformer.
QWEN3_VL_ACTIVATION_LAYERS = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35)
