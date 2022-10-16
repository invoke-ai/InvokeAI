from enum import Enum


class CrossAttention:

    class AttentionType(Enum):
        SELF = 1
        TOKENS = 2

    @classmethod
    def get_attention_module(cls, model, which: AttentionType):
        which_attn = "attn1" if which is cls.AttentionType.SELF else "attn2"
        module = next(module for name, module in model.named_modules() if
                      type(module).__name__ == "CrossAttention" and which_attn in name)
        return module

    @classmethod
    def inject_attention_mask_capture(cls, model, callback):
        pass