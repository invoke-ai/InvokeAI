# Copyright (c) 2023 The InvokeAI Development Team
"""Utilities used by the Model Manager"""


def lora_token_vector_length(checkpoint: dict) -> int:
    """
    Given a checkpoint in memory, return the lora token vector length

    :param checkpoint: The checkpoint
    """

    def _get_shape_1(key: str, tensor, checkpoint) -> int:
        lora_token_vector_length = None

        if "." not in key:
            return lora_token_vector_length  # wrong key format
        model_key, lora_key = key.split(".", 1)

        # check lora/locon
        if lora_key == "lora_down.weight":
            lora_token_vector_length = tensor.shape[1]

        # check loha (don't worry about hada_t1/hada_t2 as it used only in 4d shapes)
        elif lora_key in ["hada_w1_b", "hada_w2_b"]:
            lora_token_vector_length = tensor.shape[1]

        # check lokr (don't worry about lokr_t2 as it used only in 4d shapes)
        elif "lokr_" in lora_key:
            if model_key + ".lokr_w1" in checkpoint:
                _lokr_w1 = checkpoint[model_key + ".lokr_w1"]
            elif model_key + "lokr_w1_b" in checkpoint:
                _lokr_w1 = checkpoint[model_key + ".lokr_w1_b"]
            else:
                return lora_token_vector_length  # unknown format

            if model_key + ".lokr_w2" in checkpoint:
                _lokr_w2 = checkpoint[model_key + ".lokr_w2"]
            elif model_key + "lokr_w2_b" in checkpoint:
                _lokr_w2 = checkpoint[model_key + ".lokr_w2_b"]
            else:
                return lora_token_vector_length  # unknown format

            lora_token_vector_length = _lokr_w1.shape[1] * _lokr_w2.shape[1]

        elif lora_key == "diff":
            lora_token_vector_length = tensor.shape[1]

        # ia3 can be detected only by shape[0] in text encoder
        elif lora_key == "weight" and "lora_unet_" not in model_key:
            lora_token_vector_length = tensor.shape[0]

        return lora_token_vector_length

    lora_token_vector_length = None
    lora_te1_length = None
    lora_te2_length = None
    for key, tensor in checkpoint.items():
        if key.startswith("lora_unet_") and ("_attn2_to_k." in key or "_attn2_to_v." in key):
            lora_token_vector_length = _get_shape_1(key, tensor, checkpoint)
        elif key.startswith("lora_unet_") and (
            "time_emb_proj.lora_down" in key
        ):  # recognizes format at https://civitai.com/models/224641
            lora_token_vector_length = _get_shape_1(key, tensor, checkpoint)
        elif key.startswith("lora_te") and "_self_attn_" in key:
            tmp_length = _get_shape_1(key, tensor, checkpoint)
            if key.startswith("lora_te_"):
                lora_token_vector_length = tmp_length
            elif key.startswith("lora_te1_"):
                lora_te1_length = tmp_length
            elif key.startswith("lora_te2_"):
                lora_te2_length = tmp_length

        if lora_te1_length is not None and lora_te2_length is not None:
            lora_token_vector_length = lora_te1_length + lora_te2_length

        if lora_token_vector_length is not None:
            break

    return lora_token_vector_length
