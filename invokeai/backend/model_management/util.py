# Copyright (c) 2023 The InvokeAI Development Team
"""Utilities used by the Model Manager"""


def lora_token_vector_length(checkpoint: dict) -> int:
    """
    Given a checkpoint in memory, return the lora token vector length

    :param checkpoint: The checkpoint
    """

    def _handle_unet_key(key, tensor, checkpoint):
        lora_token_vector_length = None
        if "_attn2_to_k." not in key and "_attn2_to_v." not in key:
            return lora_token_vector_length

        # check lora/locon
        if ".lora_up.weight" in key:
            lora_token_vector_length = tensor.shape[0]
        elif ".lora_down.weight" in key:
            lora_token_vector_length = tensor.shape[1]

        # check loha (don't worry about hada_t1/hada_t2 as it used only in 4d shapes)
        elif ".hada_w1_b" in key or ".hada_w2_b" in key:
            lora_token_vector_length = tensor.shape[1]

        # check lokr (don't worry about lokr_t2 as it used only in 4d shapes)
        elif ".lokr_" in key:
            _lokr_key = key.split(".")[0]

            if _lokr_key + ".lokr_w1" in checkpoint:
                _lokr_w1 = checkpoint[_lokr_key + ".lokr_w1"]
            elif _lokr_key + "lokr_w1_b" in checkpoint:
                _lokr_w1 = checkpoint[_lokr_key + ".lokr_w1_b"]
            else:
                return lora_token_vector_length  # unknown format

            if _lokr_key + ".lokr_w2" in checkpoint:
                _lokr_w2 = checkpoint[_lokr_key + ".lokr_w2"]
            elif _lokr_key + "lokr_w2_b" in checkpoint:
                _lokr_w2 = checkpoint[_lokr_key + ".lokr_w2_b"]
            else:
                return lora_token_vector_length  # unknown format

            lora_token_vector_length = _lokr_w1.shape[1] * _lokr_w2.shape[1]

        elif ".diff" in key:
            lora_token_vector_length = tensor.shape[1]

        return lora_token_vector_length

    def _handle_te_key(key, tensor, checkpoint):
        lora_token_vector_length = None
        if "text_model_encoder_layers_" not in key:
            return lora_token_vector_length

        # skip detect by mlp
        if "_self_attn_" not in key:
            return lora_token_vector_length

        # check lora/locon
        if ".lora_up.weight" in key:
            lora_token_vector_length = tensor.shape[0]
        elif ".lora_down.weight" in key:
            lora_token_vector_length = tensor.shape[1]

        # check loha (don't worry about hada_t1/hada_t2 as it used only in 4d shapes)
        elif ".hada_w1_a" in key or ".hada_w2_a" in key:
            lora_token_vector_length = tensor.shape[0]
        elif ".hada_w1_b" in key or ".hada_w2_b" in key:
            lora_token_vector_length = tensor.shape[1]

        # check lokr (don't worry about lokr_t2 as it used only in 4d shapes)
        elif ".lokr_" in key:
            _lokr_key = key.split(".")[0]

            if _lokr_key + ".lokr_w1" in checkpoint:
                _lokr_w1 = checkpoint[_lokr_key + ".lokr_w1"]
            elif _lokr_key + "lokr_w1_b" in checkpoint:
                _lokr_w1 = checkpoint[_lokr_key + ".lokr_w1_b"]
            else:
                return lora_token_vector_length  # unknown format

            if _lokr_key + ".lokr_w2" in checkpoint:
                _lokr_w2 = checkpoint[_lokr_key + ".lokr_w2"]
            elif _lokr_key + "lokr_w2_b" in checkpoint:
                _lokr_w2 = checkpoint[_lokr_key + ".lokr_w2_b"]
            else:
                return lora_token_vector_length  # unknown format

            lora_token_vector_length = _lokr_w1.shape[1] * _lokr_w2.shape[1]

        elif ".diff" in key:
            lora_token_vector_length = tensor.shape[1]

        return lora_token_vector_length

    lora_token_vector_length = None
    lora_te1_length = None
    lora_te2_length = None
    for key, tensor in checkpoint.items():
        if key.startswith("lora_unet_"):
            lora_token_vector_length = _handle_unet_key(key, tensor, checkpoint)
        elif key.startswith("lora_te_"):
            lora_token_vector_length = _handle_te_key(key, tensor, checkpoint)

        elif key.startswith("lora_te1_"):
            lora_te1_length = _handle_te_key(key, tensor, checkpoint)
        elif key.startswith("lora_te2_"):
            lora_te2_length = _handle_te_key(key, tensor, checkpoint)

        if lora_te1_length is not None and lora_te2_length is not None:
            lora_token_vector_length = lora_te1_length + lora_te2_length

        if lora_token_vector_length is not None:
            break

    return lora_token_vector_length
