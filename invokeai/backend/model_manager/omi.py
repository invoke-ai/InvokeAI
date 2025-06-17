import torch
from invokeai.backend.util.logging import InvokeAILogger

logger = InvokeAILogger.get_logger()


def convert_from_omi(weights_sd):
    # convert from OMI to default LoRA
    # OMI format: {"prefix.module.name.lora_down.weight": weight, "prefix.module.name.lora_up.weight": weight, ...}
    # default LoRA format: {"prefix_module_name.lora_down.weight": weight, "prefix_module_name.lora_up.weight": weight, ...}

    new_weights_sd = {}
    prefix = "lora_unet_"
    lora_dims = {}
    weight_dtype = None
    for key, weight in weights_sd.items():
        omi_prefix, key_body = key.split(".", 1)
        if omi_prefix != "diffusion":
            logger.warning(f"unexpected key: {key} in OMI format")  # T5, CLIP, etc.
            continue

        # only supports lora_down, lora_up and alpha
        new_key = (
            f"{prefix}{key_body}".replace(".", "_")
            .replace("_lora_down_", ".lora_down.")
            .replace("_lora_up_", ".lora_up.")
            .replace("_alpha", ".alpha")
        )
        new_weights_sd[new_key] = weight

        lora_name = new_key.split(".")[0]  # before first dot
        if lora_name not in lora_dims and "lora_down" in new_key:
            lora_dims[lora_name] = weight.shape[0]
            if weight_dtype is None:
                weight_dtype = weight.dtype  # use first weight dtype for lora_down

    # add alpha with rank
    for lora_name, dim in lora_dims.items():
        alpha_key = f"{lora_name}.alpha"
        if alpha_key not in new_weights_sd:
            new_weights_sd[alpha_key] = torch.tensor(dim, dtype=weight_dtype)

    return new_weights_sd
