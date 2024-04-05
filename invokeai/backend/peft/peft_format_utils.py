import torch
from diffusers.utils.state_dict_utils import convert_state_dict

KOHYA_SS_TO_PEFT = {
    "lora_down": "lora_A",
    "lora_up": "lora_B",
    # This is not a comprehensive dict. See `convert_state_dict_to_peft(...)` for more info on the conversion.
}


def convert_state_dict_kohya_to_peft(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    # TODO(ryand): Check that state_dict is in Kohya format.

    peft_partial_state_dict = convert_state_dict(state_dict, KOHYA_SS_TO_PEFT)

    peft_state_dict: dict[str, torch.Tensor] = {}
    for key, weight in peft_partial_state_dict.items():


    for kohya_key, weight in kohya_ss_partial_state_dict.items():
        if "text_encoder_2." in kohya_key:
            kohya_key = kohya_key.replace("text_encoder_2.", "lora_te2.")
        elif "text_encoder." in kohya_key:
            kohya_key = kohya_key.replace("text_encoder.", "lora_te1.")
        elif "unet" in kohya_key:
            kohya_key = kohya_key.replace("unet", "lora_unet")
        kohya_key = kohya_key.replace(".", "_", kohya_key.count(".") - 2)
        kohya_key = kohya_key.replace(peft_adapter_name, "")  # Kohya doesn't take names
        kohya_ss_state_dict[kohya_key] = weight
        if "lora_down" in kohya_key:
            alpha_key = f'{kohya_key.split(".")[0]}.alpha'
            kohya_ss_state_dict[alpha_key] = torch.tensor(len(weight))
def convert_state_dict_to_kohya(state_dict, original_type=None, **kwargs):
    r"""
    Converts a `PEFT` state dict to `Kohya` format that can be used in AUTOMATIC1111, ComfyUI, SD.Next, InvokeAI, etc.
    The method only supports the conversion from PEFT to Kohya for now.

    Args:
        state_dict (`dict[str, torch.Tensor]`):
            The state dict to convert.
        original_type (`StateDictType`, *optional*):
            The original type of the state dict, if not provided, the method will try to infer it automatically.
        kwargs (`dict`, *args*):
            Additional arguments to pass to the method.

            - **adapter_name**: For example, in case of PEFT, some keys will be pre-pended
                with the adapter name, therefore needs a special handling. By default PEFT also takes care of that in
                `get_peft_model_state_dict` method:
                https://github.com/huggingface/peft/blob/ba0477f2985b1ba311b83459d29895c809404e99/src/peft/utils/save_and_load.py#L92
                but we add it here in case we don't want to rely on that method.
    """

    peft_adapter_name = kwargs.pop("adapter_name", None)
    if peft_adapter_name is not None:
        peft_adapter_name = "." + peft_adapter_name
    else:
        peft_adapter_name = ""

    if original_type is None:
        if any(f".lora_A{peft_adapter_name}.weight" in k for k in state_dict.keys()):
            original_type = StateDictType.PEFT

    if original_type not in KOHYA_STATE_DICT_MAPPINGS.keys():
        raise ValueError(f"Original type {original_type} is not supported")

    # Use the convert_state_dict function with the appropriate mapping
    kohya_ss_partial_state_dict = convert_state_dict(state_dict, KOHYA_STATE_DICT_MAPPINGS[StateDictType.PEFT])
    kohya_ss_state_dict = {}

    # Additional logic for replacing header, alpha parameters `.` with `_` in all keys
    for kohya_key, weight in kohya_ss_partial_state_dict.items():
        if "text_encoder_2." in kohya_key:
            kohya_key = kohya_key.replace("text_encoder_2.", "lora_te2.")
        elif "text_encoder." in kohya_key:
            kohya_key = kohya_key.replace("text_encoder.", "lora_te1.")
        elif "unet" in kohya_key:
            kohya_key = kohya_key.replace("unet", "lora_unet")
        kohya_key = kohya_key.replace(".", "_", kohya_key.count(".") - 2)
        kohya_key = kohya_key.replace(peft_adapter_name, "")  # Kohya doesn't take names
        kohya_ss_state_dict[kohya_key] = weight
        if "lora_down" in kohya_key:
            alpha_key = f'{kohya_key.split(".")[0]}.alpha'
            kohya_ss_state_dict[alpha_key] = torch.tensor(len(weight))

    return kohya_ss_state_dict
