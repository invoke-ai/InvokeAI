import pytest
import torch

from invokeai.backend.ip_adapter.unet_patcher import apply_ip_adapter_attention
from invokeai.backend.model_management.models.base import BaseModelType, ModelType, SubModelType
from invokeai.backend.util.test_utils import install_and_load_model


def build_dummy_sd15_unet_input(torch_device):
    batch_size = 1
    num_channels = 4
    sizes = (32, 32)

    noise = torch.randn((batch_size, num_channels) + sizes).to(torch_device)
    time_step = torch.tensor([10]).to(torch_device)
    encoder_hidden_states = torch.randn((batch_size, 77, 768)).to(torch_device)

    return {"sample": noise, "timestep": time_step, "encoder_hidden_states": encoder_hidden_states}


@pytest.mark.parametrize(
    "model_params",
    [
        # SD1.5, IPAdapter
        {
            "ip_adapter_model_id": "InvokeAI/ip_adapter_sd15",
            "ip_adapter_model_name": "ip_adapter_sd15",
            "base_model": BaseModelType.StableDiffusion1,
            "unet_model_id": "runwayml/stable-diffusion-v1-5",
            "unet_model_name": "stable-diffusion-v1-5",
        },
        # SD1.5, IPAdapterPlus
        {
            "ip_adapter_model_id": "InvokeAI/ip_adapter_plus_sd15",
            "ip_adapter_model_name": "ip_adapter_plus_sd15",
            "base_model": BaseModelType.StableDiffusion1,
            "unet_model_id": "runwayml/stable-diffusion-v1-5",
            "unet_model_name": "stable-diffusion-v1-5",
        },
    ],
)
@pytest.mark.slow
def test_ip_adapter_unet_patch(model_params, model_installer, torch_device):
    """Smoke test that IP-Adapter weights can be loaded and used to patch a UNet."""
    ip_adapter_info = install_and_load_model(
        model_installer=model_installer,
        model_path_id_or_url=model_params["ip_adapter_model_id"],
        model_name=model_params["ip_adapter_model_name"],
        base_model=model_params["base_model"],
        model_type=ModelType.IPAdapter,
    )

    unet_info = install_and_load_model(
        model_installer=model_installer,
        model_path_id_or_url=model_params["unet_model_id"],
        model_name=model_params["unet_model_name"],
        base_model=model_params["base_model"],
        model_type=ModelType.Main,
        submodel_type=SubModelType.UNet,
    )

    dummy_unet_input = build_dummy_sd15_unet_input(torch_device)

    with torch.no_grad(), ip_adapter_info as ip_adapter, unet_info as unet:
        ip_adapter.to(torch_device, dtype=torch.float32)
        unet.to(torch_device, dtype=torch.float32)

        cross_attention_kwargs = {"ip_adapter_image_prompt_embeds": [torch.randn((1, 4, 768)).to(torch_device)]}
        with apply_ip_adapter_attention(unet, [ip_adapter]):
            output = unet(**dummy_unet_input, cross_attention_kwargs=cross_attention_kwargs).sample

    assert output.shape == dummy_unet_input["sample"].shape
