from dataclasses import dataclass

import accelerate
import torch

from invokeai.backend.ip_adapter.ip_adapter import ImageProjModel


class IPDoubleStreamBlock(torch.nn.Module):
    def __init__(self, context_dim: int, hidden_dim: int):
        super().__init__()

        self.context_dim = context_dim
        self.hidden_dim = hidden_dim

        self.ip_adapter_double_stream_k_proj = torch.nn.Linear(context_dim, hidden_dim, bias=True)
        self.ip_adapter_double_stream_v_proj = torch.nn.Linear(context_dim, hidden_dim, bias=True)


@dataclass
class XlabsIpAdapterParams:
    num_double_blocks: int
    context_dim: int
    hidden_dim: int

    clip_embeddings_dim: int


class XlabsIpAdapterFlux(torch.nn.Module):
    def __init__(self, params: XlabsIpAdapterParams):
        super().__init__()
        self.image_proj = ImageProjModel(
            cross_attention_dim=params.context_dim, clip_embeddings_dim=params.clip_embeddings_dim
        )
        self.double_blocks = torch.nn.ModuleList(
            [IPDoubleStreamBlock(params.context_dim, params.hidden_dim) for _ in range(params.num_double_blocks)]
        )

    def load_xlabs_state_dict(self, state_dict: dict[str, torch.Tensor], assign: bool = False):
        """We need this custom function to load state dicts rather than using .load_state_dict(...) because the model
        structure does not match the state_dict structure.
        """
        # Split the state_dict into the image projection model and the double blocks.
        image_proj_sd: dict[str, torch.Tensor] = {}
        double_blocks_sd: dict[str, torch.Tensor] = {}
        for k, v in state_dict.items():
            if k.startswith("ip_adapter_proj_model."):
                image_proj_sd[k] = v
            elif k.startswith("double_blocks."):
                double_blocks_sd[k] = v
            else:
                raise ValueError(f"Unexpected key: {k}")

        # Initialize the image projection model.
        image_proj_sd = {k.replace("ip_adapter_proj_model.", ""): v for k, v in image_proj_sd.items()}
        self.image_proj.load_state_dict(image_proj_sd, assign=assign)

        # Initialize the double blocks.
        for i, double_block in enumerate(self.double_blocks):
            double_block_sd: dict[str, torch.Tensor] = {
                "ip_adapter_double_stream_k_proj.bias": double_blocks_sd[
                    f"double_blocks.{i}.processor.ip_adapter_double_stream_k_proj.bias"
                ],
                "ip_adapter_double_stream_k_proj.weight": double_blocks_sd[
                    f"double_blocks.{i}.processor.ip_adapter_double_stream_k_proj.weight"
                ],
                "ip_adapter_double_stream_v_proj.bias": double_blocks_sd[
                    f"double_blocks.{i}.processor.ip_adapter_double_stream_v_proj.bias"
                ],
                "ip_adapter_double_stream_v_proj.weight": double_blocks_sd[
                    f"double_blocks.{i}.processor.ip_adapter_double_stream_v_proj.weight"
                ],
            }
            double_block.load_state_dict(double_block_sd, assign=assign)


def infer_xlabs_ip_adapter_params_from_state_dict(state_dict: dict[str, torch.Tensor]) -> XlabsIpAdapterParams:
    num_double_blocks = 0
    context_dim = 0
    hidden_dim = 0

    # Count the number of double blocks.
    double_block_index = 0
    while f"double_blocks.{double_block_index}.processor.ip_adapter_double_stream_k_proj.weight" in state_dict:
        double_block_index += 1
    num_double_blocks = double_block_index

    hidden_dim = state_dict["double_blocks.0.processor.ip_adapter_double_stream_k_proj.weight"].shape[0]
    context_dim = state_dict["double_blocks.0.processor.ip_adapter_double_stream_k_proj.weight"].shape[1]
    clip_embeddings_dim = state_dict["ip_adapter_proj_model.proj.weight"].shape[1]

    return XlabsIpAdapterParams(
        num_double_blocks=num_double_blocks,
        context_dim=context_dim,
        hidden_dim=hidden_dim,
        clip_embeddings_dim=clip_embeddings_dim,
    )


def load_xlabs_ip_adapter_flux(state_dict: dict[str, torch.Tensor]) -> XlabsIpAdapterFlux:
    params = infer_xlabs_ip_adapter_params_from_state_dict(state_dict)

    with accelerate.init_empty_weights():
        model = XlabsIpAdapterFlux(params=params)

    model.load_xlabs_state_dict(state_dict)
    return model
