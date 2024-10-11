from dataclasses import dataclass

import torch

from invokeai.backend.ip_adapter.ip_adapter import ImageProjModel


class IPDoubleStreamBlock(torch.nn.Module):
    def __init__(self, context_dim: int, hidden_dim: int):
        super().__init__()

        self.context_dim = context_dim
        self.hidden_dim = hidden_dim

        self.ip_adapter_double_stream_k_proj = torch.nn.Linear(context_dim, hidden_dim, bias=True)
        self.ip_adapter_double_stream_v_proj = torch.nn.Linear(context_dim, hidden_dim, bias=True)


class XlabsIpAdapterFlux:
    def __init__(self, image_proj: ImageProjModel, double_blocks: list[IPDoubleStreamBlock]):
        self.image_proj = image_proj
        self.double_blocks = double_blocks

    @classmethod
    def from_state_dict(cls, state_dict: dict[str, torch.Tensor]) -> "XlabsIpAdapterFlux":
        # TODO

        return cls()


@dataclass
class XlabsIpAdapterParams:
    num_double_blocks: int
    context_dim: int
    hidden_dim: int

    clip_embeddings_dim: int


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
