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


class IPAdapterDoubleBlocks(torch.nn.Module):
    def __init__(self, num_double_blocks: int, context_dim: int, hidden_dim: int):
        super().__init__()
        self.double_blocks = torch.nn.ModuleList(
            [IPDoubleStreamBlock(context_dim, hidden_dim) for _ in range(num_double_blocks)]
        )


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
        self.ip_adapter_double_blocks = IPAdapterDoubleBlocks(
            num_double_blocks=params.num_double_blocks, context_dim=params.context_dim, hidden_dim=params.hidden_dim
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
        double_blocks_sd = {k.replace("processor.", ""): v for k, v in double_blocks_sd.items()}
        self.ip_adapter_double_blocks.load_state_dict(double_blocks_sd, assign=assign)
