import torch


class IPAttentionProcessorWeights(torch.nn.Module):
    """The IP-Adapter weights for a single attention processor.

    This class is a torch.nn.Module sub-class to facilitate loading from a state_dict. It does not have a forward(...)
    method.
    """

    def __init__(self, in_dim: int, out_dim: int, scale: float = 1.0):
        super().__init__()
        self.scale = scale
        self.to_k_ip = torch.nn.Linear(in_dim, out_dim, bias=False)
        self.to_v_ip = torch.nn.Linear(in_dim, out_dim, bias=False)


class IPAttentionWeights(torch.nn.Module):
    """A collection of all the `IPAttentionProcessorWeights` objects for an IP-Adapter model.

    This class is a torch.nn.Module sub-class so that it inherits the `.to(...)` functionality. It does not have a
    forward(...) method.
    """

    def __init__(self, weights: dict[int, IPAttentionProcessorWeights]):
        super().__init__()
        self.weights = weights

    def set_scale(self, scale: float):
        """Set the scale (a.k.a. 'weight') for all of the `IPAttentionProcessorWeights` in this collection."""
        for w in self.weights.values():
            w.scale = scale

    @classmethod
    def from_state_dict(cls, state_dict: dict[str, torch.Tensor]):
        attn_proc_weights: dict[int, IPAttentionProcessorWeights] = {}

        for tensor_name, tensor in state_dict.items():
            if "to_k_ip.weight" in tensor_name:
                index = int(tensor_name.split(".")[0])
                attn_proc_weights[index] = IPAttentionProcessorWeights(tensor.shape[1], tensor.shape[0])

        attn_proc_weights_module_dict = torch.nn.ModuleDict(attn_proc_weights)
        attn_proc_weights_module_dict.load_state_dict(state_dict)

        return cls(attn_proc_weights)
