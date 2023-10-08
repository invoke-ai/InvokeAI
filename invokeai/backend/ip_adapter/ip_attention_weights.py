import torch


class IPAttentionProcessorWeights(torch.nn.Module):
    """The IP-Adapter weights for a single attention processor.

    This class is a torch.nn.Module sub-class to facilitate loading from a state_dict. It does not have a forward(...)
    method.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.to_k_ip = torch.nn.Linear(in_dim, out_dim, bias=False)
        self.to_v_ip = torch.nn.Linear(in_dim, out_dim, bias=False)


class IPAttentionWeights(torch.nn.Module):
    """A collection of all the `IPAttentionProcessorWeights` objects for an IP-Adapter model.

    This class is a torch.nn.Module sub-class so that it inherits the `.to(...)` functionality. It does not have a
    forward(...) method.
    """

    def __init__(self, weights: torch.nn.ModuleDict):
        super().__init__()
        self._weights = weights

    def get_attention_processor_weights(self, idx: int) -> IPAttentionProcessorWeights:
        """Get the `IPAttentionProcessorWeights` for the idx'th attention processor."""
        # Cast to int first, because we expect the key to represent an int. Then cast back to str, because
        # `torch.nn.ModuleDict` only supports str keys.
        return self._weights[str(int(idx))]

    @classmethod
    def from_state_dict(cls, state_dict: dict[str, torch.Tensor]):
        attn_proc_weights: dict[str, IPAttentionProcessorWeights] = {}

        for tensor_name, tensor in state_dict.items():
            if "to_k_ip.weight" in tensor_name:
                index = str(int(tensor_name.split(".")[0]))
                attn_proc_weights[index] = IPAttentionProcessorWeights(tensor.shape[1], tensor.shape[0])

        attn_proc_weights_module = torch.nn.ModuleDict(attn_proc_weights)
        attn_proc_weights_module.load_state_dict(state_dict)

        return cls(attn_proc_weights_module)
