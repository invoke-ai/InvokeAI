import torch

try:
    from bitsandbytes.nn.modules import Params4bit

    bnb_available: bool = True
except ImportError:
    bnb_available: bool = False


def get_param_shape(param: torch.Tensor) -> torch.Size:
    """A helper function to get the shape of a parameter that handles `bitsandbytes.nn.Params4Bit` correctly."""
    # Accessing the `.shape` attribute of `bitsandbytes.nn.Params4Bit` will return an incorrect result. Instead, we must
    # access the `.quant_state.shape` attribute.
    if bnb_available and type(param) is Params4bit:  # type: ignore
        quant_state = param.quant_state
        if quant_state is not None:
            return quant_state.shape
    return param.shape
