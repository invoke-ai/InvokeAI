from pathlib import Path
from typing import Any, Optional, Union

import torch
from safetensors.torch import load_file


def state_dict_to(
    state_dict: dict[str, torch.Tensor], device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
) -> dict[str, torch.Tensor]:
    new_state_dict: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        new_state_dict[k] = v.to(device=device, dtype=dtype, non_blocking=True)
    return new_state_dict


def load_state_dict(file_path: Union[str, Path], device: str = "cpu") -> Any:
    """Load a state_dict from a file that may be in either PyTorch or safetensors format. The file format is inferred
    from the file extension.
    """
    file_path = Path(file_path)

    if file_path.suffix == ".safetensors":
        state_dict = load_file(
            file_path,
            device=device,
        )
    else:
        # weights_only=True is used to address a security vulnerability that allows arbitrary code execution.
        # This option was first introduced in https://github.com/pytorch/pytorch/pull/86812.
        #
        # mmap=True is used to both reduce memory usage and speed up loading. This setting causes torch.load() to more
        # closely mirror the behaviour of safetensors.torch.load_file().  This option was first introduced in
        # https://github.com/pytorch/pytorch/pull/102549. The discussion on that PR provides helpful context.
        state_dict = torch.load(file_path, map_location=device, weights_only=True, mmap=True)

    return state_dict
