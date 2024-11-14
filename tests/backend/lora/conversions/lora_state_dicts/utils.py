import torch


def keys_to_mock_state_dict(keys: dict[str, list[int]]) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {}
    for k, shape in keys.items():
        state_dict[k] = torch.empty(shape)
    return state_dict
