import torch


def keys_to_mock_state_dict(keys: list[str]) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {}
    for k in keys:
        state_dict[k] = torch.empty(1)
    return state_dict
