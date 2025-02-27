from typing import Any, Dict


def is_state_dict_likely_flux_redux(state_dict: Dict[str, Any]) -> bool:
    """Checks if the provided state dict is likely a FLUX Redux model."""

    expected_keys = {"redux_down.bias", "redux_down.weight", "redux_up.bias", "redux_up.weight"}
    if set(state_dict.keys()) == expected_keys:
        return True

    return False
