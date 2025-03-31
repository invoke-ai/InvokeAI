from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from invokeai.backend.model_manager.legacy_probe import CkptType


def get_flux_in_channels_from_state_dict(state_dict: "CkptType") -> int | None:
    """Gets the in channels from the state dict."""

    # "Standard" FLUX models use "img_in.weight", but some community fine tunes use
    # "model.diffusion_model.img_in.weight". Known models that use the latter key:
    # - https://civitai.com/models/885098?modelVersionId=990775
    # - https://civitai.com/models/1018060?modelVersionId=1596255
    # - https://civitai.com/models/978314/ultrareal-fine-tune?modelVersionId=1413133

    keys = {"img_in.weight", "model.diffusion_model.img_in.weight"}

    for key in keys:
        val = state_dict.get(key)
        if val is not None:
            return val.shape[1]

    return None
