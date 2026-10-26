"""Loader for Tiny AutoEncoder preview decoders (madebyollin's TAESD family)."""

from pathlib import Path
from typing import Any, Optional

import torch
from safetensors.torch import load_file as safetensors_load_file

from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.configs.tiny_autoencoder import TinyAutoencoder_Checkpoint_Config_Base
from invokeai.backend.model_manager.load.load_default import ModelLoader
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.taxonomy import AnyModel, BaseModelType, ModelFormat, ModelType, SubModelType
from invokeai.backend.tiny_autoencoder.checkpoint import (
    PICKLE_SUFFIXES,
    SAFETENSORS_SUFFIX,
    build_taesd_decoder,
    detect_taesd_layout,
    taesd_decoder_state_dict,
)


def _read_checkpoint(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == SAFETENSORS_SUFFIX:
        return safetensors_load_file(path)
    if path.suffix.lower() in PICKLE_SUFFIXES:
        return torch.load(path, map_location="cpu", weights_only=True)
    raise ValueError(f"Unrecognised Tiny AutoEncoder checkpoint extension: {path.suffix!r}")


@ModelLoaderRegistry.register(
    base=BaseModelType.StableDiffusion1, type=ModelType.TinyAutoencoder, format=ModelFormat.Checkpoint
)
@ModelLoaderRegistry.register(
    base=BaseModelType.StableDiffusionXL, type=ModelType.TinyAutoencoder, format=ModelFormat.Checkpoint
)
@ModelLoaderRegistry.register(
    base=BaseModelType.StableDiffusion3, type=ModelType.TinyAutoencoder, format=ModelFormat.Checkpoint
)
@ModelLoaderRegistry.register(base=BaseModelType.Flux, type=ModelType.TinyAutoencoder, format=ModelFormat.Checkpoint)
@ModelLoaderRegistry.register(base=BaseModelType.Flux2, type=ModelType.TinyAutoencoder, format=ModelFormat.Checkpoint)
@ModelLoaderRegistry.register(
    base=BaseModelType.QwenImage21, type=ModelType.TinyAutoencoder, format=ModelFormat.Checkpoint
)
class TinyAutoencoderLoader(ModelLoader):
    """Builds the TAESD decoder a checkpoint holds and loads it strictly, in either published layout."""

    def _load_model(self, config: AnyModelConfig, submodel_type: Optional[SubModelType] = None) -> AnyModel:
        if submodel_type is not None:
            raise ValueError("Unexpected submodel requested for a Tiny AutoEncoder.")
        if not isinstance(config, TinyAutoencoder_Checkpoint_Config_Base):
            raise ValueError(f"Unexpected model config type: {type(config)}.")

        state_dict = taesd_decoder_state_dict(_read_checkpoint(Path(config.path)))
        layout = detect_taesd_layout(state_dict)
        if layout is None:
            raise ValueError(f"'{config.name}' is not a TAESD decoder checkpoint.")
        decoder = build_taesd_decoder(layout)
        decoder.load_state_dict(state_dict, strict=True)
        return decoder.to(self._torch_dtype).eval().requires_grad_(False)
