from pathlib import Path
from typing import Optional

from invokeai.backend.model_manager.config import (
    AnyModel,
    AnyModelConfig,
    BaseModelType,
    CheckpointConfigBase,
    MainCheckpointConfig,
    ModelFormat,
    ModelType,
    SubModelType,
)
from invokeai.backend.model_manager.load.load_default import ModelLoader
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry


@ModelLoaderRegistry.register(base=BaseModelType.StableDiffusion35, type=ModelType.Main, format=ModelFormat.Checkpoint)
class FluxCheckpointModel(ModelLoader):
    """Class to load main models."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, CheckpointConfigBase):
            raise ValueError("Only CheckpointConfigBase models are currently supported here.")

        match submodel_type:
            case SubModelType.Transformer:
                return self._load_from_singlefile(config)

        raise ValueError(
            f"Only Transformer submodels are currently supported. Received: {submodel_type.value if submodel_type else 'None'}"
        )

    def _load_from_singlefile(
        self,
        config: AnyModelConfig,
    ) -> AnyModel:
        assert isinstance(config, MainCheckpointConfig)
        model_path = Path(config.path)

        # model = Flux(params[config.config_path])
        # sd = load_file(model_path)
        # if "model.diffusion_model.double_blocks.0.img_attn.norm.key_norm.scale" in sd:
        #     sd = convert_bundle_to_flux_transformer_checkpoint(sd)
        # new_sd_size = sum([ten.nelement() * torch.bfloat16.itemsize for ten in sd.values()])
        # self._ram_cache.make_room(new_sd_size)
        # for k in sd.keys():
        #     # We need to cast to bfloat16 due to it being the only currently supported dtype for inference
        #     sd[k] = sd[k].to(torch.bfloat16)
        # model.load_state_dict(sd, assign=True)
        return model
