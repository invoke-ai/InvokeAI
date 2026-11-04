"""Loader for SD3 transformers in GGUF files. SD3 Diffusers pipelines load through `stable_diffusion.py`."""

from pathlib import Path
from typing import Optional

import accelerate

from invokeai.backend.model_manager.checkpoint_prefix import CheckpointPrefix
from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.configs.main import Main_GGUF_SD3_Config
from invokeai.backend.model_manager.load.load_default import ModelLoader
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.taxonomy import AnyModel, BaseModelType, ModelFormat, ModelType, SubModelType
from invokeai.backend.quantization.gguf.loaders import gguf_sd_loader, unpack_ggml_at_load
from invokeai.backend.sd3.checkpoint_layout import (
    convert_mmdit_to_diffusers,
    is_mmdit_transformer_key,
    sd3_transformer_config,
)
from invokeai.backend.util.state_dict_loading import reject_incomplete_load


@ModelLoaderRegistry.register(
    base=BaseModelType.StableDiffusion3, type=ModelType.Main, format=ModelFormat.GGUFQuantized
)
class SD3GGUFModel(ModelLoader):
    """Loads an SD3 transformer from a GGUF file in Stability's MMDiT layout (see `backend/sd3/checkpoint_layout.py`).

    The Linear weights stay packed and dequantize per forward, in the dtype the SD3 denoise runs at; what the
    model reads outside a matmul and what the file stores unquantized is unpacked at load. Its text encoders and
    VAE are separate models, chosen on the SD3 model loader node.
    """

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, Main_GGUF_SD3_Config):
            raise TypeError(f"Expected Main_GGUF_SD3_Config, got {type(config).__name__}.")
        if submodel_type is not SubModelType.Transformer:
            raise ValueError(
                f"An SD3 GGUF holds only the transformer, not {submodel_type.value if submodel_type else 'None'}. "
                "Select standalone T5, CLIP-L, CLIP-G and VAE models on the SD3 model loader."
            )

        from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel

        # A bundled VAE or text encoder is never read: only the transformer is this model's.
        sd = gguf_sd_loader(Path(config.path), compute_dtype=self._torch_dtype, keep=is_mmdit_transformer_key)
        sd = CheckpointPrefix.detect(sd).strip(sd, drop_foreign=True)
        transformer_config = sd3_transformer_config(sd)
        sd = convert_mmdit_to_diffusers(sd, transformer_config)

        # Buffers too: `pos_embed` would otherwise be computed in float32 (906 MB for Medium) only to be replaced.
        with accelerate.init_empty_weights(include_buffers=True):
            model = SD3Transformer2DModel(**transformer_config)

        # No skip patterns: diffusers' own (`pos_embed`, `norm`) match every `norm1.linear` and `norm1_context.linear`
        # modulation Linear, 2.7B of Large's 8B parameters -- 5.3 GB at 16 bits instead of 1.5 GB as Q4_0 -- though
        # they run packed like any other Linear.
        unpacked = unpack_ggml_at_load(sd, model, (), self._ram_cache.make_room)

        model.load_state_dict(sd, strict=True, assign=True)
        sd.clear()
        # Strict loading covers parameters and persistent buffers; this covers a non-persistent buffer left on meta.
        reject_incomplete_load(model, what=f"SD3 GGUF transformer {config.name}")

        self._logger.info(
            f"SD3: loaded a GGUF transformer, {unpacked} unquantized or non-Linear tensor(s) dequantized at load, "
            "the remaining Linear weights kept packed"
        )
        return model
