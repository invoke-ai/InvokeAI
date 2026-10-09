"""Loaders for Baidu ERNIE-Image: diffusers pipelines and single-file transformers (safetensors or GGUF)."""

from pathlib import Path
from typing import Optional

import accelerate
import safetensors.torch

from invokeai.backend.model_manager.configs.base import Diffusers_Config_Base
from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.configs.main import Main_Checkpoint_ErnieImage_Config, Main_GGUF_ErnieImage_Config
from invokeai.backend.model_manager.load.load_default import ModelLoader, _model_declared_skip_patterns
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.load.model_loaders.generic_diffusers import GenericDiffusersLoader
from invokeai.backend.model_manager.taxonomy import (
    AnyModel,
    BaseModelType,
    ModelFormat,
    ModelType,
    SubModelType,
)
from invokeai.backend.quantization.fp8_scaled import reject_quantized_side_channel
from invokeai.backend.quantization.gguf.loaders import gguf_sd_loader, unpack_ggml_at_load
from invokeai.backend.util.devices import TorchDevice

# The geometry of the released ERNIE-Image transformer, from the pipeline's `transformer/config.json`.
# A single file carries no config, and `ErnieImageTransformer2DModel`'s own defaults describe a
# smaller model (hidden 3072, 24 layers), so building from them would fail on every weight.
ERNIE_IMAGE_TRANSFORMER_CONFIG = {
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_layers": 36,
    "ffn_hidden_size": 12288,
    "in_channels": 128,
    "out_channels": 128,
    "patch_size": 1,
    "text_in_dim": 3072,
    "rope_theta": 256,
    "rope_axes_dim": (32, 48, 48),
    "eps": 1e-06,
    "qk_layernorm": True,
}


@ModelLoaderRegistry.register(base=BaseModelType.ErnieImage, type=ModelType.Main, format=ModelFormat.Diffusers)
class ErnieImageDiffusersModel(GenericDiffusersLoader):
    """Loads ERNIE-Image submodels (transformer, vae, text_encoder, tokenizer, pe, pe_tokenizer)
    from a diffusers pipeline directory.
    """

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if submodel_type is None:
            raise Exception("A submodel type must be provided when loading ERNIE-Image pipelines.")

        model_path = Path(config.path)
        load_class = self.get_hf_load_class(model_path, submodel_type)

        repo_variant = config.repo_variant if isinstance(config, Diffusers_Config_Base) else None
        variant = repo_variant.value if repo_variant else None

        # The SubModelType values match the ERNIE-Image pipeline's subdir names 1:1, including
        # the prompt enhancer ("pe" / "pe_tokenizer").
        model_path = model_path / submodel_type.value

        # Tokenizers take neither a dtype nor a repo variant — mirror the sibling loaders and load
        # them bare. `local_files_only=True` everywhere keeps loading offline-safe: the files are
        # already on disk, and without it transformers/diffusers may reach out to the Hub to
        # validate the repo.
        if submodel_type in (SubModelType.Tokenizer, SubModelType.PromptEnhancerTokenizer):
            result: AnyModel = load_class.from_pretrained(model_path, local_files_only=True)
            return result

        target_device = TorchDevice.choose_torch_device()
        dtype = TorchDevice.choose_bfloat16_safe_dtype(target_device)
        try:
            result = load_class.from_pretrained(model_path, torch_dtype=dtype, variant=variant, local_files_only=True)
        except OSError as e:
            if variant and "no file named" in str(e):
                result = load_class.from_pretrained(model_path, torch_dtype=dtype, local_files_only=True)
            else:
                raise

        return self._apply_fp8_layerwise_casting(result, config, submodel_type)


@ModelLoaderRegistry.register(base=BaseModelType.ErnieImage, type=ModelType.Main, format=ModelFormat.Checkpoint)
class ErnieImageCheckpointModel(ModelLoader):
    """Loads an ERNIE-Image transformer from a single safetensors file.

    The file holds the transformer alone, under the same keys as the diffusers checkpoint. Its text
    encoder, VAE and (optional) prompt enhancer are separate models, chosen on the loader node.
    """

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, Main_Checkpoint_ErnieImage_Config):
            raise ValueError(f"Expected Main_Checkpoint_ErnieImage_Config, got {type(config).__name__}.")
        if submodel_type is not None and submodel_type is not SubModelType.Transformer:
            raise ValueError(
                f"A single-file ERNIE-Image checkpoint holds only the transformer, not {submodel_type.value}. "
                "Select a standalone text encoder and VAE on the ERNIE-Image model loader."
            )

        from diffusers import ErnieImageTransformer2DModel

        target_device = TorchDevice.choose_torch_device()
        model_dtype = TorchDevice.choose_bfloat16_safe_dtype(target_device)

        sd = safetensors.torch.load_file(Path(config.path))
        reject_quantized_side_channel(sd, f"ERNIE-Image checkpoint {Path(config.path).name}")
        self._ram_cache.make_room(sum(t.nelement() * model_dtype.itemsize for t in sd.values()))
        for key in sd:
            sd[key] = sd[key].to(model_dtype)

        with accelerate.init_empty_weights():
            model = ErnieImageTransformer2DModel(**ERNIE_IMAGE_TRANSFORMER_CONFIG)
        model.load_state_dict(sd, assign=True)
        # `assign=True` aliases every parameter to its state dict tensor, so the dict would keep a
        # second reference to the whole model alive through the fp8 cast below.
        sd.clear()

        return self._apply_fp8_layerwise_casting(model, config, SubModelType.Transformer)


@ModelLoaderRegistry.register(base=BaseModelType.ErnieImage, type=ModelType.Main, format=ModelFormat.GGUFQuantized)
class ErnieImageGGUFModel(ModelLoader):
    """Loads an ERNIE-Image transformer from a GGUF file.

    The community GGUFs keep the diffusers key layout, so the weights load into
    ``ErnieImageTransformer2DModel`` as they are. The Linear weights stay packed and dequantize per
    forward; what the model reads outside a matmul (the norms, the patch convolution) and what is not
    actually quantized is unpacked at load (see `unpack_ggml_at_load`). Its text encoder and VAE are
    separate models, chosen on the loader node.

    No `fp8_storage` declaration: FP8 Storage is refused for every quantized format by rule.
    """

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, Main_GGUF_ErnieImage_Config):
            raise ValueError(f"Expected Main_GGUF_ErnieImage_Config, got {type(config).__name__}.")
        if submodel_type is not None and submodel_type is not SubModelType.Transformer:
            raise ValueError(
                f"A single-file ERNIE-Image GGUF holds only the transformer, not {submodel_type.value}. "
                "Select a standalone text encoder and VAE on the ERNIE-Image model loader."
            )

        from diffusers import ErnieImageTransformer2DModel

        target_device = TorchDevice.choose_torch_device()
        compute_dtype = TorchDevice.choose_bfloat16_safe_dtype(target_device)

        sd = gguf_sd_loader(Path(config.path), compute_dtype=compute_dtype)

        with accelerate.init_empty_weights():
            model = ErnieImageTransformer2DModel(**ERNIE_IMAGE_TRANSFORMER_CONFIG)

        unpacked = unpack_ggml_at_load(sd, model, _model_declared_skip_patterns(model), self._ram_cache.make_room)

        # Strict: every published GGUF is key-for-key the diffusers transformer.
        model.load_state_dict(sd, strict=True, assign=True)
        sd.clear()

        self._logger.info(
            f"ERNIE-Image: loaded a GGUF transformer, {unpacked} unquantized or non-Linear tensor(s) "
            "dequantized at load, the remaining Linear weights kept packed"
        )
        return model
