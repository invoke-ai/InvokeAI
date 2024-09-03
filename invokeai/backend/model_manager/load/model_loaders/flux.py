# Copyright (c) 2024, Brandon W. Rising and the InvokeAI Development Team
"""Class for Flux model loading in InvokeAI."""

from pathlib import Path
from typing import Optional

import accelerate
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForTextEncoding, CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer

from invokeai.app.services.config.config_default import get_config
from invokeai.backend.flux.model import Flux
from invokeai.backend.flux.modules.autoencoder import AutoEncoder
from invokeai.backend.flux.util import ae_params, params
from invokeai.backend.model_manager import (
    AnyModel,
    AnyModelConfig,
    BaseModelType,
    ModelFormat,
    ModelType,
    SubModelType,
)
from invokeai.backend.model_manager.config import (
    CheckpointConfigBase,
    CLIPEmbedDiffusersConfig,
    MainBnbQuantized4bCheckpointConfig,
    MainCheckpointConfig,
    T5EncoderBnbQuantizedLlmInt8bConfig,
    T5EncoderConfig,
    VAECheckpointConfig,
)
from invokeai.backend.model_manager.load.load_default import ModelLoader
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.util.model_util import convert_bundle_to_flux_transformer_checkpoint
from invokeai.backend.util.silence_warnings import SilenceWarnings

try:
    from invokeai.backend.quantization.bnb_llm_int8 import quantize_model_llm_int8
    from invokeai.backend.quantization.bnb_nf4 import quantize_model_nf4

    bnb_available = True
except ImportError:
    bnb_available = False

app_config = get_config()


@ModelLoaderRegistry.register(base=BaseModelType.Flux, type=ModelType.VAE, format=ModelFormat.Checkpoint)
class FluxVAELoader(ModelLoader):
    """Class to load VAE models."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, VAECheckpointConfig):
            raise ValueError("Only VAECheckpointConfig models are currently supported here.")
        model_path = Path(config.path)

        with SilenceWarnings():
            model = AutoEncoder(ae_params[config.config_path])
            sd = load_file(model_path)
            model.load_state_dict(sd, assign=True)
            model.to(dtype=self._torch_dtype)

        return model


@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.CLIPEmbed, format=ModelFormat.Diffusers)
class ClipCheckpointModel(ModelLoader):
    """Class to load main models."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, CLIPEmbedDiffusersConfig):
            raise ValueError("Only CLIPEmbedDiffusersConfig models are currently supported here.")

        match submodel_type:
            case SubModelType.Tokenizer:
                return CLIPTokenizer.from_pretrained(Path(config.path) / "tokenizer")
            case SubModelType.TextEncoder:
                return CLIPTextModel.from_pretrained(Path(config.path) / "text_encoder")

        raise ValueError(
            f"Only Tokenizer and TextEncoder submodels are currently supported. Received: {submodel_type.value if submodel_type else 'None'}"
        )


@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.T5Encoder, format=ModelFormat.BnbQuantizedLlmInt8b)
class BnbQuantizedLlmInt8bCheckpointModel(ModelLoader):
    """Class to load main models."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, T5EncoderBnbQuantizedLlmInt8bConfig):
            raise ValueError("Only T5EncoderBnbQuantizedLlmInt8bConfig models are currently supported here.")
        if not bnb_available:
            raise ImportError(
                "The bnb modules are not available. Please install bitsandbytes if available on your platform."
            )
        match submodel_type:
            case SubModelType.Tokenizer2:
                return T5Tokenizer.from_pretrained(Path(config.path) / "tokenizer_2", max_length=512)
            case SubModelType.TextEncoder2:
                te2_model_path = Path(config.path) / "text_encoder_2"
                model_config = AutoConfig.from_pretrained(te2_model_path)
                with accelerate.init_empty_weights():
                    model = AutoModelForTextEncoding.from_config(model_config)
                    model = quantize_model_llm_int8(model, modules_to_not_convert=set())

                state_dict_path = te2_model_path / "bnb_llm_int8_model.safetensors"
                state_dict = load_file(state_dict_path)
                self._load_state_dict_into_t5(model, state_dict)

                return model

        raise ValueError(
            f"Only Tokenizer and TextEncoder submodels are currently supported. Received: {submodel_type.value if submodel_type else 'None'}"
        )

    @classmethod
    def _load_state_dict_into_t5(cls, model: T5EncoderModel, state_dict: dict[str, torch.Tensor]):
        # There is a shared reference to a single weight tensor in the model.
        # Both "encoder.embed_tokens.weight" and "shared.weight" refer to the same tensor, so only the latter should
        # be present in the state_dict.
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False, assign=True)
        assert len(unexpected_keys) == 0
        assert set(missing_keys) == {"encoder.embed_tokens.weight"}
        # Assert that the layers we expect to be shared are actually shared.
        assert model.encoder.embed_tokens.weight is model.shared.weight


@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.T5Encoder, format=ModelFormat.T5Encoder)
class T5EncoderCheckpointModel(ModelLoader):
    """Class to load main models."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, T5EncoderConfig):
            raise ValueError("Only T5EncoderConfig models are currently supported here.")

        match submodel_type:
            case SubModelType.Tokenizer2:
                return T5Tokenizer.from_pretrained(Path(config.path) / "tokenizer_2", max_length=512)
            case SubModelType.TextEncoder2:
                return T5EncoderModel.from_pretrained(Path(config.path) / "text_encoder_2")

        raise ValueError(
            f"Only Tokenizer and TextEncoder submodels are currently supported. Received: {submodel_type.value if submodel_type else 'None'}"
        )


@ModelLoaderRegistry.register(base=BaseModelType.Flux, type=ModelType.Main, format=ModelFormat.Checkpoint)
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

        with SilenceWarnings():
            model = Flux(params[config.config_path])
            sd = load_file(model_path)
            if "model.diffusion_model.double_blocks.0.img_attn.norm.key_norm.scale" in sd:
                sd = convert_bundle_to_flux_transformer_checkpoint(sd)
            model.load_state_dict(sd, assign=True)
        return model


@ModelLoaderRegistry.register(base=BaseModelType.Flux, type=ModelType.Main, format=ModelFormat.BnbQuantizednf4b)
class FluxBnbQuantizednf4bCheckpointModel(ModelLoader):
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
        assert isinstance(config, MainBnbQuantized4bCheckpointConfig)
        if not bnb_available:
            raise ImportError(
                "The bnb modules are not available. Please install bitsandbytes if available on your platform."
            )
        model_path = Path(config.path)

        with SilenceWarnings():
            with accelerate.init_empty_weights():
                model = Flux(params[config.config_path])
                model = quantize_model_nf4(model, modules_to_not_convert=set(), compute_dtype=torch.bfloat16)
            sd = load_file(model_path)
            if "model.diffusion_model.double_blocks.0.img_attn.norm.key_norm.scale" in sd:
                sd = convert_bundle_to_flux_transformer_checkpoint(sd)
            model.load_state_dict(sd, assign=True)
        return model
