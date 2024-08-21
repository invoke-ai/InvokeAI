# Copyright (c) 2024, Brandon W. Rising and the InvokeAI Development Team
"""Class for Flux model loading in InvokeAI."""

from dataclasses import fields
from pathlib import Path
from typing import Any, Optional

import accelerate
import torch
import yaml
from safetensors.torch import load_file
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer

from invokeai.app.services.config.config_default import get_config
from invokeai.backend.flux.model import Flux, FluxParams
from invokeai.backend.flux.modules.autoencoder import AutoEncoder, AutoEncoderParams
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
    T5Encoder8bConfig,
    T5EncoderConfig,
    VAECheckpointConfig,
)
from invokeai.backend.model_manager.load.load_default import ModelLoader
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.quantization.bnb_nf4 import quantize_model_nf4
from invokeai.backend.quantization.fast_quantized_transformers_model import FastQuantizedTransformersModel
from invokeai.backend.util.silence_warnings import SilenceWarnings

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
        legacy_config_path = app_config.legacy_conf_path / config.config_path
        config_path = legacy_config_path.as_posix()
        with open(config_path, "r") as stream:
            flux_conf = yaml.safe_load(stream)

        dataclass_fields = {f.name for f in fields(AutoEncoderParams)}
        filtered_data = {k: v for k, v in flux_conf["params"].items() if k in dataclass_fields}
        params = AutoEncoderParams(**filtered_data)

        with SilenceWarnings():
            model = AutoEncoder(params)
            sd = load_file(model_path)
            model.load_state_dict(sd, assign=True)

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
                return CLIPTokenizer.from_pretrained(config.path)
            case SubModelType.TextEncoder:
                return CLIPTextModel.from_pretrained(config.path)

        raise ValueError(f"Only Tokenizer and TextEncoder submodels are currently supported. Received: {submodel_type.value if submodel_type else 'None'}")


@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.T5Encoder, format=ModelFormat.T5Encoder8b)
class T5Encoder8bCheckpointModel(ModelLoader):
    """Class to load main models."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, T5Encoder8bConfig):
            raise ValueError("Only T5Encoder8bConfig models are currently supported here.")

        match submodel_type:
            case SubModelType.Tokenizer2:
                return T5Tokenizer.from_pretrained(Path(config.path) / "tokenizer_2", max_length=512)
            case SubModelType.TextEncoder2:
                return FastQuantizedTransformersModel.from_pretrained(Path(config.path) / "text_encoder_2")

        raise ValueError(f"Only Tokenizer and TextEncoder submodels are currently supported. Received: {submodel_type.value if submodel_type else 'None'}")


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
                return T5EncoderModel.from_pretrained(
                    Path(config.path) / "text_encoder_2"
                )  # TODO: Fix hf subfolder install

        raise ValueError(f"Only Tokenizer and TextEncoder submodels are currently supported. Received: {submodel_type.value if submodel_type else 'None'}")


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
        legacy_config_path = app_config.legacy_conf_path / config.config_path
        config_path = legacy_config_path.as_posix()
        with open(config_path, "r") as stream:
            flux_conf = yaml.safe_load(stream)

        match submodel_type:
            case SubModelType.Transformer:
                return self._load_from_singlefile(config, flux_conf)

        raise ValueError(f"Only Transformer submodels are currently supported. Received: {submodel_type.value if submodel_type else 'None'}")

    def _load_from_singlefile(
        self,
        config: AnyModelConfig,
        flux_conf: Any,
    ) -> AnyModel:
        assert isinstance(config, MainCheckpointConfig)
        model_path = Path(config.path)
        dataclass_fields = {f.name for f in fields(FluxParams)}
        filtered_data = {k: v for k, v in flux_conf["params"].items() if k in dataclass_fields}
        params = FluxParams(**filtered_data)

        with SilenceWarnings():
            model = Flux(params)
            sd = load_file(model_path)
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
        legacy_config_path = app_config.legacy_conf_path / config.config_path
        config_path = legacy_config_path.as_posix()
        with open(config_path, "r") as stream:
            flux_conf = yaml.safe_load(stream)

        match submodel_type:
            case SubModelType.Transformer:
                return self._load_from_singlefile(config, flux_conf)

        raise ValueError(f"Only Transformer submodels are currently supported. Received: {submodel_type.value if submodel_type else 'None'}")

    def _load_from_singlefile(
        self,
        config: AnyModelConfig,
        flux_conf: Any,
    ) -> AnyModel:
        assert isinstance(config, MainBnbQuantized4bCheckpointConfig)
        model_path = Path(config.path)
        dataclass_fields = {f.name for f in fields(FluxParams)}
        filtered_data = {k: v for k, v in flux_conf["params"].items() if k in dataclass_fields}
        params = FluxParams(**filtered_data)

        with SilenceWarnings():
            with accelerate.init_empty_weights():
                model = Flux(params)
                model = quantize_model_nf4(model, modules_to_not_convert=set(), compute_dtype=torch.bfloat16)
            sd = load_file(model_path)
            model.load_state_dict(sd, assign=True)
        return model
