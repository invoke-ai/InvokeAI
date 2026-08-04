"""Loader registrations for MiniMax H3 (Hailuo 3.0) audio-video generation models.

Currently covers the diffusers-format Modular Diffusers layout only (the layout at the root of
the ``MiniMaxAI/MiniMax-H3`` HF repo). The model classes are vendored under
``invokeai.backend.minimax_h3`` because they only exist in an unreleased diffusers branch, so this
loader dispatches submodels explicitly instead of subclassing ``GenericDiffusersLoader`` (whose
``get_hf_load_class`` resolves class names against the installed diffusers and would fail).

Submodel map (subfolder == ``SubModelType`` value):
- ``transformer``  -> vendored ``MiniMaxH3Transformer3DModel`` (33B DiT, bf16)
- ``text_encoder`` -> ``transformers.Qwen3VLForConditionalGeneration`` (Qwen3-VL-32B, bf16; H3
  conditions on layer-50 hidden states and never uses the LM head, but the checkpoint is the full
  model per the repo's modular_model_index.json)
- ``tokenizer``    -> ``AutoTokenizer`` (Qwen2TokenizerFast)
- ``processor``    -> ``AutoProcessor`` (Qwen3VLProcessor; needed even for text-only encoding,
  which uses its multimodal token-type ids)
- ``vae``          -> vendored ``AutoencoderKLMiniMaxH3`` (video VAE, bf16)
- ``audio_vae``    -> vendored ``AutoencoderKLMiniMaxH3Audio`` (kept fp32: it is ~0.6 GB and
  half-precision artifacts in decoded audio are audible)

The two ``MiniMaxH3Scheduler`` instances (video shift 12.0, audio shift 3.0) are constructed
directly by the denoise invocation - they are stateless configs, not loaded weights.
"""

from pathlib import Path
from typing import Optional

import torch

from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.configs.main import Main_Diffusers_MiniMaxH3_Config
from invokeai.backend.model_manager.load.load_default import ModelLoader
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.taxonomy import (
    AnyModel,
    BaseModelType,
    ModelFormat,
    ModelType,
    SubModelType,
)
from invokeai.backend.model_manager.util.qwen3_vl import normalize_qwen3vl_rope_config
from invokeai.backend.util.devices import TorchDevice


@ModelLoaderRegistry.register(base=BaseModelType.MiniMaxH3, type=ModelType.Main, format=ModelFormat.Diffusers)
class MiniMaxH3DiffusersModel(ModelLoader):
    """Loader for MiniMax H3 diffusers-format models (FL2VA)."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, Main_Diffusers_MiniMaxH3_Config):
            raise ValueError(f"Unexpected config type {type(config).__name__} for a MiniMax H3 loader.")
        if submodel_type is None:
            raise Exception("A submodel type must be provided when loading MiniMax H3 main models.")

        model_path = Path(config.path)
        submodel_path = model_path / submodel_type.value

        target_device = TorchDevice.choose_torch_device()
        dtype = TorchDevice.choose_bfloat16_safe_dtype(target_device)

        match submodel_type:
            case SubModelType.Transformer:
                from invokeai.backend.minimax_h3 import MiniMaxH3Transformer3DModel

                return MiniMaxH3Transformer3DModel.from_pretrained(
                    submodel_path, torch_dtype=dtype, local_files_only=True
                )
            case SubModelType.TextEncoder:
                from transformers import AutoConfig, Qwen3VLForConditionalGeneration

                te_config = normalize_qwen3vl_rope_config(
                    AutoConfig.from_pretrained(submodel_path, local_files_only=True)
                )
                return Qwen3VLForConditionalGeneration.from_pretrained(
                    submodel_path,
                    config=te_config,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                    local_files_only=True,
                )
            case SubModelType.Tokenizer:
                from transformers import AutoTokenizer

                return AutoTokenizer.from_pretrained(submodel_path, local_files_only=True)
            case SubModelType.Processor:
                from transformers import AutoProcessor

                return AutoProcessor.from_pretrained(submodel_path, local_files_only=True)
            case SubModelType.VAE:
                from invokeai.backend.minimax_h3 import AutoencoderKLMiniMaxH3

                return AutoencoderKLMiniMaxH3.from_pretrained(submodel_path, torch_dtype=dtype, local_files_only=True)
            case SubModelType.AudioVAE:
                from invokeai.backend.minimax_h3 import AutoencoderKLMiniMaxH3Audio

                return AutoencoderKLMiniMaxH3Audio.from_pretrained(
                    submodel_path, torch_dtype=torch.float32, local_files_only=True
                )
            case _:
                raise ValueError(f"Unsupported submodel type {submodel_type} for MiniMax H3 models.")
