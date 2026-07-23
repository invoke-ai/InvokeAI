"""Loader for the Gemma-2 text encoder used by PiD.

PiD only consumes the decoder block of the causal LM (see
`pid/_src/models/pixeldit_model.py::_load_text_encoder`:
`AutoModelForCausalLM.from_pretrained(...).get_decoder()`), so this loader
returns the decoder sub-module for the `TextEncoder` submodel and the
tokenizer for the `Tokenizer` submodel.
"""

from pathlib import Path
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer

from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.configs.gemma2_encoder import Gemma2Encoder_Gemma2Encoder_Config
from invokeai.backend.model_manager.load.load_default import ModelLoader
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.taxonomy import AnyModel, BaseModelType, ModelFormat, ModelType, SubModelType
from invokeai.backend.util.devices import TorchDevice


@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.Gemma2Encoder, format=ModelFormat.Gemma2Encoder)
class Gemma2EncoderLoader(ModelLoader):
    """Loads a Gemma-2 causal LM directory and exposes its decoder + tokenizer."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, Gemma2Encoder_Gemma2Encoder_Config):
            raise ValueError("Only Gemma2Encoder_Gemma2Encoder_Config models are supported here.")

        model_path = Path(config.path)

        match submodel_type:
            case SubModelType.Tokenizer:
                return AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            case SubModelType.TextEncoder:
                target_device = TorchDevice.choose_torch_device()
                model_dtype = TorchDevice.choose_bfloat16_safe_dtype(target_device)
                causal_lm = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=model_dtype,
                    low_cpu_mem_usage=True,
                    local_files_only=True,
                )
                # PiD only ever uses the decoder block — the transformer stack
                # without the LM head. Upstream calls `.get_decoder()`, but
                # transformers 4.56 returns None for Gemma2, so we reach for
                # `.model` (the underlying Gemma2Model) directly and let the
                # rest of `causal_lm` (lm_head etc.) be garbage-collected.
                inner = getattr(causal_lm, "get_decoder", lambda: None)() or causal_lm.model
                inner.eval()
                inner.requires_grad_(False)
                return inner

        raise ValueError(
            f"Unsupported submodel type for Gemma2 encoder: {submodel_type!r}. Expected Tokenizer or TextEncoder."
        )
