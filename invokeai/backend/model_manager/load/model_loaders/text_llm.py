"""Loaders for TextLLM models: transformers folders, and single-file Qwen3.5 vision-language models.

Both serve the model, its tokenizer and its processor as submodels, so a caller reaches the text side of either
container the same way, through the model cache (`load_text_llm_tokenizer_and_processor`).
"""

from collections.abc import Callable
from logging import Logger
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import accelerate
import torch
from safetensors.torch import load_file
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)
from transformers.utils import GENERATION_CONFIG_NAME

from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.configs.qwen3_5_encoder import get_qwen3_5_variant
from invokeai.backend.model_manager.configs.text_llm import (
    IMAGE_TEXT_TO_TEXT_LLMS,
    TextLLM_Checkpoint_Config,
    TextLLM_Diffusers_Config,
)
from invokeai.backend.model_manager.load.load_default import ModelLoader, _model_declared_skip_patterns
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.taxonomy import AnyModel, BaseModelType, ModelFormat, ModelType, SubModelType
from invokeai.backend.quantization.fp8_scaled import cast_state_dict, reject_quantized_side_channel
from invokeai.backend.quantization.int8_convrot import (
    drop_unconsumed_quantization_sidecars,
    extract_int8_convrot_markers,
    install_int8_convrot_layers,
    reject_unmarked_int8_weights,
)
from invokeai.backend.quantization.load_plan import reserve_for_load
from invokeai.backend.qwen3_5.qwen3_5_vl import (
    QWEN3_5_VL_LANGUAGE_MODEL_PREFIX,
    build_qwen3_5_vl_processor,
    qwen3_5_vl_config,
    qwen3_5_vl_generation_config,
)
from invokeai.backend.util.state_dict_loading import load_state_dict_ignoring_extras, reject_incomplete_load

if TYPE_CHECKING:
    from transformers import Qwen3_5ForConditionalGeneration

# The model types with Gated DeltaNet layers, which do not run in float16: these load in bfloat16 -- float32 on a
# device without it -- whatever precision is configured. Qwen3.5's MoE and Qwen3-Next folders load here as causal LMs.
_GATED_DELTANET_MODEL_TYPES = frozenset({"qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text", "qwen3_next"})

# Qwen3.5's model types. A stock Qwen3.5 release ships no generation config: one derived from its `config.json` stops
# only at `<|endoftext|>`, while the chat template ends the assistant's turn with `<|im_end|>`. Such a folder takes the
# end-of-turn and padding ids of the vendored generation config, written for this tokenizer.
_QWEN3_5_MODEL_TYPES = frozenset({"qwen3_5", "qwen3_5_text"})

# Widened on load although the int8 build stores it quantized. The LM head is 248320 x 4096: as an
# `Int8ConvrotLinear` it would materialize two bf16 copies on every forward -- 3.8 GiB, more than the default working
# memory -- and every generated token would dequantize all of it for a one-row matmul. Dense, it costs 0.95 GiB more.
_DEQUANTIZED_ON_LOAD = ("lm_head",)


class _TextLLMLoader(ModelLoader):
    """What both TextLLM containers share: tokenizer and processor sizes, and the compute dtype."""

    def get_size_fs(
        self, config: AnyModelConfig, model_path: Path, submodel_type: Optional[SubModelType] = None
    ) -> int:
        # A few small files, or none at all: the generic sizing would charge them the whole model.
        if submodel_type in (SubModelType.Tokenizer, SubModelType.Processor):
            return 0
        return super().get_size_fs(config, model_path, submodel_type)

    def _compute_dtype(self, cpu_only: bool | None, *, avoid_float16: bool) -> torch.dtype:
        # float32 for CPU-only models: CPU float16 is emulated and slow.
        if cpu_only is True:
            return torch.float32
        return self._torch_dtype_avoiding_float16() if avoid_float16 else self._torch_dtype


@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.TextLLM, format=ModelFormat.Diffusers)
class TextLLMModelLoader(_TextLLMLoader):
    """Loads a transformers language model folder (Llama, Phi, Qwen, Gemma-4, Qwen3.5, ...), its tokenizer and its
    processor."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, TextLLM_Diffusers_Config):
            raise ValueError(f"Expected a TextLLM_Diffusers_Config, got {type(config).__name__}.")

        model_path = Path(config.path)
        match submodel_type:
            case None:
                hf_config = AutoConfig.from_pretrained(model_path, local_files_only=True)
                architecture = (hf_config.architectures or [""])[0]
                auto_class = (
                    AutoModelForImageTextToText if architecture in IMAGE_TEXT_TO_TEXT_LLMS else AutoModelForCausalLM
                )
                dtype = self._compute_dtype(
                    config.cpu_only, avoid_float16=hf_config.model_type in _GATED_DELTANET_MODEL_TYPES
                )
                model = auto_class.from_pretrained(model_path, config=hf_config, local_files_only=True, dtype=dtype)
                if hf_config.model_type in _QWEN3_5_MODEL_TYPES and not (model_path / GENERATION_CONFIG_NAME).is_file():
                    vendored = qwen3_5_vl_generation_config()
                    model.generation_config.eos_token_id = vendored.eos_token_id
                    model.generation_config.pad_token_id = vendored.pad_token_id
                return model
            case SubModelType.Tokenizer:
                return AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            case SubModelType.Processor:
                return AutoProcessor.from_pretrained(model_path, local_files_only=True)

        raise ValueError(
            "Only the model, its Tokenizer and its Processor can be loaded from a TextLLM. "
            f"Received: {submodel_type.value if submodel_type else 'None'}"
        )


@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.TextLLM, format=ModelFormat.Checkpoint)
class TextLLMCheckpointLoader(_TextLLMLoader):
    """Loads a single-file Qwen3.5 vision-language model, and serves its own instances of the vendored Qwen3.5
    tokenizer and processor."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, TextLLM_Checkpoint_Config):
            raise ValueError(f"Expected a TextLLM_Checkpoint_Config, got {type(config).__name__}.")

        match submodel_type:
            case None:
                path = Path(config.path)
                what = f"Qwen3.5 checkpoint {path.name}"
                model = qwen3_5_vl_from_state_dict(
                    load_file(path),
                    self._compute_dtype(config.cpu_only, avoid_float16=True),
                    reserve=self._ram_cache.make_room,
                    what=what,
                    logger=self._logger,
                )
                reject_incomplete_load(model, what=what)
                return model
            case SubModelType.Tokenizer:
                from invokeai.backend.qwen3_5.qwen3_5_encoder import load_qwen3_5_tokenizer

                # Its own instance, cached by the model cache: the process-wide one is shared with the Qwen3.5
                # encoders, whose truncating encodes would race Expand Prompt's on the same object.
                return load_qwen3_5_tokenizer()
            case SubModelType.Processor:
                return build_qwen3_5_vl_processor()

        raise ValueError(
            "Only the model, its Tokenizer and its Processor can be loaded from a TextLLM. "
            f"Received: {submodel_type.value if submodel_type else 'None'}"
        )


def qwen3_5_vl_from_state_dict(
    sd: dict[str, torch.Tensor],
    dtype: torch.dtype,
    *,
    reserve: Callable[[int], None],
    what: str,
    logger: Logger,
) -> "Qwen3_5ForConditionalGeneration":
    """Build a `Qwen3_5ForConditionalGeneration` from a single file's state dict, bf16 or `int8_tensorwise`.

    The keys are transformers' own, so nothing is renamed; the architecture is the vendored config of the
    checkpoint's width. The multi-token-prediction head (`mtp.*`) the int8 build ships is dropped: generation never
    runs it. Marked Linears stay int8 and dequantize per forward. The embedding, which is not a Linear, and the LM
    head (see `_DEQUANTIZED_ON_LOAD`) are widened to `dtype` on load, after `reserve` has been asked for room.

    Args:
        sd: The file's tensors. Consumed: the model takes them over.
        dtype: The compute dtype for every tensor that does not stay int8.
        reserve: The cache's `make_room`, asked once before anything is widened.
        what: Names the checkpoint in errors and log lines.
        logger: For the dropped and kept counts.
    """
    from transformers import Qwen3_5ForConditionalGeneration

    mtp = [key for key in sd if key.startswith("mtp.")]
    for key in mtp:
        del sd[key]
    if mtp:
        logger.info(
            f"{what}: dropped {len(mtp)} multi-token-prediction tensor(s) (mtp.*), which generation never runs."
        )

    markers = extract_int8_convrot_markers(sd)
    # Outside the int8 branch, so an int8 weight whose marker is missing or unreadable is refused, not cast as codes.
    reject_unmarked_int8_weights(sd, markers, "Qwen3.5")

    variant = get_qwen3_5_variant(sd, QWEN3_5_VL_LANGUAGE_MODEL_PREFIX)
    if variant is None:
        raise ValueError(f"{what} holds no Qwen3.5 language model of a known width under model.language_model.")
    with accelerate.init_empty_weights():
        model = Qwen3_5ForConditionalGeneration(qwen3_5_vl_config(variant))
    # The config alone knows one end-of-turn token; the release stops at both.
    model.generation_config = qwen3_5_vl_generation_config()

    skip_patterns = _model_declared_skip_patterns(model)
    if markers:
        sd = drop_unconsumed_quantization_sidecars(sd)
        kept = install_int8_convrot_layers(
            model,
            sd,
            markers,
            dtype,
            architecture="Qwen3.5",
            reserve=reserve,
            skip_patterns=(*skip_patterns, *_DEQUANTIZED_ON_LOAD),
        )
        logger.info(
            f"{what}: kept {len(kept)} of {len(markers)} layer(s) in int8 (int8_tensorwise checkpoint, dequantized "
            "per forward); the embedding and the LM head are widened on load."
        )
    else:
        # A scaled fp8 or nvfp4 build would load its codes without their scales on this path.
        reject_quantized_side_channel(sd, what)
        reserve_for_load(
            reserve,
            sd,
            dtype,
            keep_fp8=False,
            model=model,
            skip_patterns=skip_patterns,
            fp8_layers={},
            nvfp4_payloads={},
        )
        cast_state_dict(sd, dtype, keep_fp8=False, model=model, skip_patterns=skip_patterns)

    load_state_dict_ignoring_extras(model, sd, source=what, assign=True)
    sd.clear()
    return model
