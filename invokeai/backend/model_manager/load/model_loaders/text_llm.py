"""Loaders for TextLLM models: transformers folders, single-file Qwen3.5 vision-language models, and the Text LLM view
of single-file Qwen3-VL encoders.

Each serves the model, its tokenizer and, where it reads images, its processor as submodels, so a caller reaches the
text side of any container the same way, through the model cache (`load_text_llm_tokenizer_and_processor`).
"""

from collections.abc import Callable
from logging import Logger
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import accelerate
import torch
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
from invokeai.backend.model_manager.configs.qwen3_vl_encoder import Qwen3VLEncoder_Checkpoint_Config
from invokeai.backend.model_manager.configs.roles import view_as
from invokeai.backend.model_manager.configs.text_llm import (
    IMAGE_TEXT_TO_TEXT_LLMS,
    TextLLM_Checkpoint_Config,
    TextLLM_Diffusers_Config,
    TextLLM_Qwen3VLEncoder_Config,
)
from invokeai.backend.model_manager.load.load_base import LoadedModel
from invokeai.backend.model_manager.load.load_default import ModelLoader, _model_declared_skip_patterns
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.load.model_loaders.krea2 import Qwen3VLEncoderCheckpointLoader
from invokeai.backend.model_manager.model_on_disk import safetensors_meta_state_dict
from invokeai.backend.model_manager.taxonomy import AnyModel, BaseModelType, ModelFormat, ModelType, SubModelType
from invokeai.backend.model_manager.util.qwen3_vl import (
    qwen3vl_lm_head_entries,
    qwen3vl_lm_head_hints,
    qwen3vl_lm_head_refusal,
    qwen3vl_token_embedding,
    read_qwen3vl_lm_head_tensors,
)
from invokeai.backend.quantization.fp8_scaled import (
    FP8_WEIGHT_DTYPES,
    cast_state_dict,
    dequantize_fp8_scaled,
    extract_fp8_scaled_layers,
    fold_transient_bytes,
    load_safetensors_checkpoint,
    read_safetensors_metadata,
    reject_quantized_side_channel,
)
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
from invokeai.backend.qwen3_vl.qwen3_vl_assets import (
    load_qwen3_vl_tokenizer,
    qwen3_vl_instruct_generation_config,
    qwen3_vl_ties_word_embeddings,
)
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.state_dict_loading import load_state_dict_ignoring_extras, reject_incomplete_load

if TYPE_CHECKING:
    from transformers import Qwen3_5ForConditionalGeneration, Qwen3VLForConditionalGeneration

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

    def _without_mtp_head(self, names: dict[str, str], what: str) -> dict[str, str]:
        """What of a single file is worth reading: not the multi-token-prediction head, which is never read at all."""
        kept = {key: value for key, value in names.items() if not is_multi_token_prediction_key(key)}
        _log_dropped_mtp(self._logger, what, len(names) - len(kept))
        return kept

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
                    load_safetensors_checkpoint(path, select=lambda names: self._without_mtp_head(names, what)),
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


class _Qwen3VLTextLLMDecoder(Qwen3VLEncoderCheckpointLoader):
    """The encoder loader's decode, with a Text LLM's compute dtype: float32 on the CPU, where float16 and bfloat16 are
    emulated or slow, otherwise the configured one except float16. Everything else the encoder's load decision settles
    (where fp8 stays packed, the cache-key variant) is the encoder's own."""

    def _select_model_dtype(
        self, config: AnyModelConfig, submodel_type: Optional[SubModelType], device: torch.device
    ) -> torch.dtype:
        if device.type == "cpu":
            return torch.float32
        return TorchDevice.dtype_avoiding_float16(self._torch_dtype, device)


@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.TextLLM, format=ModelFormat.Qwen3VLEncoder)
class TextLLMQwen3VLLoader(_TextLLMLoader):
    """Loads the Text LLM view of a single-file Qwen3-VL encoder (`TextLLM_Qwen3VLEncoder_Config`): the whole Qwen3-VL
    Instruct model, text only, with its own instance of the vendored Qwen3-VL tokenizer.

    The language model is the encoder loader's own decode, so every quantization the encoder takes loads here alike.
    The visual tower stays on disk. The LM head is the token embedding where the variant ties them (the 4B), and
    otherwise the file's, widened to the compute dtype on load.
    """

    _decoder_for: Optional[tuple[AnyModelConfig, Qwen3VLEncoder_Checkpoint_Config, _Qwen3VLTextLLMDecoder]] = None

    def load_model(self, model_config: AnyModelConfig, submodel_type: Optional[SubModelType] = None) -> LoadedModel:
        self._decoder_for = None
        return super().load_model(model_config, submodel_type)

    def _decoder(self, config: AnyModelConfig) -> tuple[Qwen3VLEncoder_Checkpoint_Config, _Qwen3VLTextLLMDecoder]:
        """The file as an encoder, and the loader that decodes it. Kept for the load, so the cache key and the decode
        read one load decision rather than two probes that could disagree."""
        if self._decoder_for is None or self._decoder_for[0] is not config:
            encoder_config = view_as(config, ModelType.Qwen3VLEncoder)
            if not isinstance(encoder_config, Qwen3VLEncoder_Checkpoint_Config):
                raise ValueError(f"'{config.name}' is not a single-file Qwen3-VL encoder; re-identify it.")
            decoder = _Qwen3VLTextLLMDecoder(self._app_config, self._logger, self._ram_cache)
            self._decoder_for = (config, encoder_config, decoder)
        return self._decoder_for[1], self._decoder_for[2]

    def _cache_key_variant(self, config: AnyModelConfig, submodel_type: Optional[SubModelType] = None) -> str:
        """The language model's storage format, which depends on the device it is decoded for (fp8 kept packed or
        folded, the compute dtype): devices that would build different weights must not adopt each other's copy. Asked
        of the decode itself, so it is the encoder's own answer."""
        if submodel_type is not None:
            return ""
        encoder_config, decoder = self._decoder(config)
        return decoder._cache_key_variant(encoder_config, SubModelType.TextEncoder)

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, TextLLM_Qwen3VLEncoder_Config):
            raise ValueError(f"Expected a TextLLM_Qwen3VLEncoder_Config, got {type(config).__name__}.")

        match submodel_type:
            case None:
                return self._load_qwen3_vl(config)
            case SubModelType.Tokenizer:
                # Its own instance, as for Qwen3.5: the process-wide one is shared with the Qwen3-VL encoders.
                return load_qwen3_vl_tokenizer()
            case SubModelType.Processor:
                raise ValueError(
                    f"'{config.name}' runs as a text-only LLM: its vision tower is not loaded, so it cannot read images."
                )

        raise ValueError(
            "Only the model and its Tokenizer can be loaded from a Qwen3-VL encoder serving as a TextLLM. "
            f"Received: {submodel_type.value if submodel_type else 'None'}"
        )

    def _load_qwen3_vl(self, config: TextLLM_Qwen3VLEncoder_Config) -> "Qwen3VLForConditionalGeneration":
        from transformers import Qwen3VLForConditionalGeneration

        encoder_config, decoder = self._decoder(config)
        path = Path(config.path)
        what = f"Qwen3-VL checkpoint {path.name}"
        # float32 on the CPU with the fp8 weights folded there, rather than dequantized on every token.
        dtype = decoder._decision(encoder_config, SubModelType.TextEncoder).dtype

        # The geometry is the encoder's: the view's own variant is the role's, which a later edit of the record's
        # variant leaves behind.
        tied = qwen3_vl_ties_word_embeddings(encoder_config.variant)
        hints: dict[str, Any] = {}
        head_bytes = 0
        if not tied:
            # Judged from the header, so a head the loader refuses costs no load; its bytes are read after the decode.
            header = safetensors_meta_state_dict(path)
            head_meta = qwen3vl_lm_head_entries(header)
            hints = qwen3vl_lm_head_hints(path, read_safetensors_metadata(path, self._logger))
            refusal = qwen3vl_lm_head_refusal(head_meta, hints, qwen3vl_token_embedding(header))
            if refusal is not None:
                raise ValueError(f"'{config.name}' cannot run as a Text LLM: {refusal}. Re-identify it.")
            head_bytes = _lm_head_widening_bytes(head_meta, dtype)

        # The decode makes the one reservation, before it widens anything; the head widens after it.
        language_model = decoder.load_language_model(
            encoder_config, reserve=lambda needed: self._ram_cache.make_room(needed + head_bytes)
        )

        with accelerate.init_empty_weights():
            model = Qwen3VLForConditionalGeneration(language_model.config)
        model.model = language_model
        if tied:
            model.lm_head.weight = model.get_input_embeddings().weight
        else:
            head = read_qwen3vl_lm_head_tensors(path)
            layers = extract_fp8_scaled_layers(head, layer_hints={"lm_head": hints})
            dequantize_fp8_scaled(head, layers, dtype)
            cast_state_dict(head, dtype, keep_fp8=False)
            model.lm_head.weight = torch.nn.Parameter(head.pop("lm_head.weight"), requires_grad=False)
        model.generation_config = qwen3_vl_instruct_generation_config()
        reject_incomplete_load(model, what=what)
        self._logger.info(
            f"{what}: LM head {'tied to the token embedding' if tied else f'read from the file in {dtype}'}."
        )
        return model


def _lm_head_widening_bytes(head: dict[str, Any], dtype: torch.dtype) -> int:
    """What widening the LM head to `dtype` holds beside the language model: the result, the stored weight until it is
    replaced (nothing when it already is `dtype`), and for scaled fp8 the float32 scales and the fold's slabs.

    Reads `qwen3vl_lm_head_entries` of the header, before anything is loaded.
    """
    weight = head["weight"]
    total = weight.numel() * dtype.itemsize
    if weight.dtype != dtype:
        total += weight.numel() * weight.element_size()
    scale = head.get("weight_scale", head.get("scale_weight"))
    if weight.dtype in FP8_WEIGHT_DTYPES and scale is not None:
        scales = [
            head[suffix] for suffix in ("weight_scale", "scale_weight", "input_scale", "scale_input") if suffix in head
        ]
        total += sum(tensor.numel() * torch.float32.itemsize for tensor in scales)
        total += fold_transient_bytes(weight, scale)
    return total


def is_multi_token_prediction_key(key: str) -> bool:
    """The `mtp.*` head the int8 build ships: generation never runs it."""
    return key.startswith("mtp.")


def _log_dropped_mtp(logger: Logger, what: str, count: int) -> None:
    if count:
        logger.info(f"{what}: dropped {count} multi-token-prediction tensor(s) (mtp.*), which generation never runs.")


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

    dropped = [key for key in sd if is_multi_token_prediction_key(key)]
    for key in dropped:
        del sd[key]
    _log_dropped_mtp(logger, what, len(dropped))

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
