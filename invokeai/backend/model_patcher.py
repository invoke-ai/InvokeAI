# Copyright (c) 2024 Ryan Dick, Lincoln D. Stein, and the InvokeAI Development Team
"""These classes implement model patching with LoRAs and Textual Inversions."""

from __future__ import annotations

import pickle
from contextlib import contextmanager
from typing import Any, Iterator, List, Optional, Tuple, Type, Union

import torch
from diffusers import UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from invokeai.app.shared.models import FreeUConfig
from invokeai.backend.model_manager.load.optimizations import skip_torch_weight_init
from invokeai.backend.textual_inversion import TextualInversionManager, TextualInversionModelRaw
from invokeai.backend.util.devices import TorchDevice


class ModelPatcher:
    @staticmethod
    @contextmanager
    def patch_unet_attention_processor(unet: UNet2DConditionModel, processor_cls: Type[Any]):
        """A context manager that patches `unet` with the provided attention processor.

        Args:
            unet (UNet2DConditionModel): The UNet model to patch.
            processor (Type[Any]): Class which will be initialized for each key and passed to set_attn_processor(...).
        """
        unet_orig_processors = unet.attn_processors

        # create separate instance for each attention, to be able modify each attention separately
        unet_new_processors = {key: processor_cls() for key in unet_orig_processors.keys()}
        try:
            unet.set_attn_processor(unet_new_processors)
            yield None

        finally:
            unet.set_attn_processor(unet_orig_processors)

    @classmethod
    @contextmanager
    def apply_ti(
        cls,
        tokenizer: CLIPTokenizer,
        text_encoder: Union[CLIPTextModel, CLIPTextModelWithProjection],
        ti_list: List[Tuple[str, TextualInversionModelRaw]],
    ) -> Iterator[Tuple[CLIPTokenizer, TextualInversionManager]]:
        init_tokens_count = None
        new_tokens_added = None

        # TODO: This is required since Transformers 4.32 see
        # https://github.com/huggingface/transformers/pull/25088
        # More information by NVIDIA:
        # https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc
        # This value might need to be changed in the future and take the GPUs model into account as there seem
        # to be ideal values for different GPUS. This value is temporary!
        # For references to the current discussion please see https://github.com/invoke-ai/InvokeAI/pull/4817
        pad_to_multiple_of = 8

        try:
            # HACK: The CLIPTokenizer API does not include a way to remove tokens after calling add_tokens(...). As a
            # workaround, we create a full copy of `tokenizer` so that its original behavior can be restored after
            # exiting this `apply_ti(...)` context manager.
            #
            # In a previous implementation, the deep copy was obtained with `ti_tokenizer = copy.deepcopy(tokenizer)`,
            # but a pickle roundtrip was found to be much faster (1 sec vs. 0.05 secs).
            ti_tokenizer = pickle.loads(pickle.dumps(tokenizer))
            ti_manager = TextualInversionManager(ti_tokenizer)
            init_tokens_count = text_encoder.resize_token_embeddings(None, pad_to_multiple_of).num_embeddings

            def _get_trigger(ti_name: str, index: int) -> str:
                trigger = ti_name
                if index > 0:
                    trigger += f"-!pad-{i}"
                return f"<{trigger}>"

            def _get_ti_embedding(model_embeddings: torch.nn.Module, ti: TextualInversionModelRaw) -> torch.Tensor:
                # for SDXL models, select the embedding that matches the text encoder's dimensions
                if ti.embedding_2 is not None:
                    return (
                        ti.embedding_2
                        if ti.embedding_2.shape[1] == model_embeddings.weight.data[0].shape[0]
                        else ti.embedding
                    )
                else:
                    return ti.embedding

            # modify tokenizer
            new_tokens_added = 0
            for ti_name, ti in ti_list:
                ti_embedding = _get_ti_embedding(text_encoder.get_input_embeddings(), ti)

                for i in range(ti_embedding.shape[0]):
                    new_tokens_added += ti_tokenizer.add_tokens(_get_trigger(ti_name, i))

            # Modify text_encoder.
            # resize_token_embeddings(...) constructs a new torch.nn.Embedding internally. Initializing the weights of
            # this embedding is slow and unnecessary, so we wrap this step in skip_torch_weight_init() to save some
            # time.
            with skip_torch_weight_init():
                text_encoder.resize_token_embeddings(init_tokens_count + new_tokens_added, pad_to_multiple_of)
            model_embeddings = text_encoder.get_input_embeddings()

            for ti_name, ti in ti_list:
                assert isinstance(ti, TextualInversionModelRaw)
                ti_embedding = _get_ti_embedding(text_encoder.get_input_embeddings(), ti)

                ti_tokens = []
                for i in range(ti_embedding.shape[0]):
                    embedding = ti_embedding[i]
                    trigger = _get_trigger(ti_name, i)

                    token_id = ti_tokenizer.convert_tokens_to_ids(trigger)
                    if token_id == ti_tokenizer.unk_token_id:
                        raise RuntimeError(f"Unable to find token id for token '{trigger}'")

                    if model_embeddings.weight.data[token_id].shape != embedding.shape:
                        raise ValueError(
                            f"Cannot load embedding for {trigger}. It was trained on a model with token dimension"
                            f" {embedding.shape[0]}, but the current model has token dimension"
                            f" {model_embeddings.weight.data[token_id].shape[0]}."
                        )

                    model_embeddings.weight.data[token_id] = embedding.to(
                        device=TorchDevice.choose_torch_device(), dtype=text_encoder.dtype
                    )
                    ti_tokens.append(token_id)

                if len(ti_tokens) > 1:
                    ti_manager.pad_tokens[ti_tokens[0]] = ti_tokens[1:]

            yield ti_tokenizer, ti_manager

        finally:
            if init_tokens_count and new_tokens_added:
                text_encoder.resize_token_embeddings(init_tokens_count, pad_to_multiple_of)

    @classmethod
    @contextmanager
    def apply_clip_skip(
        cls,
        text_encoder: Union[CLIPTextModel, CLIPTextModelWithProjection],
        clip_skip: int,
    ) -> None:
        skipped_layers = []
        try:
            for _i in range(clip_skip):
                skipped_layers.append(text_encoder.text_model.encoder.layers.pop(-1))

            yield

        finally:
            while len(skipped_layers) > 0:
                text_encoder.text_model.encoder.layers.append(skipped_layers.pop())

    @classmethod
    @contextmanager
    def apply_freeu(
        cls,
        unet: UNet2DConditionModel,
        freeu_config: Optional[FreeUConfig] = None,
    ) -> None:
        did_apply_freeu = False
        try:
            assert hasattr(unet, "enable_freeu")  # mypy doesn't pick up this attribute?
            if freeu_config is not None:
                unet.enable_freeu(b1=freeu_config.b1, b2=freeu_config.b2, s1=freeu_config.s1, s2=freeu_config.s2)
                did_apply_freeu = True

            yield

        finally:
            assert hasattr(unet, "disable_freeu")  # mypy doesn't pick up this attribute?
            if did_apply_freeu:
                unet.disable_freeu()
