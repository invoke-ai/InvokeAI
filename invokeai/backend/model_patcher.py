# Copyright (c) 2024 Ryan Dick, Lincoln D. Stein, and the InvokeAI Development Team
"""These classes implement model patching with LoRAs and Textual Inversions."""
from __future__ import annotations

import pickle
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers import OnnxRuntimeModel, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from invokeai.app.shared.models import FreeUConfig
from invokeai.backend.model_manager import AnyModel
from invokeai.backend.model_manager.load.optimizations import skip_torch_weight_init
from invokeai.backend.onnx.onnx_runtime import IAIOnnxRuntimeModel

from .lora import LoRAModelRaw
from .textual_inversion import TextualInversionManager, TextualInversionModelRaw

"""
loras = [
    (lora_model1, 0.7),
    (lora_model2, 0.4),
]
with LoRAHelper.apply_lora_unet(unet, loras):
    # unet with applied loras
# unmodified unet

"""


# TODO: rename smth like ModelPatcher and add TI method?
class ModelPatcher:
    @staticmethod
    def _resolve_lora_key(model: torch.nn.Module, lora_key: str, prefix: str) -> Tuple[str, torch.nn.Module]:
        assert "." not in lora_key

        if not lora_key.startswith(prefix):
            raise Exception(f"lora_key with invalid prefix: {lora_key}, {prefix}")

        module = model
        module_key = ""
        key_parts = lora_key[len(prefix) :].split("_")

        submodule_name = key_parts.pop(0)

        while len(key_parts) > 0:
            try:
                module = module.get_submodule(submodule_name)
                module_key += "." + submodule_name
                submodule_name = key_parts.pop(0)
            except Exception:
                submodule_name += "_" + key_parts.pop(0)

        module = module.get_submodule(submodule_name)
        module_key = (module_key + "." + submodule_name).lstrip(".")

        return (module_key, module)

    @classmethod
    @contextmanager
    def apply_lora_unet(
        cls,
        unet: UNet2DConditionModel,
        loras: Iterator[Tuple[LoRAModelRaw, float]],
    ) -> None:
        with cls.apply_lora(unet, loras, "lora_unet_"):
            yield

    @classmethod
    @contextmanager
    def apply_lora_text_encoder(
        cls,
        text_encoder: CLIPTextModel,
        loras: Iterator[Tuple[LoRAModelRaw, float]],
    ) -> None:
        with cls.apply_lora(text_encoder, loras, "lora_te_"):
            yield

    @classmethod
    @contextmanager
    def apply_sdxl_lora_text_encoder(
        cls,
        text_encoder: CLIPTextModel,
        loras: List[Tuple[LoRAModelRaw, float]],
    ) -> None:
        with cls.apply_lora(text_encoder, loras, "lora_te1_"):
            yield

    @classmethod
    @contextmanager
    def apply_sdxl_lora_text_encoder2(
        cls,
        text_encoder: CLIPTextModel,
        loras: List[Tuple[LoRAModelRaw, float]],
    ) -> None:
        with cls.apply_lora(text_encoder, loras, "lora_te2_"):
            yield

    @classmethod
    @contextmanager
    def apply_lora(
        cls,
        model: AnyModel,
        loras: Iterator[Tuple[LoRAModelRaw, float]],
        prefix: str,
    ) -> None:
        original_weights = {}
        try:
            with torch.no_grad():
                for lora, lora_weight in loras:
                    # assert lora.device.type == "cpu"
                    for layer_key, layer in lora.layers.items():
                        if not layer_key.startswith(prefix):
                            continue

                        # TODO(ryand): A non-negligible amount of time is currently spent resolving LoRA keys. This
                        # should be improved in the following ways:
                        # 1. The key mapping could be more-efficiently pre-computed. This would save time every time a
                        #    LoRA model is applied.
                        # 2. From an API perspective, there's no reason that the `ModelPatcher` should be aware of the
                        #    intricacies of Stable Diffusion key resolution. It should just expect the input LoRA
                        #    weights to have valid keys.
                        assert isinstance(model, torch.nn.Module)
                        module_key, module = cls._resolve_lora_key(model, layer_key, prefix)

                        # All of the LoRA weight calculations will be done on the same device as the module weight.
                        # (Performance will be best if this is a CUDA device.)
                        device = module.weight.device
                        dtype = module.weight.dtype

                        if module_key not in original_weights:
                            original_weights[module_key] = module.weight.detach().to(device="cpu", copy=True)

                        layer_scale = layer.alpha / layer.rank if (layer.alpha and layer.rank) else 1.0

                        # We intentionally move to the target device first, then cast. Experimentally, this was found to
                        # be significantly faster for 16-bit CPU tensors being moved to a CUDA device than doing the
                        # same thing in a single call to '.to(...)'.
                        layer.to(device=device)
                        layer.to(dtype=torch.float32)
                        # TODO(ryand): Using torch.autocast(...) over explicit casting may offer a speed benefit on CUDA
                        # devices here. Experimentally, it was found to be very slow on CPU. More investigation needed.
                        layer_weight = layer.get_weight(module.weight) * (lora_weight * layer_scale)
                        layer.to(device=torch.device("cpu"))

                        assert isinstance(layer_weight, torch.Tensor)  # mypy thinks layer_weight is a float|Any ??!
                        if module.weight.shape != layer_weight.shape:
                            # TODO: debug on lycoris
                            assert hasattr(layer_weight, "reshape")
                            layer_weight = layer_weight.reshape(module.weight.shape)

                        assert isinstance(layer_weight, torch.Tensor)  # mypy thinks layer_weight is a float|Any ??!
                        module.weight += layer_weight.to(dtype=dtype)

            yield  # wait for context manager exit

        finally:
            assert hasattr(model, "get_submodule")  # mypy not picking up fact that torch.nn.Module has get_submodule()
            with torch.no_grad():
                for module_key, weight in original_weights.items():
                    model.get_submodule(module_key).weight.copy_(weight)

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
                        device=text_encoder.device, dtype=text_encoder.dtype
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


class ONNXModelPatcher:
    @classmethod
    @contextmanager
    def apply_lora_unet(
        cls,
        unet: OnnxRuntimeModel,
        loras: Iterator[Tuple[LoRAModelRaw, float]],
    ) -> None:
        with cls.apply_lora(unet, loras, "lora_unet_"):
            yield

    @classmethod
    @contextmanager
    def apply_lora_text_encoder(
        cls,
        text_encoder: OnnxRuntimeModel,
        loras: List[Tuple[LoRAModelRaw, float]],
    ) -> None:
        with cls.apply_lora(text_encoder, loras, "lora_te_"):
            yield

    # based on
    # https://github.com/ssube/onnx-web/blob/ca2e436f0623e18b4cfe8a0363fcfcf10508acf7/api/onnx_web/convert/diffusion/lora.py#L323
    @classmethod
    @contextmanager
    def apply_lora(
        cls,
        model: IAIOnnxRuntimeModel,
        loras: List[Tuple[LoRAModelRaw, float]],
        prefix: str,
    ) -> None:
        from .models.base import IAIOnnxRuntimeModel

        if not isinstance(model, IAIOnnxRuntimeModel):
            raise Exception("Only IAIOnnxRuntimeModel models supported")

        orig_weights = {}

        try:
            blended_loras: Dict[str, torch.Tensor] = {}

            for lora, lora_weight in loras:
                for layer_key, layer in lora.layers.items():
                    if not layer_key.startswith(prefix):
                        continue

                    layer.to(dtype=torch.float32)
                    layer_key = layer_key.replace(prefix, "")
                    # TODO: rewrite to pass original tensor weight(required by ia3)
                    layer_weight = layer.get_weight(None).detach().cpu().numpy() * lora_weight
                    if layer_key in blended_loras:
                        blended_loras[layer_key] += layer_weight
                    else:
                        blended_loras[layer_key] = layer_weight

            node_names = {}
            for node in model.nodes.values():
                node_names[node.name.replace("/", "_").replace(".", "_").lstrip("_")] = node.name

            for layer_key, lora_weight in blended_loras.items():
                conv_key = layer_key + "_Conv"
                gemm_key = layer_key + "_Gemm"
                matmul_key = layer_key + "_MatMul"

                if conv_key in node_names or gemm_key in node_names:
                    if conv_key in node_names:
                        conv_node = model.nodes[node_names[conv_key]]
                    else:
                        conv_node = model.nodes[node_names[gemm_key]]

                    weight_name = [n for n in conv_node.input if ".weight" in n][0]
                    orig_weight = model.tensors[weight_name]

                    if orig_weight.shape[-2:] == (1, 1):
                        if lora_weight.shape[-2:] == (1, 1):
                            new_weight = orig_weight.squeeze((3, 2)) + lora_weight.squeeze((3, 2))
                        else:
                            new_weight = orig_weight.squeeze((3, 2)) + lora_weight

                        new_weight = np.expand_dims(new_weight, (2, 3))
                    else:
                        if orig_weight.shape != lora_weight.shape:
                            new_weight = orig_weight + lora_weight.reshape(orig_weight.shape)
                        else:
                            new_weight = orig_weight + lora_weight

                    orig_weights[weight_name] = orig_weight
                    model.tensors[weight_name] = new_weight.astype(orig_weight.dtype)

                elif matmul_key in node_names:
                    weight_node = model.nodes[node_names[matmul_key]]
                    matmul_name = [n for n in weight_node.input if "MatMul" in n][0]

                    orig_weight = model.tensors[matmul_name]
                    new_weight = orig_weight + lora_weight.transpose()

                    orig_weights[matmul_name] = orig_weight
                    model.tensors[matmul_name] = new_weight.astype(orig_weight.dtype)

                else:
                    # warn? err?
                    pass

            yield

        finally:
            # restore original weights
            for name, orig_weight in orig_weights.items():
                model.tensors[name] = orig_weight

    @classmethod
    @contextmanager
    def apply_ti(
        cls,
        tokenizer: CLIPTokenizer,
        text_encoder: IAIOnnxRuntimeModel,
        ti_list: List[Tuple[str, Any]],
    ) -> Iterator[Tuple[CLIPTokenizer, TextualInversionManager]]:
        from .models.base import IAIOnnxRuntimeModel

        if not isinstance(text_encoder, IAIOnnxRuntimeModel):
            raise Exception("Only IAIOnnxRuntimeModel models supported")

        orig_embeddings = None

        try:
            # HACK: The CLIPTokenizer API does not include a way to remove tokens after calling add_tokens(...). As a
            # workaround, we create a full copy of `tokenizer` so that its original behavior can be restored after
            # exiting this `apply_ti(...)` context manager.
            #
            # In a previous implementation, the deep copy was obtained with `ti_tokenizer = copy.deepcopy(tokenizer)`,
            # but a pickle roundtrip was found to be much faster (1 sec vs. 0.05 secs).
            ti_tokenizer = pickle.loads(pickle.dumps(tokenizer))
            ti_manager = TextualInversionManager(ti_tokenizer)

            def _get_trigger(ti_name: str, index: int) -> str:
                trigger = ti_name
                if index > 0:
                    trigger += f"-!pad-{i}"
                return f"<{trigger}>"

            # modify text_encoder
            orig_embeddings = text_encoder.tensors["text_model.embeddings.token_embedding.weight"]

            # modify tokenizer
            new_tokens_added = 0
            for ti_name, ti in ti_list:
                if ti.embedding_2 is not None:
                    ti_embedding = (
                        ti.embedding_2 if ti.embedding_2.shape[1] == orig_embeddings.shape[0] else ti.embedding
                    )
                else:
                    ti_embedding = ti.embedding

                for i in range(ti_embedding.shape[0]):
                    new_tokens_added += ti_tokenizer.add_tokens(_get_trigger(ti_name, i))

            embeddings = np.concatenate(
                (np.copy(orig_embeddings), np.zeros((new_tokens_added, orig_embeddings.shape[1]))),
                axis=0,
            )

            for ti_name, _ in ti_list:
                ti_tokens = []
                for i in range(ti_embedding.shape[0]):
                    embedding = ti_embedding[i].detach().numpy()
                    trigger = _get_trigger(ti_name, i)

                    token_id = ti_tokenizer.convert_tokens_to_ids(trigger)
                    if token_id == ti_tokenizer.unk_token_id:
                        raise RuntimeError(f"Unable to find token id for token '{trigger}'")

                    if embeddings[token_id].shape != embedding.shape:
                        raise ValueError(
                            f"Cannot load embedding for {trigger}. It was trained on a model with token dimension"
                            f" {embedding.shape[0]}, but the current model has token dimension"
                            f" {embeddings[token_id].shape[0]}."
                        )

                    embeddings[token_id] = embedding
                    ti_tokens.append(token_id)

                if len(ti_tokens) > 1:
                    ti_manager.pad_tokens[ti_tokens[0]] = ti_tokens[1:]

            text_encoder.tensors["text_model.embeddings.token_embedding.weight"] = embeddings.astype(
                orig_embeddings.dtype
            )

            yield ti_tokenizer, ti_manager

        finally:
            # restore
            if orig_embeddings is not None:
                text_encoder.tensors["text_model.embeddings.token_embedding.weight"] = orig_embeddings
