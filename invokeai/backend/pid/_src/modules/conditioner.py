# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union

import torch
import torch.nn as nn
from torch.distributed import ProcessGroup

from invokeai.backend.pid._ext.imaginaire.lazy_config import instantiate
from invokeai.backend.pid._ext.imaginaire.utils import log
from invokeai.backend.pid._ext.imaginaire.utils.count_params import count_params, disabled_train
from invokeai.backend.pid._src.utils.context_parallel import broadcast


def batch_mul(x, y):
    """Broadcast-multiply x by y, padding the shorter shape with trailing 1s."""
    nd1, nd2 = x.ndim, y.ndim
    common = min(nd1, nd2)
    for axis in range(common):
        assert x.shape[axis] == y.shape[axis], f"Dimensions not equal at axis {axis}"
    if nd1 < nd2:
        x = x.reshape(x.shape + (1,) * (nd2 - nd1))
    elif nd2 < nd1:
        y = y.reshape(y.shape + (1,) * (nd1 - nd2))
    return x * y


T = TypeVar("T", bound="BaseCondition")


def broadcast_condition(condition: BaseCondition, process_group: Optional[ProcessGroup] = None) -> BaseCondition:
    """
    Broadcast the condition from the minimum rank in the specified group(s).
    """
    if condition.is_broadcasted:
        return condition

    kwargs = condition.to_dict(skip_underscore=False)
    for key, value in kwargs.items():
        if value is not None:
            kwargs[key] = broadcast(value, process_group)
    kwargs["_is_broadcasted"] = True
    return type(condition)(**kwargs)


@dataclass(frozen=True)
class BaseCondition(ABC):  # noqa: B024  # upstream marker base class — no abstract methods by design
    """
    Attributes:
        _is_broadcasted: Flag indicating if parallel broadcast splitting
            has been performed. This is an internal implementation detail.
    """

    _is_broadcasted: bool = False

    def to_dict(self, skip_underscore: bool = True) -> Dict[str, Any]:
        """Converts the condition to a dictionary.

        Returns:
            Dictionary containing the condition's fields and values.
        """
        # return {f.name: getattr(self, f.name) for f in fields(self) if not f.name.startswith("_")}
        return {f.name: getattr(self, f.name) for f in fields(self) if not (f.name.startswith("_") and skip_underscore)}

    @property
    def is_broadcasted(self) -> bool:
        return self._is_broadcasted

    def broadcast(self, process_group: torch.distributed.ProcessGroup) -> BaseCondition:
        """Broadcasts and splits the condition across the checkpoint parallelism group.
        For most condition, such asT2VCondition, we do not need split.

        Args:
            process_group: The process group for broadcast and split

        Returns:
            A new BaseCondition instance with the broadcasted and split condition.
        """
        if self.is_broadcasted:
            return self
        return broadcast_condition(self, process_group)


@dataclass(frozen=True)
class PixelDiTCondition(BaseCondition):
    """Condition for PixelDiT T2I models.

    caption: list[str] — raw caption strings (after dropout). The model's internal
        text encoder (e.g. Gemma-2-2b-it) handles encoding.
    """

    caption: Optional[list] = None


@dataclass(frozen=True)
class PidCondition(BaseCondition):
    """Condition for PID (PixelDiT SR) models.

    caption: list[str] — raw caption strings (after dropout).
    lq_video_or_image: [B, 3, H_lq, W_lq] — LQ image at original low resolution.
    lq_latent: [B, z_dim, zH, zW] — LQ VAE latent.
    """

    caption: Optional[list] = None
    lq_video_or_image: Optional[torch.Tensor] = None
    lq_latent: Optional[torch.Tensor] = None


class AbstractEmbModel(nn.Module):
    def __init__(self):
        super().__init__()

        self._is_trainable = None
        self._dropout_rate = None
        self._input_key = None
        self._return_dict = False

    @property
    def is_trainable(self) -> bool:
        return self._is_trainable

    @property
    def dropout_rate(self) -> Union[float, torch.Tensor]:
        return self._dropout_rate

    @property
    def input_key(self) -> str:
        return self._input_key

    @property
    def is_return_dict(self) -> bool:
        return self._return_dict

    @is_trainable.setter
    def is_trainable(self, value: bool):
        self._is_trainable = value

    @dropout_rate.setter
    def dropout_rate(self, value: Union[float, torch.Tensor]):
        self._dropout_rate = value

    @input_key.setter
    def input_key(self, value: str):
        self._input_key = value

    @is_return_dict.setter
    def is_return_dict(self, value: bool):
        self._return_dict = value

    @is_trainable.deleter
    def is_trainable(self):
        del self._is_trainable

    @dropout_rate.deleter
    def dropout_rate(self):
        del self._dropout_rate

    @input_key.deleter
    def input_key(self):
        del self._input_key

    @is_return_dict.deleter
    def is_return_dict(self):
        del self._return_dict

    def random_dropout_input(
        self, in_tensor: torch.Tensor, dropout_rate: Optional[float] = None, key: Optional[str] = None
    ) -> torch.Tensor:
        del key
        dropout_rate = dropout_rate if dropout_rate is not None else self.dropout_rate
        return batch_mul(
            torch.bernoulli((1.0 - dropout_rate) * torch.ones(in_tensor.shape[0])).type_as(in_tensor),
            in_tensor,
        )

    def details(self) -> str:
        return ""

    def summary(self) -> str:
        input_key = self.input_key if self.input_key is not None else getattr(self, "input_keys", None)
        return (
            f"{self.__class__.__name__} \n\tinput key: {input_key}"
            f"\n\tParam count: {count_params(self, False)} \n\tTrainable: {self.is_trainable}"
            f"\n\tDropout rate: {self.dropout_rate}"
            f"\n\t{self.details()}"
        )


class CaptionStringDrop(AbstractEmbModel):
    """Embedder for raw caption strings with dropout (replaces with empty string).

    Unlike TextAttrEmptyStringDrop which operates on pre-computed tensor embeddings,
    this embedder handles raw caption strings (list[str]) from the data batch. On
    dropout, the caption is replaced with an empty string so the model's own text
    encoder produces null embeddings.

    Used by PixelDiT which encodes text inside the model (Gemma-2-2b-it) rather
    than consuming pre-computed UMT5 embeddings from the dataset.

    Args:
        input_key: key in data_batch containing caption strings (default: "caption")
        output_key: key in condition output (default: "caption")
        dropout_rate: probability of replacing caption with "" (for CFG training)
    """

    def __init__(self, input_key: str = "caption", output_key: str = "caption", dropout_rate: float = 0.0):
        super().__init__()
        self._input_key = input_key
        self._dropout_rate = dropout_rate
        self._output_key = output_key

    def forward(self, captions):
        # Ensure list[str] — random_dropout_input normalizes, but guard forward too
        if isinstance(captions, str):
            captions = [captions]
        return {self._output_key: captions}

    def random_dropout_input(self, in_data, dropout_rate=None, key=None):
        """Per-sample caption dropout: replace each caption with "" independently."""
        del key
        import random as _random

        if in_data is None:
            return in_data
        # Normalize: webdataset collate may return a single string when batch_size=1
        if isinstance(in_data, str):
            in_data = [in_data]
        dropout_rate = dropout_rate if dropout_rate is not None else self.dropout_rate
        if dropout_rate <= 0:
            return in_data
        return ["" if _random.random() < dropout_rate else cap for cap in in_data]

    def details(self) -> str:
        return f"Output key: [{self._output_key}]"


class GeneralConditioner(nn.Module, ABC):
    """
    An abstract module designed to handle various embedding models with conditional and unconditional configurations.
    This abstract base class initializes and manages a collection of embedders that can dynamically adjust
    their dropout rates based on conditioning.

    Attributes:
        KEY2DIM (dict): A mapping from output keys to dimensions used for concatenation.
        embedders (nn.ModuleDict): A dictionary containing all embedded models initialized and configured
                                   based on the provided configurations.

    Parameters:
        emb_models (Union[List, Any]): A dictionary where keys are embedder names and values are configurations
                                       for initializing the embedders.

    Example:
        See Edify4ConditionerConfig
    """

    KEY2DIM = {"crossattn_emb": 1}

    def __init__(self, **emb_models: Union[List, Any]):
        super().__init__()
        self.embedders = nn.ModuleDict()
        for n, (emb_name, emb_config) in enumerate(emb_models.items()):
            embedder = instantiate(emb_config)
            # assert isinstance(
            #     embedder, AbstractEmbModel
            # ), f"embedder model {embedder.__class__.__name__} has to inherit from AbstractEmbModel"
            embedder.is_trainable = getattr(emb_config, "is_trainable", True)
            embedder.dropout_rate = getattr(emb_config, "dropout_rate", 0.0)
            if not embedder.is_trainable:
                embedder.train = disabled_train
                for param in embedder.parameters():
                    param.requires_grad = False
                embedder.eval()

            log.info(f"Initialized embedder #{n}-{emb_name}: \n {embedder.summary()}")
            self.embedders[emb_name] = embedder

    @abstractmethod
    def forward(
        self,
        batch: Dict,
        override_dropout_rate: Optional[Dict[str, float]] = None,
    ) -> Any:
        """Should be implemented in subclasses to handle conditon datatype"""
        raise NotImplementedError

    def _forward(
        self,
        batch: Dict,
        override_dropout_rate: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """
        Processes the input batch through all configured embedders, applying conditional dropout rates if specified.
        Output tensors for each key are concatenated along the dimensions specified in KEY2DIM.

        Parameters:
            batch (Dict): The input data batch to process.
            override_dropout_rate (Optional[Dict[str, float]]): Optional dictionary to override default dropout rates
                                                                per embedder key.

        Returns:
            Dict: A dictionary of output tensors concatenated by specified dimensions.

        Note:
            In case the network code is sensitive to the order of concatenation, you can either control the order via \
            config file or make sure the embedders return a unique key for each output.
        """
        output = defaultdict(list)
        if override_dropout_rate is None:
            override_dropout_rate = {}

        # make sure emb_name in override_dropout_rate is valid
        for emb_name in override_dropout_rate.keys():
            assert emb_name in self.embedders, f"invalid name found {emb_name}"

        for emb_name, embedder in self.embedders.items():
            embedding_context = nullcontext if embedder.is_trainable else torch.no_grad
            with embedding_context():
                if isinstance(embedder.input_key, str):
                    emb_out = embedder(
                        embedder.random_dropout_input(
                            batch[embedder.input_key], override_dropout_rate.get(emb_name, None)
                        )
                    )
                elif isinstance(embedder.input_key, list):
                    emb_out = embedder(
                        *[
                            embedder.random_dropout_input(batch.get(k), override_dropout_rate.get(emb_name, None), k)
                            for k in embedder.input_key
                        ]
                    )
                else:
                    raise KeyError(
                        f"Embedder '{embedder.__class__.__name__}' requires an 'input_key' attribute to be defined as either a string or list of strings"
                    )
            for k, v in emb_out.items():
                output[k].append(v)
        # Concatenate the outputs
        return {k: torch.cat(v, dim=self.KEY2DIM.get(k, -1)) for k, v in output.items()}

    def get_condition_uncondition(
        self,
        data_batch: Dict,
    ) -> Tuple[Any, Any]:
        """
        Processes the provided data batch to generate two sets of outputs: conditioned and unconditioned. This method
        manipulates the dropout rates of embedders to simulate two scenarios — one where all conditions are applied
        (conditioned), and one where they are removed or reduced to the minimum (unconditioned).

        This method first sets the dropout rates to zero for the conditioned scenario to fully apply the embedders' effects.
        For the unconditioned scenario, it sets the dropout rates to 1 (or to 0 if the initial unconditional dropout rate
        is insignificant) to minimize the embedders' influences, simulating an unconditioned generation.

        Parameters:
            data_batch (Dict): The input data batch that contains all necessary information for embedding processing. The
                            data is expected to match the required format and keys expected by the embedders.

        Returns:
            Tuple[Any, Any]: A tuple containing two condition:
                - The first one contains the outputs with all embedders fully applied (conditioned outputs).
                - The second one contains the outputs with embedders minimized or not applied (unconditioned outputs).
        """
        cond_dropout_rates, dropout_rates = {}, {}
        for emb_name, embedder in self.embedders.items():
            cond_dropout_rates[emb_name] = 0.0
            dropout_rates[emb_name] = 1.0 if embedder.dropout_rate > 1e-4 else 0.0

        condition: Any = self(data_batch, override_dropout_rate=cond_dropout_rates)
        un_condition: Any = self(data_batch, override_dropout_rate=dropout_rates)
        return condition, un_condition


class PixelDiTConditioner(GeneralConditioner):
    """Conditioner for PixelDiT T2I models. Returns PixelDiTCondition.

    Unlike FPDConditioner which works with pre-computed tensor embeddings,
    this conditioner handles raw caption strings. The model's internal text
    encoder does the actual encoding after conditioning.

    Overrides _forward to skip torch.cat (caption outputs are list[str], not tensors).

    Embedders typically include:
      - caption: CaptionStringDrop (raw string with empty-string dropout for CFG)
    """

    def _forward(
        self,
        batch: Dict,
        override_dropout_rate: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """Like GeneralConditioner._forward but returns values directly (no torch.cat)."""
        output = {}
        if override_dropout_rate is None:
            override_dropout_rate = {}
        for emb_name, embedder in self.embedders.items():
            embedding_context = nullcontext if embedder.is_trainable else torch.no_grad
            with embedding_context():
                in_data = batch[embedder.input_key]
                in_data = embedder.random_dropout_input(in_data, override_dropout_rate.get(emb_name, None))
                emb_out = embedder(in_data)
            output.update(emb_out)
        return output

    def forward(
        self,
        batch: Dict,
        override_dropout_rate: Optional[Dict[str, float]] = None,
    ) -> PixelDiTCondition:
        output = self._forward(batch, override_dropout_rate)
        return PixelDiTCondition(**output)

    def get_condition_uncondition(self, data_batch: Dict) -> Tuple[PixelDiTCondition, PixelDiTCondition]:
        """Returns (condition, uncondition) pair for CFG inference."""
        condition = self(data_batch, override_dropout_rate=dict.fromkeys(self.embedders, 0.0))
        uncondition = self(data_batch, override_dropout_rate=dict.fromkeys(self.embedders, 1.0))
        return condition, uncondition


# =============================================================================
# PID (PixelDiT SR) — condition, embedder, and conditioner
# =============================================================================


class LQTensorDrop(AbstractEmbModel):
    """Embedder for LQ tensors (image or latent) with per-sample zero dropout.

    On dropout, the tensor is replaced with a zero tensor of the same shape.
    Supports coupled dropout: when coupled_with is set, this embedder reuses
    the dropout mask from the coupled embedder (stored in _shared_lq_keep_mask).

    Args:
        input_key: key in data_batch (e.g. "LQ_video_or_image" or "LQ_latent").
        output_key: key in condition output (e.g. "lq_video_or_image" or "lq_latent").
        dropout_rate: probability of zeroing out the tensor (for CFG training).
        is_primary: if True, this embedder generates the shared dropout mask.
            If False, it reuses the mask from the primary embedder.
    """

    # Class-level shared mask for coupled dropout (reset each forward pass)
    _shared_lq_keep_mask: Optional[torch.Tensor] = None

    def __init__(
        self,
        input_key: str = "LQ_video_or_image",
        output_key: str = "lq_video_or_image",
        dropout_rate: float = 0.0,
        is_primary: bool = True,
    ):
        super().__init__()
        self._input_key = input_key
        self._dropout_rate = dropout_rate
        self._output_key = output_key
        self._is_primary = is_primary

    def forward(self, element: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {self._output_key: element}

    def random_dropout_input(
        self, in_tensor: torch.Tensor, dropout_rate: Optional[float] = None, key: Optional[str] = None
    ) -> torch.Tensor:
        del key
        dropout_rate = dropout_rate if dropout_rate is not None else self.dropout_rate
        if dropout_rate <= 0 or in_tensor is None:
            if self._is_primary:
                LQTensorDrop._shared_lq_keep_mask = None
            return in_tensor

        B = in_tensor.shape[0]
        if self._is_primary:
            # Generate and store shared mask
            keep_mask = torch.bernoulli((1.0 - dropout_rate) * torch.ones(B, device=in_tensor.device))
            LQTensorDrop._shared_lq_keep_mask = keep_mask
        else:
            # Reuse mask from primary embedder
            keep_mask = LQTensorDrop._shared_lq_keep_mask
            if keep_mask is None:
                # Fallback: generate own mask if primary hasn't run yet
                keep_mask = torch.bernoulli((1.0 - dropout_rate) * torch.ones(B, device=in_tensor.device))

        keep_mask_expanded = keep_mask.view(B, *[1] * (in_tensor.dim() - 1)).type_as(in_tensor)
        return keep_mask_expanded * in_tensor

    def details(self) -> str:
        return f"Output key: {self._output_key}, primary: {self._is_primary}"


class PidConditioner(PixelDiTConditioner):
    """Conditioner for PID (PixelDiT SR) models. Returns PidCondition.

    Handles caption strings (CaptionStringDrop) + LQ tensors (LQTensorDrop).
    LQ image and LQ latent share coupled dropout: when one is dropped, both are.

    Inherits get_condition_uncondition from GeneralConditioner which respects
    per-embedder dropout_rate: if caption dropout_rate=0, caption is never
    dropped in uncondition (only LQ gets dropped for CFG).

    Embedders typically include:
      - caption: CaptionStringDrop (raw string dropout)
      - lq_video_or_image: LQTensorDrop (primary, generates shared mask)
      - lq_latent: LQTensorDrop (secondary, reuses shared mask)
    """

    def _forward(
        self,
        batch: Dict,
        override_dropout_rate: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """Process embedders. Handles both string (caption) and tensor (LQ) outputs."""
        output = {}
        if override_dropout_rate is None:
            override_dropout_rate = {}
        # Reset shared mask at start of each forward
        LQTensorDrop._shared_lq_keep_mask = None
        for emb_name, embedder in self.embedders.items():
            embedding_context = nullcontext if embedder.is_trainable else torch.no_grad
            with embedding_context():
                in_data = batch[embedder.input_key]
                in_data = embedder.random_dropout_input(in_data, override_dropout_rate.get(emb_name, None))
                emb_out = embedder(in_data)
            output.update(emb_out)
        return output

    def forward(
        self,
        batch: Dict,
        override_dropout_rate: Optional[Dict[str, float]] = None,
    ) -> PidCondition:
        output = self._forward(batch, override_dropout_rate)
        return PidCondition(**output)

    def get_condition_uncondition(self, data_batch: Dict) -> Tuple[PidCondition, PidCondition]:
        """Returns (condition, uncondition) pair for CFG inference.

        Respects per-embedder dropout_rate: embedders with dropout_rate=0 in config
        are NOT dropped in uncondition (e.g. caption with dropout_rate=0 stays).
        """
        cond_dropout_rates, uncond_dropout_rates = {}, {}
        for emb_name, embedder in self.embedders.items():
            cond_dropout_rates[emb_name] = 0.0
            uncond_dropout_rates[emb_name] = 1.0 if embedder.dropout_rate > 1e-4 else 0.0

        condition = self(data_batch, override_dropout_rate=cond_dropout_rates)
        uncondition = self(data_batch, override_dropout_rate=uncond_dropout_rates)
        return condition, uncondition
