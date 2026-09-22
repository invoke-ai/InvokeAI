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

from typing import Any

import torch

from invokeai.backend.pid._ext.imaginaire.lazy_config import LazyDict, instantiate


class ImaginaireModel(torch.nn.Module):
    """The base model class of Imaginaire. It is inherited from torch.nn.Module.

    All models in Imaginaire should inherit ImaginaireModel. It should include the implementions for all the
    computation graphs. All inheriting child classes should implement the following methods:
    - training_step(): The training step of the model, including the loss computation.
    - validation_step(): The validation step of the model, including the loss computation.
    - forward(): The computation graph for model inference.
    The following methods have default implementations in ImaginaireModel:
    - init_optimizer_scheduler(): Creates the optimizer and scheduler for the model.
    """

    def __init__(self) -> None:
        super().__init__()

    def init_optimizer_scheduler(
        self, optimizer_config: LazyDict, scheduler_config: LazyDict
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        """Creates the optimizer and scheduler for the model.

        Args:
            config_model (ModelConfig): The config object for the model.

        Returns:
            optimizer (torch.optim.Optimizer): The model optimizer.
            scheduler (torch.optim.lr_scheduler.LRScheduler): The optimization scheduler.
        """
        optimizer_config.params = self.parameters()
        optimizer = instantiate(optimizer_config)
        scheduler_config.optimizer = optimizer
        scheduler = instantiate(scheduler_config)
        return optimizer, scheduler

    def training_step(
        self, data_batch: dict[str, torch.Tensor], iteration: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """The training step of the model, including the loss computation.

        Args:
            data (dict[str, torch.Tensor]): Data batch (dictionary of tensors).
            iteration (int): Current iteration number.

        Returns:
            output_batch (dict[str, torch.Tensor]): Auxiliary model output from the training batch.
            loss (torch.Tensor): The total loss for backprop (weighted sum of various losses).
        """
        raise NotImplementedError

    @torch.no_grad()
    def validation_step(
        self, data_batch: dict[str, torch.Tensor], iteration: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """The validation step of the model, including the loss computation.

        Args:
            data (dict[str, torch.Tensor]): Data batch (dictionary of tensors).
            iteration (int): Current iteration number.

        Returns:
            output_batch (dict[str, torch.Tensor]): Auxiliary model output from the validation batch.
            loss (torch.Tensor): The total loss (weighted sum of various losses).
        """
        raise NotImplementedError

    @torch.inference_mode()
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """The computation graph for model inference.

        Args:
            *args: Whatever you decide to pass into the forward method.
            **kwargs: Keyword arguments are also possible.

        Return:
            Your model's output.
        """
        raise NotImplementedError

    def on_train_start(self, memory_format: torch.memory_format = torch.preserve_format) -> None:
        """The model preparation before the training is launched

        Args:
            memory_format (torch.memory_format): Memory format of the model.
        """
        pass

    def on_before_zero_grad(
        self, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler, iteration: int
    ) -> None:
        """Hook before zero_grad() is called.

        Args:
            optimizer (torch.optim.Optimizer): The model optimizer.
            scheduler (torch.optim.lr_scheduler.LRScheduler): The optimization scheduler.
            iteration (int): Current iteration number.
        """
        pass

    def on_after_backward(self, iteration: int = 0) -> None:
        """Hook after loss.backward() is called.

        This method is called immediately after the backward pass, allowing for custom operations
        or modifications to be performed on the gradients before the optimizer step.

        Args:
            iteration (int): Current iteration number.
        """
        pass
