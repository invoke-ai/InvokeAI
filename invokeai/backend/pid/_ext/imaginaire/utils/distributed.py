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

import collections
import collections.abc
import ctypes
import functools
import os
from contextlib import contextmanager
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Callable, Container, Optional

import torch
import torch.distributed as dist
from torch.distributed import get_process_group_ranks

from invokeai.backend.pid._ext.imaginaire.utils.device import Device

if dist.is_available():
    from torch.distributed.distributed_c10d import _get_default_group
    from torch.distributed.utils import _sync_module_states, _verify_param_shape_across_processes

from invokeai.backend.pid._ext.imaginaire.utils import log

if TYPE_CHECKING:
    DDPConfig = Any  # config module not vendored; type hint kept for parity

try:
    from megatron.core import parallel_state
except ImportError:
    parallel_state = None  # type: ignore[assignment]


def init() -> int | None:
    """Initialize distributed training."""
    import pynvml

    if dist.is_initialized():
        return torch.cuda.current_device()

    # Set GPU affinity.
    pynvml.nvmlInit()
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    try:
        device = Device(local_rank)
        os.sched_setaffinity(0, device.get_cpu_affinity())
    except Exception as e:
        log.warning(f"Failed to set device affinity: {e}")
    # Set up NCCL communication.
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "0"
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    if dist.is_available():
        torch.cuda.set_device(local_rank)
        # Get the timeout value from environment variable
        timeout_seconds = os.getenv("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", 1800)
        # Convert the timeout to an integer (if it isn't already) and then to a timedelta
        timeout_timedelta = timedelta(seconds=int(timeout_seconds))
        dist.init_process_group(backend="nccl", init_method="env://", timeout=timeout_timedelta)
        log.info(
            f"Initialized distributed training with local rank {local_rank} with timeout {timeout_seconds}",
            rank0_only=False,
        )
    # Increase the L2 fetch granularity for faster speed.
    _libcudart = ctypes.CDLL("libcudart.so")
    # Set device limit on the current device.
    p_value = ctypes.cast((ctypes.c_int * 1)(), ctypes.POINTER(ctypes.c_int))
    _libcudart.cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))
    _libcudart.cudaDeviceGetLimit(p_value, ctypes.c_int(0x05))
    log.info(f"Training with {get_world_size()} GPUs.")


def get_rank(group: Optional[dist.ProcessGroup] = None) -> int:
    """Get the rank (GPU device) of the worker.

    Returns:
        rank (int): The rank of the worker.
    """
    rank = 0
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank(group)
    return rank


def get_world_size(group: Optional[dist.ProcessGroup] = None) -> int:
    """Get world size. How many GPUs are available in this job.

    Returns:
        world_size (int): The total number of GPUs available in this job.
    """
    world_size = 1
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size(group)
    return world_size


def is_rank0() -> bool:
    """Check if current process is the master GPU.

    Returns:
        (bool): True if this function is called from the master GPU, else False.
    """
    return get_rank() == 0


def is_local_rank0() -> bool:
    """Check if current process is the local master GPU in the current node.

    Returns:
        (bool): True if this function is called from the local master GPU, else False.
    """
    return torch.cuda.current_device() == 0


def rank0_only(func: Callable) -> Callable:
    """Apply this function only to the master GPU.

    Example usage:
        @rank0_only
        def func(x):
            return x + 3

    Args:
        func (Callable): a function.

    Returns:
        (Callable): A function wrapper executing the function only on the master GPU.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):  # noqa: ANN202
        if is_rank0():
            return func(*args, **kwargs)
        else:
            return None

    return wrapper


def barrier() -> None:
    """Barrier for all GPUs."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def rank0_first(func: Callable) -> Callable:
    """run the function on rank 0 first, then on other ranks."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):  # noqa: ANN202
        if is_rank0():
            result = func(*args, **kwargs)
        barrier()
        if not is_rank0():
            result = func(*args, **kwargs)
        return result

    return wrapper


def parallel_model_wrapper(config_ddp: DDPConfig, model: torch.nn.Module) -> torch.nn.Module | DistributedDataParallel:
    """Wraps the model to enable data parallalism for training across multiple GPU devices.

    Args:
        config_ddp (DDPConfig): The data parallel config.
        model (torch.nn.Module): The PyTorch module.

    Returns:
        model (torch.nn.Module | DistributedDataParallel): The data parallel model wrapper
            if distributed environment is available, otherwise return the original model.
    """
    if dist.is_available() and dist.is_initialized():
        local_rank = int(os.getenv("LOCAL_RANK", 0))
        try:
            ddp_group = parallel_state.get_data_parallel_group(with_context_parallel=True)
        except Exception as e:
            log.info(e)
            log.info("parallel_state not initialized, treating all GPUs equally for DDP")
            ddp_group = None

        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=config_ddp.find_unused_parameters,
            static_graph=config_ddp.static_graph,
            broadcast_buffers=config_ddp.broadcast_buffers,
            process_group=ddp_group,
        )
    return model


class DistributedDataParallel(torch.nn.parallel.DistributedDataParallel):
    """This extends torch.nn.parallel.DistributedDataParallel with .training_step().

    This borrows the concept of `forward-redirection` from Pytorch lightning. It wraps an ImaginaireModel such that
    model.training_step() would be executed when calling self.training_step(), while preserving the behavior of calling
    model() for Pytorch modules. Internally, this is a double rerouting mechanism (training_step -> forward ->
    training_step), allowing us to preserve the function names and signatures.
    """

    def __init__(self, model: torch.nn.Module, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.show_sync_grad_static_graph_warning = True

    def training_step(self, *args, **kwargs) -> Any:
        # Cache the original model.forward() method.
        original_forward = self.module.forward

        def wrapped_training_step(*_args, **_kwargs):  # noqa: ANN202
            # Unpatch immediately before calling training_step() because itself may want to call the real forward.
            self.module.forward = original_forward
            # The actual .training_step().
            return self.module.training_step(*_args, **_kwargs)

        # Patch the original_module's forward so we can redirect the arguments back to the real method.
        self.module.forward = wrapped_training_step
        # Call self, which implicitly calls self.forward() --> model.forward(), which is now model.training_step().
        # Without calling self.forward() or model.forward() explciitly, implicit hooks are also executed.
        return self(*args, **kwargs)


@contextmanager
def ddp_sync_grad(model, enabled):
    r"""
    Context manager to enable/disable gradient synchronizations across DDP processes for DDP model.
    Modified from:
    https://pytorch.org/docs/stable/_modules/torch/nn/parallel/distributed.html#DistributedDataParallel.no_sync
    Note that this is incompatible with static_graph=True and will be an no-op if static_graph=True.

    Within this context, gradients will be accumulated on module
    variables, which will later be synchronized in the first
    forward-backward pass exiting the context.

    .. warning::
        The forward pass should be included inside the context manager, or
        else gradients will still be synchronized.
    """
    assert isinstance(model, torch.nn.Module)
    if isinstance(model, DistributedDataParallel):
        old_require_backward_grad_sync = model.require_backward_grad_sync
        if model.static_graph and model.require_backward_grad_sync != enabled:
            if model.show_sync_grad_static_graph_warning:
                log.warning("DDP static_graph=True is incompatible with sync_grad(). Performance will be reduced.")
                model.show_sync_grad_static_graph_warning = False
        else:
            model.require_backward_grad_sync = enabled
    try:
        yield
    finally:
        if isinstance(model, DistributedDataParallel):
            model.require_backward_grad_sync = old_require_backward_grad_sync


def collate_batches(data_batches: list[dict[str, torch.Tensor]]) -> torch.Tensor | dict[str, torch.Tensor]:
    """Aggregate the list of data batches from all devices and process the results.

    This is used for gathering validation data batches with pid._ext.imaginaire.utils.dataloader.DistributedEvalSampler.
    It will return the data/output of the entire validation set in its original index order. The sizes of data_batches
    in different ranks may differ by 1 (if dataset size is not evenly divisible), in which case a dummy sample will be
    created before calling dis.all_gather().

    Args:
        data_batches (list[dict[str, torch.Tensor]]): List of tensors or (hierarchical) dictionary where
            leaf entries are tensors.

    Returns:
        data_gather (torch.Tensor | dict[str, torch.Tensor]): tensors or (hierarchical) dictionary where
            leaf entries are concatenated tensors.
    """
    if isinstance(data_batches[0], torch.Tensor):
        # Concatenate the local data batches.
        data_concat = torch.cat(data_batches, dim=0)  # type: ignore
        # Get the largest number of local samples from all ranks to determine whether to dummy-pad on this rank.
        max_num_local_samples = torch.tensor(len(data_concat), device="cuda")
        dist.all_reduce(max_num_local_samples, op=dist.ReduceOp.MAX)
        if len(data_concat) < max_num_local_samples:
            assert len(data_concat) + 1 == max_num_local_samples
            dummy = torch.empty_like(data_concat[:1])
            data_concat = torch.cat([data_concat, dummy], dim=0)
            dummy_count = torch.tensor(1, device="cuda")
        else:
            dummy_count = torch.tensor(0, device="cuda")
        # Get all concatenated batches from all ranks and concatenate again.
        dist.all_reduce(dummy_count, op=dist.ReduceOp.SUM)
        data_concat = all_gather_tensor(data_concat.contiguous())
        data_collate = torch.stack(data_concat, dim=1).flatten(start_dim=0, end_dim=1)
        # Remove the dummy samples.
        if dummy_count > 0:
            data_collate = data_collate[:-dummy_count]
    elif isinstance(data_batches[0], collections.abc.Mapping):
        data_collate = {}
        for key in data_batches[0].keys():
            data_collate[key] = collate_batches([data[key] for data in data_batches])  # type: ignore
    else:
        raise TypeError
    return data_collate


@torch.no_grad()
def all_gather_tensor(tensor: torch.Tensor) -> list[torch.Tensor]:
    """Gather the corresponding tensor from all GPU devices to a list.

    Args:
        tensor (torch.Tensor): Pytorch tensor.

    Returns:
        tensor_list (list[torch.Tensor]): A list of Pytorch tensors gathered from all GPU devices.
    """
    tensor_list = [torch.zeros_like(tensor) for _ in range(get_world_size())]
    dist.all_gather(tensor_list, tensor)
    return tensor_list


def broadcast(tensor, src, group=None, async_op=False):
    world_size = get_world_size()
    if world_size < 2:
        return tensor
    dist.broadcast(tensor, src=src, group=group, async_op=async_op)


def dist_reduce_tensor(tensor, rank=0, reduce="mean"):
    r"""Reduce to rank 0"""
    world_size = get_world_size()
    if world_size < 2:
        return tensor
    with torch.no_grad():
        dist.reduce(tensor, dst=rank)
        if get_rank() == rank:
            if reduce == "mean":
                tensor /= world_size
            elif reduce == "sum":
                pass
            else:
                raise NotImplementedError
    return tensor


def sync_model_states(
    model: torch.nn.Module,
    process_group: Optional[dist.ProcessGroup] = None,
    src: int = 0,
    params_and_buffers_to_ignore: Optional[Container[str]] = None,
    broadcast_buffers: bool = True,
):
    """
    Modify based on DDP source code
    Synchronizes the parameters and buffers of a model across different processes in a distributed setting.

    This function ensures that all processes in the specified process group have the same initial parameters and
    buffers from the source rank, typically rank 0. It is useful when different processes start with different model
    states and a synchronization is required to ensure consistency across all ranks.

    Args:
        model (nn.Module): The model whose parameters and buffers are to be synchronized.
        process_group (dist.ProcessGroup, optional): The process group for communication. If None,
            the default group is used. Defaults to None.
        src (int, optional): The source rank from which parameters and buffers will be broadcasted.
            Defaults to 0.
        params_and_buffers_to_ignore (Optional[Container[str]], optional): A container of parameter and buffer
            names to exclude from synchronization. Defaults to None, which means all parameters and buffers are
            included.
        broadcast_buffers (bool, optional): Whether to broadcast buffers or not. Defaults to True.

    Side Effects:
        This function modifies the state of the model in-place to synchronize it with the source rank's model state.

    Raises:
        RuntimeError: If the shapes of parameters across processes do not match, a runtime error will be raised.

    Examples:
        >>> # downloading duplicated model weights from s3 in each rank and save network bandwidth
        >>> # useful and save our time when model weights are huge
        >>> if dist.get_rank == 0:
        >>>     model.load_state_dict(network_bound_weights_download_fn(s3_weights_path))
        >>> dist.barrir()
        >>> sync_model_states(model) # sync rank0 weights to other ranks
    """
    if not dist.is_available() or not dist.is_initialized():
        return
    if process_group is None:
        process_group = _get_default_group()
    if not params_and_buffers_to_ignore:
        params_and_buffers_to_ignore = set()

    log.info(
        f"Synchronizing model states from rank {src} to all ranks in process group {get_process_group_ranks(process_group)}."
    )

    # Build tuple of (module, parameter) for all parameters that require grads.
    modules_and_parameters = [
        (module, parameter)
        for module_name, module in model.named_modules()
        for parameter in [
            param
            # Note that we access module.named_parameters instead of
            # parameters(module). parameters(module) is only needed in the
            # single-process multi device case, where it accesses replicated
            # parameters through _former_parameters.
            for param_name, param in module.named_parameters(recurse=False)
            if f"{module_name}.{param_name}" not in params_and_buffers_to_ignore
            # if param.requires_grad
            # and f"{module_name}.{param_name}" not in params_and_buffers_to_ignore
        ]
    ]

    # Deduplicate any parameters that might be shared across child modules.
    memo = set()
    modules_and_parameters = [
        # "p not in memo" is the deduplication check.
        # "not memo.add(p)" is always True, and it's only there to cause "add(p)" if needed.
        (m, p)
        for m, p in modules_and_parameters
        if p not in memo and not memo.add(p)  # type: ignore[func-returns-value]
    ]

    # Build list of parameters.
    parameters = [parameter for _, parameter in modules_and_parameters]
    if len(parameters) == 0:
        return

    _verify_param_shape_across_processes(process_group, parameters)

    _sync_module_states(
        module=model,
        process_group=process_group,
        broadcast_bucket_size=int(250 * 1024 * 1024),
        src=src,
        params_and_buffers_to_ignore=params_and_buffers_to_ignore,
        broadcast_buffers=broadcast_buffers,
    )
