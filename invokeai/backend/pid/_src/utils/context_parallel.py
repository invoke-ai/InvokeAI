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


from typing import Optional

import torch
from torch import Tensor
from torch.distributed import ProcessGroup, all_gather, broadcast_object_list, get_process_group_ranks, get_world_size
from torch.distributed.utils import _verify_param_shape_across_processes

from invokeai.backend.pid._ext.imaginaire.utils import distributed


def split_inputs_cp(x: Tensor, seq_dim: int, cp_group: ProcessGroup) -> Tensor:
    """
    Split input tensor along the sequence dimension for context parallelism.

    This function divides the input tensor into equal parts along the specified
    sequence dimension, based on the number of ranks in the context parallelism group.
    It then selects the part corresponding to the current rank.

    Args:
        x: Input tensor to be split.
        seq_dim: The dimension along which to split the input (sequence dimension).
        cp_group: The process group for context parallelism.

    Returns:
        A slice of the input tensor corresponding to the current rank.

    Raises:
        AssertionError: If the sequence dimension is not divisible by the number of ranks.
    """
    cp_ranks = get_process_group_ranks(cp_group)
    cp_size = len(cp_ranks)

    assert x.shape[seq_dim] % cp_size == 0, f"{x.shape[seq_dim]} cannot divide cp_size {cp_size}"
    x = x.view(*x.shape[:seq_dim], cp_size, x.shape[seq_dim] // cp_size, *x.shape[(seq_dim + 1) :])
    seq_idx = torch.tensor([cp_group.rank()], device=x.device)
    x = x.index_select(seq_dim, seq_idx)
    # Note that the new sequence length is the original sequence length / cp_size
    x = x.view(*x.shape[:seq_dim], -1, *x.shape[(seq_dim + 2) :])
    return x


def cat_outputs_cp(x: Tensor, seq_dim: int, cp_group: ProcessGroup) -> Tensor:
    """
    Concatenate outputs from different ranks in the checkpoint parallelism group.

    This function gathers tensors from all ranks in the checkpoint parallelism group
    and concatenates them along the specified sequence dimension.

    Args:
        x: Input tensor to be concatenated.
        seq_dim: The dimension along which to concatenate the tensors (sequence dimension).
        cp_group: The process group for checkpoint parallelism.

    Returns:
        A tensor that is the concatenation of tensors from all ranks in the cp_group.

    Raises:
        RuntimeError: If the gather operation fails.
    """
    # Get the world size (number of processes in the group)
    world_size = get_world_size(cp_group)

    # Create a list to store tensors from all ranks
    gathered_tensors = [torch.zeros_like(x) for _ in range(world_size)]

    # Gather tensors from all ranks
    try:
        all_gather(gathered_tensors, x, group=cp_group)
    except RuntimeError as e:
        raise RuntimeError(f"Failed to gather tensors: {e}")

    # Concatenate the gathered tensors along the specified dimension
    return torch.cat(gathered_tensors, dim=seq_dim)


def cat_outputs_cp_with_grad(x: Tensor, seq_dim: int, cp_group: ProcessGroup) -> Tensor:
    """
    Concatenate outputs from different ranks in the context parallelism group.

    This function gathers tensors from all ranks in the checkpoint parallelism group
    and concatenates them along the specified sequence dimension.

    It retains computational graph locally for each rank by replacing the portion of the tensor with original output.

    Args:
        x: Input tensor to be concatenated.
        seq_dim: The dimension along which to concatenate the tensors (sequence dimension).
        cp_group: The process group for checkpoint parallelism.

    Returns:
        A tensor that is the concatenation of tensors from all ranks in the cp_group.

    Raises:
        RuntimeError: If the gather operation fails.
    """
    # Get the world size (number of processes in the group)
    cp_size = cp_group.size()
    assert cp_size > 0, "cp_size should be greater than 0"

    # Create a list to store tensors from all ranks
    gathered_tensors = [torch.zeros_like(x) for _ in range(cp_size)]

    # Gather tensors from all ranks
    try:
        all_gather(gathered_tensors, x, group=cp_group)
    except RuntimeError as e:
        raise RuntimeError(f"Failed to gather tensors: {e}")

    rank = cp_group.rank()
    gathered_tensors[rank] = x
    # Concatenate the gathered tensors along the specified dimension
    return torch.cat(gathered_tensors, dim=seq_dim)


def robust_broadcast(tensor: torch.Tensor, src: int, pg: ProcessGroup, is_check_shape: bool = False) -> torch.Tensor:
    """
    Perform a robust broadcast operation that works regardless of tensor shapes on different ranks.

    Args:
        tensor (torch.Tensor): The tensor to broadcast (on src rank) or receive (on other ranks).
        src (int): The source rank for the broadcast. Defaults to 0.

    Returns:
        torch.Tensor: The broadcasted tensor on all ranks.
    """
    # First, broadcast the shape of the tensor
    if distributed.get_rank() == src:
        shape = torch.tensor(tensor.shape, dtype=torch.long).cuda()
    else:
        shape = torch.empty(tensor.dim(), dtype=torch.long).cuda()
    if is_check_shape:
        _verify_param_shape_across_processes(pg, [shape])
    torch.distributed.broadcast(shape, src, group=pg)

    # Resize the tensor on non-src ranks if necessary
    if distributed.get_rank() != src:
        tensor = tensor.new_empty(shape.tolist()).type_as(tensor)

    # Now broadcast the tensor data; torch.distributed.broadcast requires contiguous tensors
    # (e.g. tensors from expand() are non-contiguous views with stride=0)
    tensor = tensor.contiguous()
    torch.distributed.broadcast(tensor, src, group=pg)

    return tensor


def broadcast(
    item: torch.Tensor | str | None, process_group: Optional[ProcessGroup] = None
) -> torch.Tensor | str | None:
    """
    Broadcast the item from the minimum rank in the specified group(s).
    """
    if process_group is None:
        return item

    min_rank = min(get_process_group_ranks(process_group))
    if isinstance(item, torch.Tensor):  # assume the device is cuda
        item = robust_broadcast(item, min_rank, process_group)
    elif item is not None:
        broadcastable_list = [item]
        broadcast_object_list(broadcastable_list, min_rank, group=process_group)
        item = broadcastable_list[0]
    return item


def broadcast_split_tensor(
    tensor: torch.Tensor,
    seq_dim: int,
    process_group: Optional[ProcessGroup] = None,
) -> torch.Tensor:
    """
    Broadcast the tensor from the minimum rank in the specified group(s).
    """
    if tensor is None:
        return tensor
    min_rank = min(get_process_group_ranks(process_group))
    tensor = robust_broadcast(tensor, min_rank, process_group)
    return split_inputs_cp(tensor, seq_dim, process_group)
