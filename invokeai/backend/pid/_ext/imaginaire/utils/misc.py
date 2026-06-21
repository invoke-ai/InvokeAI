# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Minimal stand-in for the upstream misc module. The full module pulled in
# wandb / straggler / termcolor / easy_io / DTensor helpers that the decoder
# inference subset does not use.

from __future__ import annotations

import random
import time
from contextlib import contextmanager
from typing import Iterator

import numpy as np
import torch

from invokeai.backend.pid._ext.imaginaire.utils.log import logger


@contextmanager
def timer(label: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        logger.info("%s took %.2fs", label, time.perf_counter() - start)


def set_random_seed(seed: int, by_rank: bool = False) -> None:
    if by_rank:
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                seed = seed + dist.get_rank()
        except Exception:
            pass
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def requires_grad(model: torch.nn.Module, value: bool = True) -> None:
    for p in model.parameters():
        p.requires_grad = value
