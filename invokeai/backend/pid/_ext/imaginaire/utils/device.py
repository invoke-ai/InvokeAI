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

import gc
import math
import os

from invokeai.backend.pid._ext.imaginaire.utils.log import logger as logging


def get_gpu_architecture():
    """
    Retrieves the GPU architecture of the available GPUs.

    Returns:
        str: The GPU architecture, which can be "H100", "A100", or "Other".
    """
    import pynvml

    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            model_name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(model_name, bytes):
                model_name = model_name.decode("utf-8")
            print(f"GPU {i}: Model: {model_name}")

            # Check for specific models like H100 or A100
            if "H100" in model_name or "H200" in model_name:
                return "H100"
            elif "A100" in model_name:
                return "A100"
            elif "L40S" in model_name:
                return "L40S"
            elif "B200" in model_name:
                return "B200"
    except pynvml.NVMLError as error:
        print(f"Failed to get GPU info: {error}")
    finally:
        pynvml.nvmlShutdown()

    # return "Other" incase of non hopper/ampere or error
    return "Other"


class GPUArchitectureNotSupported(Exception):
    """
    Custom exception raised when the expected GPU architecture is not supported.
    """

    pass


def print_gpu_mem(str=None):
    import pynvml

    try:
        pynvml.nvmlInit()
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0))
        logging.info(
            f"{str}: {meminfo.used / 1024 / 1024}/{meminfo.total / 1024 / 1024}MiB used ({meminfo.free / 1024 / 1024}MiB free)"
        )
    except pynvml.NVMLError as error:
        print(f"Failed to get GPU memory info: {error}")


def force_gc():
    print_gpu_mem()
    print("gc()")
    gc.collect()
    print_gpu_mem()
    print("empty cuda cache")
    # print(torch.cuda.memory_summary())
    print_gpu_mem()


def gpu0_has_80gb_or_less():
    import pynvml

    try:
        pynvml.nvmlInit()
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0))
        return meminfo.total / 1024 / 1024 / 1024 <= 80
    except pynvml.NVMLError as error:
        print(f"Failed to get GPU memory info: {error}")


class Device:
    _nvml_affinity_elements = math.ceil(os.cpu_count() / 64)  # type: ignore

    def __init__(self, device_idx: int):
        import pynvml

        super().__init__()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)

    def get_name(self) -> str:
        import pynvml

        return pynvml.nvmlDeviceGetName(self.handle)

    def get_cpu_affinity(self) -> list[int]:
        import pynvml

        affinity_string = ""
        for j in pynvml.nvmlDeviceGetCpuAffinity(self.handle, Device._nvml_affinity_elements):
            # assume nvml returns list of 64 bit ints
            affinity_string = "{:064b}".format(j) + affinity_string
        affinity_list = [int(x) for x in affinity_string]
        affinity_list.reverse()  # so core 0 is in 0th element of list
        return [i for i, e in enumerate(affinity_list) if e != 0]
