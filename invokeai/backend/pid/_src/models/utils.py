# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import torch
from safetensors.torch import load as safetensors_torch_load

from invokeai.backend.pid._ext.imaginaire.utils.easy_io import easy_io


def load_state_dict_from_safetensors(file_path, torch_dtype=None, s3_credential_path=None):
    backend_args = (
        {"backend": "s3", "s3_credential_path": s3_credential_path} if file_path.startswith("s3://") else None
    )
    byte_stream = easy_io.load(file_path, backend_args=backend_args, file_format="byte")
    return safetensors_torch_load(byte_stream)


def load_state_dict_from_folder(file_path, torch_dtype=None):
    state_dict = {}
    for file_name in os.listdir(file_path):
        if "." in file_name and file_name.split(".")[-1] in ["safetensors", "bin", "ckpt", "pth", "pt"]:
            state_dict.update(load_state_dict(os.path.join(file_path, file_name), torch_dtype=torch_dtype))
    return state_dict


def load_state_dict_from_bin(file_path, torch_dtype=None, s3_credential_path=None):
    backend_args = (
        {"backend": "s3", "s3_credential_path": s3_credential_path} if file_path.startswith("s3://") else None
    )
    state_dict = easy_io.load(
        file_path, backend_args=backend_args, file_format="pt", map_location="cpu", weights_only=False
    )
    if torch_dtype is not None:
        for i in state_dict:
            if isinstance(state_dict[i], torch.Tensor):
                state_dict[i] = state_dict[i].to(torch_dtype)
    return state_dict


def load_state_dict(file_path, torch_dtype=None, s3_credential_path=None):
    if file_path.endswith(".safetensors"):
        return load_state_dict_from_safetensors(
            file_path, torch_dtype=torch_dtype, s3_credential_path=s3_credential_path
        )
    return load_state_dict_from_bin(file_path, torch_dtype=torch_dtype, s3_credential_path=s3_credential_path)
