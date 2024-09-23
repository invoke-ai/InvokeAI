import typing
from enum import Enum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from platform import python_version
from typing import Optional

import torch
from fastapi import Body
from fastapi.routing import APIRouter
from pydantic import BaseModel, Field

from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.invocations.upscale import ESRGAN_MODELS
from invokeai.app.services.invocation_cache.invocation_cache_common import InvocationCacheStatus
from invokeai.backend.image_util.infill_methods.patchmatch import PatchMatch
from invokeai.backend.util.logging import logging
from invokeai.version import __version__


class LogLevel(int, Enum):
    NotSet = logging.NOTSET
    Debug = logging.DEBUG
    Info = logging.INFO
    Warning = logging.WARNING
    Error = logging.ERROR
    Critical = logging.CRITICAL


class Upscaler(BaseModel):
    upscaling_method: str = Field(description="Name of upscaling method")
    upscaling_models: list[str] = Field(description="List of upscaling models for this method")


app_router = APIRouter(prefix="/v1/app", tags=["app"])


class AppVersion(BaseModel):
    """App Version Response"""

    version: str = Field(description="App version")


class AppDependencyVersions(BaseModel):
    """App depencency Versions Response"""

    accelerate: str = Field(description="accelerate version")
    compel: str = Field(description="compel version")
    cuda: Optional[str] = Field(description="CUDA version")
    diffusers: str = Field(description="diffusers version")
    numpy: str = Field(description="Numpy version")
    opencv: str = Field(description="OpenCV version")
    onnx: str = Field(description="ONNX version")
    pillow: str = Field(description="Pillow (PIL) version")
    python: str = Field(description="Python version")
    torch: str = Field(description="PyTorch version")
    torchvision: str = Field(description="PyTorch Vision version")
    transformers: str = Field(description="transformers version")
    xformers: Optional[str] = Field(description="xformers version")


class AppConfig(BaseModel):
    """App Config Response"""

    infill_methods: list[str] = Field(description="List of available infill methods")
    upscaling_methods: list[Upscaler] = Field(description="List of upscaling methods")
    nsfw_methods: list[str] = Field(description="List of NSFW checking methods")
    watermarking_methods: list[str] = Field(description="List of invisible watermark methods")


@app_router.get("/version", operation_id="app_version", status_code=200, response_model=AppVersion)
async def get_version() -> AppVersion:
    return AppVersion(version=__version__)


@app_router.get("/app_deps", operation_id="get_app_deps", status_code=200, response_model=AppDependencyVersions)
async def get_app_deps() -> AppDependencyVersions:
    try:
        xformers = version("xformers")
    except PackageNotFoundError:
        xformers = None
    return AppDependencyVersions(
        accelerate=version("accelerate"),
        compel=version("compel"),
        cuda=torch.version.cuda,
        diffusers=version("diffusers"),
        numpy=version("numpy"),
        opencv=version("opencv-python"),
        onnx=version("onnx"),
        pillow=version("pillow"),
        python=python_version(),
        torch=torch.version.__version__,
        torchvision=version("torchvision"),
        transformers=version("transformers"),
        xformers=xformers,
    )


@app_router.get("/config", operation_id="get_config", status_code=200, response_model=AppConfig)
async def get_config() -> AppConfig:
    infill_methods = ["lama", "tile", "cv2", "color"]  # TODO: add mosaic back
    if PatchMatch.patchmatch_available():
        infill_methods.append("patchmatch")

    upscaling_models = []
    for model in typing.get_args(ESRGAN_MODELS):
        upscaling_models.append(str(Path(model).stem))
    upscaler = Upscaler(upscaling_method="esrgan", upscaling_models=upscaling_models)

    nsfw_methods = ["nsfw_checker"]

    watermarking_methods = ["invisible_watermark"]

    return AppConfig(
        infill_methods=infill_methods,
        upscaling_methods=[upscaler],
        nsfw_methods=nsfw_methods,
        watermarking_methods=watermarking_methods,
    )


@app_router.get(
    "/logging",
    operation_id="get_log_level",
    responses={200: {"description": "The operation was successful"}},
    response_model=LogLevel,
)
async def get_log_level() -> LogLevel:
    """Returns the log level"""
    return LogLevel(ApiDependencies.invoker.services.logger.level)


@app_router.post(
    "/logging",
    operation_id="set_log_level",
    responses={200: {"description": "The operation was successful"}},
    response_model=LogLevel,
)
async def set_log_level(
    level: LogLevel = Body(description="New log verbosity level"),
) -> LogLevel:
    """Sets the log verbosity level"""
    ApiDependencies.invoker.services.logger.setLevel(level)
    return LogLevel(ApiDependencies.invoker.services.logger.level)


@app_router.delete(
    "/invocation_cache",
    operation_id="clear_invocation_cache",
    responses={200: {"description": "The operation was successful"}},
)
async def clear_invocation_cache() -> None:
    """Clears the invocation cache"""
    ApiDependencies.invoker.services.invocation_cache.clear()


@app_router.put(
    "/invocation_cache/enable",
    operation_id="enable_invocation_cache",
    responses={200: {"description": "The operation was successful"}},
)
async def enable_invocation_cache() -> None:
    """Clears the invocation cache"""
    ApiDependencies.invoker.services.invocation_cache.enable()


@app_router.put(
    "/invocation_cache/disable",
    operation_id="disable_invocation_cache",
    responses={200: {"description": "The operation was successful"}},
)
async def disable_invocation_cache() -> None:
    """Clears the invocation cache"""
    ApiDependencies.invoker.services.invocation_cache.disable()


@app_router.get(
    "/invocation_cache/status",
    operation_id="get_invocation_cache_status",
    responses={200: {"model": InvocationCacheStatus}},
)
async def get_invocation_cache_status() -> InvocationCacheStatus:
    """Clears the invocation cache"""
    return ApiDependencies.invoker.services.invocation_cache.get_status()
