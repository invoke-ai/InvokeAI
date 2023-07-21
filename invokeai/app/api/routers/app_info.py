from enum import Enum
from fastapi import Body
from fastapi.routing import APIRouter
from pydantic import BaseModel, Field

from invokeai.backend.image_util.patchmatch import PatchMatch
from invokeai.version import __version__

from ..dependencies import ApiDependencies
from invokeai.backend.util.logging import logging

class LogLevel(int, Enum):
    NotSet = logging.NOTSET
    Debug = logging.DEBUG
    Info = logging.INFO
    Warning = logging.WARNING
    Error = logging.ERROR
    Critical = logging.CRITICAL
    
app_router = APIRouter(prefix="/v1/app", tags=["app"])


class AppVersion(BaseModel):
    """App Version Response"""

    version: str = Field(description="App version")


class AppConfig(BaseModel):
    """App Config Response"""

    infill_methods: list[str] = Field(description="List of available infill methods")


@app_router.get(
    "/version", operation_id="app_version", status_code=200, response_model=AppVersion
)
async def get_version() -> AppVersion:
    return AppVersion(version=__version__)


@app_router.get(
    "/config", operation_id="get_config", status_code=200, response_model=AppConfig
)
async def get_config() -> AppConfig:
    infill_methods = ['tile']
    if PatchMatch.patchmatch_available():
        infill_methods.append('patchmatch')
    return AppConfig(infill_methods=infill_methods)

@app_router.get(
    "/logging",
    operation_id="get_log_level",
    responses={200: {"description" : "The operation was successful"}},
    response_model = LogLevel,
)
async def get_log_level(
) -> LogLevel:
    """Returns the log level"""
    return LogLevel(ApiDependencies.invoker.services.logger.level)

@app_router.post(
    "/logging",
    operation_id="set_log_level",
    responses={200: {"description" : "The operation was successful"}},
    response_model = LogLevel,
)
async def set_log_level(
        level: LogLevel = Body(description="New log verbosity level"),
) -> LogLevel:
    """Sets the log verbosity level"""
    ApiDependencies.invoker.services.logger.setLevel(level)
    return LogLevel(ApiDependencies.invoker.services.logger.level)
