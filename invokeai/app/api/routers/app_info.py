from fastapi.routing import APIRouter
from pydantic import BaseModel, Field

from invokeai.backend.image_util.patchmatch import PatchMatch
from invokeai.version import __version__

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
