from fastapi.routing import APIRouter
from pydantic import BaseModel
from invokeai.backend.image_util.patchmatch import PatchMatch

from invokeai.version import __version__

app_router = APIRouter(prefix="/v1/app", tags=['app'])


class AppVersion(BaseModel):
    """App Version Response"""
    version: str

class AppConfig(BaseModel):
    """App Config Response"""
    patchmatch_enabled: bool


@app_router.get('/version', operation_id="app_version",
                status_code=200,
                response_model=AppVersion)
async def get_version() -> AppVersion:
    return AppVersion(version=__version__)

@app_router.get('/config', operation_id="get_config",
                status_code=200,
                response_model=AppConfig)
async def get_config() -> AppConfig:
    return AppConfig(patchmatch_enabled=PatchMatch.patchmatch_available())
