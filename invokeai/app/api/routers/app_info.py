from fastapi.routing import APIRouter
from pydantic import BaseModel

from invokeai.version import __version__

app_router = APIRouter(prefix="/v1/app", tags=['app'])


class AppVersion(BaseModel):
    """App Version Response"""
    version: str


@app_router.get('/version', operation_id="app_version",
                status_code=200,
                response_model=AppVersion)
async def get_version() -> AppVersion:
    return AppVersion(version=__version__)
