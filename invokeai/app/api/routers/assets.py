from fastapi import APIRouter, HTTPException, Path
from fastapi.responses import FileResponse

from invokeai.app.api.dependencies import ApiDependencies

ASSET_MAX_AGE = 31536000

assets_router = APIRouter(prefix="/v1/assets", tags=["assets"])


@assets_router.get(
    "/i/{asset_name}",
    operation_id="get_asset",
    responses={
        200: {"description": "The 3D asset was fetched successfully"},
        404: {"description": "The 3D asset could not be found"},
    },
    status_code=200,
)
async def get_asset(
    asset_name: str = Path(description="The name of the 3D asset file to get"),
) -> FileResponse:
    """Gets a 3D asset file (e.g. a Gaussian-splat .ply)."""
    try:
        path = ApiDependencies.invoker.services.asset_files.get_path(asset_name)
        response = FileResponse(
            path,
            media_type="application/octet-stream",
            filename=asset_name,
            content_disposition_type="inline",
        )
        response.headers["Cache-Control"] = f"max-age={ASSET_MAX_AGE}"
        return response
    except Exception:
        raise HTTPException(status_code=404)
