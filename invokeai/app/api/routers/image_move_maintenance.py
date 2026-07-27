from fastapi import HTTPException, status

from invokeai.app.api.dependencies import ApiDependencies

IMAGE_MOVE_MAINTENANCE_ACTIVE_DETAIL = "Image storage maintenance is active"


def assert_image_move_maintenance_inactive() -> None:
    invoker = getattr(ApiDependencies, "invoker", None)
    if invoker is None:
        return
    image_moves = getattr(invoker.services, "image_moves", None)
    if image_moves is not None and image_moves.is_maintenance_active():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=IMAGE_MOVE_MAINTENANCE_ACTIVE_DETAIL,
        )
