# from fastapi import Body, HTTPException, Path, Query
# from fastapi.routing import APIRouter
# from invokeai.app.services.board_record_storage import BoardRecord, BoardChanges
# from invokeai.app.services.image_record_storage import OffsetPaginatedResults

# from ..dependencies import ApiDependencies

# boards_router = APIRouter(prefix="/v1/boards", tags=["boards"])


# @boards_router.post(
#     "/",
#     operation_id="create_board",
#     responses={
#         201: {"description": "The board was created successfully"},
#     },
#     status_code=201,
# )
# async def create_board(
#     board_name: str = Body(description="The name of the board to create"),
# ):
#     """Creates a board"""
#     try:
#         result = ApiDependencies.invoker.services.boards.save(board_name=board_name)
#         return result
#     except Exception as e:
#         raise HTTPException(status_code=500, detail="Failed to create board")


# @boards_router.delete("/{board_id}", operation_id="delete_board")
# async def delete_board(
#     board_id: str = Path(description="The id of board to delete"),
# ) -> None:
#     """Deletes a board"""

#     try:
#         ApiDependencies.invoker.services.boards.delete(board_id=board_id)
#     except Exception as e:
#         # TODO: Does this need any exception handling at all?
#         pass


# @boards_router.get(
#     "/",
#     operation_id="list_boards",
#     response_model=OffsetPaginatedResults[BoardRecord],
# )
# async def list_boards(
#     offset: int = Query(default=0, description="The page offset"),
#     limit: int = Query(default=10, description="The number of boards per page"),
# ) -> OffsetPaginatedResults[BoardRecord]:
#     """Gets a list of boards"""

#     results = ApiDependencies.invoker.services.boards.get_many(
#         offset,
#         limit,
#     )

#     boards = list(
#         map(
#             lambda r: board_record_to_dto(
#                 r,
#                 generate_cover_photo_url(r.id)
#             ),
#             results.boards,
#         )
#     )

#     return boards



# def board_record_to_dto(
#     board_record: BoardRecord, cover_image_url: str
# ) -> BoardDTO:
#     """Converts an image record to an image DTO."""
#     return BoardDTO(
#         **board_record.dict(),
#         cover_image_url=cover_image_url,
#     )

# def generate_cover_photo_url(board_id: str) -> str | None:
#     cover_photo = ApiDependencies.invoker.services.images._services.records.get_board_cover_photo(board_id)
#     if cover_photo is not None:
#         url = ApiDependencies.invoker.services.images._services.urls.get_image_url(cover_photo.image_origin, cover_photo.image_name)
#         return url
