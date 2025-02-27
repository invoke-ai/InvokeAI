from invokeai.app.services.board_records.board_records_common import BoardChanges, BoardRecordOrderBy
from invokeai.app.services.boards.boards_base import BoardServiceABC
from invokeai.app.services.boards.boards_common import BoardDTO, board_record_to_dto
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.shared.pagination import OffsetPaginatedResults
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection


class BoardService(BoardServiceABC):
    __invoker: Invoker

    def start(self, invoker: Invoker) -> None:
        self.__invoker = invoker

    def create(
        self,
        board_name: str,
    ) -> BoardDTO:
        board_record = self.__invoker.services.board_records.save(board_name)
        return board_record_to_dto(board_record, None, 0)

    def get_dto(self, board_id: str) -> BoardDTO:
        board_record = self.__invoker.services.board_records.get(board_id)
        cover_image = self.__invoker.services.image_records.get_most_recent_image_for_board(board_record.board_id)
        if cover_image:
            cover_image_name = cover_image.image_name
        else:
            cover_image_name = None
        image_count = self.__invoker.services.board_image_records.get_image_count_for_board(board_id)
        return board_record_to_dto(board_record, cover_image_name, image_count)

    def update(
        self,
        board_id: str,
        changes: BoardChanges,
    ) -> BoardDTO:
        board_record = self.__invoker.services.board_records.update(board_id, changes)
        cover_image = self.__invoker.services.image_records.get_most_recent_image_for_board(board_record.board_id)
        if cover_image:
            cover_image_name = cover_image.image_name
        else:
            cover_image_name = None

        image_count = self.__invoker.services.board_image_records.get_image_count_for_board(board_id)
        return board_record_to_dto(board_record, cover_image_name, image_count)

    def delete(self, board_id: str) -> None:
        self.__invoker.services.board_records.delete(board_id)

    def get_many(
        self,
        order_by: BoardRecordOrderBy,
        direction: SQLiteDirection,
        offset: int = 0,
        limit: int = 10,
        include_archived: bool = False,
    ) -> OffsetPaginatedResults[BoardDTO]:
        board_records = self.__invoker.services.board_records.get_many(
            order_by, direction, offset, limit, include_archived
        )
        board_dtos = []
        for r in board_records.items:
            cover_image = self.__invoker.services.image_records.get_most_recent_image_for_board(r.board_id)
            if cover_image:
                cover_image_name = cover_image.image_name
            else:
                cover_image_name = None

            image_count = self.__invoker.services.board_image_records.get_image_count_for_board(r.board_id)
            board_dtos.append(board_record_to_dto(r, cover_image_name, image_count))

        return OffsetPaginatedResults[BoardDTO](items=board_dtos, offset=offset, limit=limit, total=len(board_dtos))

    def get_all(
        self, order_by: BoardRecordOrderBy, direction: SQLiteDirection, include_archived: bool = False
    ) -> list[BoardDTO]:
        board_records = self.__invoker.services.board_records.get_all(order_by, direction, include_archived)
        board_dtos = []
        for r in board_records:
            cover_image = self.__invoker.services.image_records.get_most_recent_image_for_board(r.board_id)
            if cover_image:
                cover_image_name = cover_image.image_name
            else:
                cover_image_name = None

            image_count = self.__invoker.services.board_image_records.get_image_count_for_board(r.board_id)
            board_dtos.append(board_record_to_dto(r, cover_image_name, image_count))

        return board_dtos
