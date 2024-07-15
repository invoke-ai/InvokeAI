from invokeai.app.services.board_records.board_records_common import BoardChanges
from invokeai.app.services.boards.boards_base import BoardServiceABC
from invokeai.app.services.boards.boards_common import BoardDTO
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.shared.pagination import OffsetPaginatedResults


class BoardService(BoardServiceABC):
    __invoker: Invoker

    def start(self, invoker: Invoker) -> None:
        self.__invoker = invoker

    def create(
        self,
        board_name: str,
    ) -> BoardDTO:
        board_record = self.__invoker.services.board_records.save(board_name)
        return BoardDTO.model_validate(board_record.model_dump())

    def get_dto(self, board_id: str) -> BoardDTO:
        board_record = self.__invoker.services.board_records.get(board_id)
        return BoardDTO.model_validate(board_record.model_dump())

    def update(
        self,
        board_id: str,
        changes: BoardChanges,
    ) -> BoardDTO:
        board_record = self.__invoker.services.board_records.update(board_id, changes)
        return BoardDTO.model_validate(board_record.model_dump())

    def delete(self, board_id: str) -> None:
        self.__invoker.services.board_records.delete(board_id)

    def get_many(
        self, offset: int = 0, limit: int = 10, include_archived: bool = False
    ) -> OffsetPaginatedResults[BoardDTO]:
        board_records = self.__invoker.services.board_records.get_many(offset, limit, include_archived)
        board_dtos = [BoardDTO.model_validate(r.model_dump()) for r in board_records.items]
        return OffsetPaginatedResults[BoardDTO](items=board_dtos, offset=offset, limit=limit, total=len(board_dtos))

    def get_all(self, include_archived: bool = False) -> list[BoardDTO]:
        board_records = self.__invoker.services.board_records.get_all(include_archived)
        board_dtos = [BoardDTO.model_validate(r.model_dump()) for r in board_records]
        return board_dtos
