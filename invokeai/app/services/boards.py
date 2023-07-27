from abc import ABC, abstractmethod

from logging import Logger
from invokeai.app.services.board_image_record_storage import BoardImageRecordStorageBase
from invokeai.app.services.board_images import board_record_to_dto

from invokeai.app.services.board_record_storage import (
    BoardChanges,
    BoardRecordStorageBase,
)
from invokeai.app.services.image_record_storage import (
    ImageRecordStorageBase,
    OffsetPaginatedResults,
)
from invokeai.app.services.models.board_record import BoardDTO
from invokeai.app.services.urls import UrlServiceBase


class BoardServiceABC(ABC):
    """High-level service for board management."""

    @abstractmethod
    def create(
        self,
        board_name: str,
    ) -> BoardDTO:
        """Creates a board."""
        pass

    @abstractmethod
    def get_dto(
        self,
        board_id: str,
    ) -> BoardDTO:
        """Gets a board."""
        pass

    @abstractmethod
    def update(
        self,
        board_id: str,
        changes: BoardChanges,
    ) -> BoardDTO:
        """Updates a board."""
        pass

    @abstractmethod
    def delete(
        self,
        board_id: str,
    ) -> None:
        """Deletes a board."""
        pass

    @abstractmethod
    def get_many(
        self,
        offset: int = 0,
        limit: int = 10,
    ) -> OffsetPaginatedResults[BoardDTO]:
        """Gets many boards."""
        pass

    @abstractmethod
    def get_all(
        self,
    ) -> list[BoardDTO]:
        """Gets all boards."""
        pass


class BoardServiceDependencies:
    """Service dependencies for the BoardService."""

    board_image_records: BoardImageRecordStorageBase
    board_records: BoardRecordStorageBase
    image_records: ImageRecordStorageBase
    urls: UrlServiceBase
    logger: Logger

    def __init__(
        self,
        board_image_record_storage: BoardImageRecordStorageBase,
        image_record_storage: ImageRecordStorageBase,
        board_record_storage: BoardRecordStorageBase,
        url: UrlServiceBase,
        logger: Logger,
    ):
        self.board_image_records = board_image_record_storage
        self.image_records = image_record_storage
        self.board_records = board_record_storage
        self.urls = url
        self.logger = logger


class BoardService(BoardServiceABC):
    _services: BoardServiceDependencies

    def __init__(self, services: BoardServiceDependencies):
        self._services = services

    def create(
        self,
        board_name: str,
    ) -> BoardDTO:
        board_record = self._services.board_records.save(board_name)
        return board_record_to_dto(board_record, None, 0)

    def get_dto(self, board_id: str) -> BoardDTO:
        board_record = self._services.board_records.get(board_id)
        cover_image = self._services.image_records.get_most_recent_image_for_board(board_record.board_id)
        if cover_image:
            cover_image_name = cover_image.image_name
        else:
            cover_image_name = None
        image_count = self._services.board_image_records.get_image_count_for_board(board_id)
        return board_record_to_dto(board_record, cover_image_name, image_count)

    def update(
        self,
        board_id: str,
        changes: BoardChanges,
    ) -> BoardDTO:
        board_record = self._services.board_records.update(board_id, changes)
        cover_image = self._services.image_records.get_most_recent_image_for_board(board_record.board_id)
        if cover_image:
            cover_image_name = cover_image.image_name
        else:
            cover_image_name = None

        image_count = self._services.board_image_records.get_image_count_for_board(board_id)
        return board_record_to_dto(board_record, cover_image_name, image_count)

    def delete(self, board_id: str) -> None:
        self._services.board_records.delete(board_id)

    def get_many(self, offset: int = 0, limit: int = 10) -> OffsetPaginatedResults[BoardDTO]:
        board_records = self._services.board_records.get_many(offset, limit)
        board_dtos = []
        for r in board_records.items:
            cover_image = self._services.image_records.get_most_recent_image_for_board(r.board_id)
            if cover_image:
                cover_image_name = cover_image.image_name
            else:
                cover_image_name = None

            image_count = self._services.board_image_records.get_image_count_for_board(r.board_id)
            board_dtos.append(board_record_to_dto(r, cover_image_name, image_count))

        return OffsetPaginatedResults[BoardDTO](items=board_dtos, offset=offset, limit=limit, total=len(board_dtos))

    def get_all(self) -> list[BoardDTO]:
        board_records = self._services.board_records.get_all()
        board_dtos = []
        for r in board_records:
            cover_image = self._services.image_records.get_most_recent_image_for_board(r.board_id)
            if cover_image:
                cover_image_name = cover_image.image_name
            else:
                cover_image_name = None

            image_count = self._services.board_image_records.get_image_count_for_board(r.board_id)
            board_dtos.append(board_record_to_dto(r, cover_image_name, image_count))

        return board_dtos
