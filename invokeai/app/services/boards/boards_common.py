from invokeai.app.services.board_records.board_records_common import BoardRecord


# TODO(psyche): BoardDTO is now identical to BoardRecord. We should consider removing it.
class BoardDTO(BoardRecord):
    """Deserialized board record."""

    pass
