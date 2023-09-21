# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

from invokeai.app.invocations.metadata import CoreMetadata
from invokeai.app.invocations.primitives import ImageField

from .baseinvocation import BaseInvocation, FieldDescriptions, InputField, InvocationContext, invocation, invocation_output, OutputField, BaseInvocationOutput


@invocation_output("image_to_board_output")
class ImageToBoardOutput(BaseInvocationOutput):
    image: ImageField = OutputField(description="The image that was associated")
    board_name: str = OutputField(description="Board this image was associated with")


@invocation(
    "image_to_board",
    title="Associate Image with Board",
    tags=["primitives", "board"],
    category="primitives",
    version="1.0.0",
    use_cache=False,
)
class ImageToBoardInvocation(BaseInvocation):
    """Associates an image with a board so you view it in the board's gallery."""

    image: ImageField = InputField(description="The image to associate")
    board_name: str = InputField(description="Name of the board you'd like associate this image with")

    metadata: CoreMetadata = InputField(
        default=None,
        description=FieldDescriptions.core_metadata,
        ui_hidden=True,
    )

    def invoke(self, context: InvocationContext) -> ImageToBoardOutput:
        board_id = None
        page = 0
        page_size = 10
        while not board_id:
            boards = context.services.boards.get_many(page * page_size, page_size)
            board_id = next(board.board_id for board in boards.items if board.board_name == self.board_name)
            page += 1
            if len(boards < page_size):
                break

        context.services.board_images.add_image_to_board(
            board_id=board_id,
            image_name=self.image.image_name,
        )

        return ImageToBoardOutput(
            image=self.image,
            board_id=board_id
        )
