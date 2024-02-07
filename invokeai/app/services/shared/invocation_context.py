from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from deprecated import deprecated
from PIL.Image import Image
from torch import Tensor

from invokeai.app.invocations.fields import MetadataField, WithBoard, WithMetadata
from invokeai.app.services.boards.boards_common import BoardDTO
from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.images.images_common import ImageDTO
from invokeai.app.services.invocation_services import InvocationServices
from invokeai.app.services.workflow_records.workflow_records_common import WorkflowWithoutID
from invokeai.app.util.step_callback import stable_diffusion_step_callback
from invokeai.backend.model_management.model_manager import ModelInfo
from invokeai.backend.model_management.models.base import BaseModelType, ModelType, SubModelType
from invokeai.backend.stable_diffusion.diffusers_pipeline import PipelineIntermediateState
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ConditioningFieldData

if TYPE_CHECKING:
    from invokeai.app.invocations.baseinvocation import BaseInvocation

"""
The InvocationContext provides access to various services and data about the current invocation.

We do not provide the invocation services directly, as their methods are both dangerous and
inconvenient to use.

For example:
- The `images` service allows nodes to delete or unsafely modify existing images.
- The `configuration` service allows nodes to change the app's config at runtime.
- The `events` service allows nodes to emit arbitrary events.

Wrapping these services provides a simpler and safer interface for nodes to use.

When a node executes, a fresh `InvocationContext` is built for it, ensuring nodes cannot interfere
with each other.

Many of the wrappers have the same signature as the methods they wrap. This allows us to write
user-facing docstrings and not need to go and update the internal services to match.

Note: The docstrings are in weird places, but that's where they must be to get IDEs to see them.
"""


@dataclass
class InvocationContextData:
    invocation: "BaseInvocation"
    """The invocation that is being executed."""
    session_id: str
    """The session that is being executed."""
    queue_id: str
    """The queue in which the session is being executed."""
    source_node_id: str
    """The ID of the node from which the currently executing invocation was prepared."""
    queue_item_id: int
    """The ID of the queue item that is being executed."""
    batch_id: str
    """The ID of the batch that is being executed."""
    workflow: Optional[WorkflowWithoutID] = None
    """The workflow associated with this queue item, if any."""


class InvocationContextInterface:
    def __init__(self, services: InvocationServices, context_data: InvocationContextData) -> None:
        self._services = services
        self._context_data = context_data


class BoardsInterface(InvocationContextInterface):
    def create(self, board_name: str) -> BoardDTO:
        """
        Creates a board.

        :param board_name: The name of the board to create.
        """
        return self._services.boards.create(board_name)

    def get_dto(self, board_id: str) -> BoardDTO:
        """
        Gets a board DTO.

        :param board_id: The ID of the board to get.
        """
        return self._services.boards.get_dto(board_id)

    def get_all(self) -> list[BoardDTO]:
        """
        Gets all boards.
        """
        return self._services.boards.get_all()

    def add_image_to_board(self, board_id: str, image_name: str) -> None:
        """
        Adds an image to a board.

        :param board_id: The ID of the board to add the image to.
        :param image_name: The name of the image to add to the board.
        """
        return self._services.board_images.add_image_to_board(board_id, image_name)

    def get_all_image_names_for_board(self, board_id: str) -> list[str]:
        """
        Gets all image names for a board.

        :param board_id: The ID of the board to get the image names for.
        """
        return self._services.board_images.get_all_board_image_names_for_board(board_id)


class LoggerInterface(InvocationContextInterface):
    def debug(self, message: str) -> None:
        """
        Logs a debug message.

        :param message: The message to log.
        """
        self._services.logger.debug(message)

    def info(self, message: str) -> None:
        """
        Logs an info message.

        :param message: The message to log.
        """
        self._services.logger.info(message)

    def warning(self, message: str) -> None:
        """
        Logs a warning message.

        :param message: The message to log.
        """
        self._services.logger.warning(message)

    def error(self, message: str) -> None:
        """
        Logs an error message.

        :param message: The message to log.
        """
        self._services.logger.error(message)


class ImagesInterface(InvocationContextInterface):
    def save(
        self,
        image: Image,
        board_id: Optional[str] = None,
        image_category: ImageCategory = ImageCategory.GENERAL,
        metadata: Optional[MetadataField] = None,
    ) -> ImageDTO:
        """
        Saves an image, returning its DTO.

        If the current queue item has a workflow or metadata, it is automatically saved with the image.

        :param image: The image to save, as a PIL image.
        :param board_id: The board ID to add the image to, if it should be added. It the invocation \
            inherits from `WithBoard`, that board will be used automatically. **Use this only if \
            you want to override or provide a board manually!**
        :param image_category: The category of the image. Only the GENERAL category is added \
            to the gallery.
        :param metadata: The metadata to save with the image, if it should have any. If the \
            invocation inherits from `WithMetadata`, that metadata will be used automatically. \
            **Use this only if you want to override or provide metadata manually!**
        """

        # If the invocation inherits metadata, use that. Else, use the metadata passed in.
        metadata_ = (
            self._context_data.invocation.metadata
            if isinstance(self._context_data.invocation, WithMetadata)
            else metadata
        )

        # If the invocation inherits WithBoard, use that. Else, use the board_id passed in.
        board_ = self._context_data.invocation.board if isinstance(self._context_data.invocation, WithBoard) else None
        board_id_ = board_.board_id if board_ is not None else board_id

        return self._services.images.create(
            image=image,
            is_intermediate=self._context_data.invocation.is_intermediate,
            image_category=image_category,
            board_id=board_id_,
            metadata=metadata_,
            image_origin=ResourceOrigin.INTERNAL,
            workflow=self._context_data.workflow,
            session_id=self._context_data.session_id,
            node_id=self._context_data.invocation.id,
        )

    def get_pil(self, image_name: str) -> Image:
        """
        Gets an image as a PIL Image object.

        :param image_name: The name of the image to get.
        """
        return self._services.images.get_pil_image(image_name)

    def get_metadata(self, image_name: str) -> Optional[MetadataField]:
        """
        Gets an image's metadata, if it has any.

        :param image_name: The name of the image to get the metadata for.
        """
        return self._services.images.get_metadata(image_name)

    def get_dto(self, image_name: str) -> ImageDTO:
        """
        Gets an image as an ImageDTO object.

        :param image_name: The name of the image to get.
        """
        return self._services.images.get_dto(image_name)


class TensorsInterface(InvocationContextInterface):
    def save(self, tensor: Tensor) -> str:
        """
        Saves a tensor, returning its name.

        :param tensor: The tensor to save.
        """

        name = self._services.tensors.set(item=tensor)
        return name

    def get(self, tensor_name: str) -> Tensor:
        """
        Gets a tensor by name.

        :param tensor_name: The name of the tensor to get.
        """
        return self._services.tensors.get(tensor_name)


class ConditioningInterface(InvocationContextInterface):
    def save(self, conditioning_data: ConditioningFieldData) -> str:
        """
        Saves a conditioning data object, returning its name.

        :param conditioning_context_data: The conditioning data to save.
        """

        name = self._services.conditioning.set(item=conditioning_data)
        return name

    def get(self, conditioning_name: str) -> ConditioningFieldData:
        """
        Gets conditioning data by name.

        :param conditioning_name: The name of the conditioning data to get.
        """

        return self._services.conditioning.get(conditioning_name)


class ModelsInterface(InvocationContextInterface):
    def exists(self, model_name: str, base_model: BaseModelType, model_type: ModelType) -> bool:
        """
        Checks if a model exists.

        :param model_name: The name of the model to check.
        :param base_model: The base model of the model to check.
        :param model_type: The type of the model to check.
        """
        return self._services.model_manager.model_exists(model_name, base_model, model_type)

    def load(
        self, model_name: str, base_model: BaseModelType, model_type: ModelType, submodel: Optional[SubModelType] = None
    ) -> ModelInfo:
        """
        Loads a model, returning its `ModelInfo` object.

        :param model_name: The name of the model to get.
        :param base_model: The base model of the model to get.
        :param model_type: The type of the model to get.
        :param submodel: The submodel of the model to get.
        """

        # During this call, the model manager emits events with model loading status. The model
        # manager itself has access to the events services, but does not have access to the
        # required metadata for the events.
        #
        # For example, it needs access to the node's ID so that the events can be associated
        # with the execution of a specific node.
        #
        # While this is available within the node, it's tedious to need to pass it in on every
        # call. We can avoid that by wrapping the method here.

        return self._services.model_manager.get_model(
            model_name, base_model, model_type, submodel, context_data=self._context_data
        )

    def get_info(self, model_name: str, base_model: BaseModelType, model_type: ModelType) -> dict:
        """
        Gets a model's info, an dict-like object.

        :param model_name: The name of the model to get.
        :param base_model: The base model of the model to get.
        :param model_type: The type of the model to get.
        """
        return self._services.model_manager.model_info(model_name, base_model, model_type)


class ConfigInterface(InvocationContextInterface):
    def get(self) -> InvokeAIAppConfig:
        """Gets the app's config."""

        return self._services.configuration.get_config()


class UtilInterface(InvocationContextInterface):
    def sd_step_callback(self, intermediate_state: PipelineIntermediateState, base_model: BaseModelType) -> None:
        """
        The step callback emits a progress event with the current step, the total number of
        steps, a preview image, and some other internal metadata.

        This should be called after each denoising step.

        :param intermediate_state: The intermediate state of the diffusion pipeline.
        :param base_model: The base model for the current denoising step.
        """

        # The step callback needs access to the events and the invocation queue services, but this
        # represents a dangerous level of access.
        #
        # We wrap the step callback so that nodes do not have direct access to these services.

        stable_diffusion_step_callback(
            context_data=self._context_data,
            intermediate_state=intermediate_state,
            base_model=base_model,
            invocation_queue=self._services.queue,
            events=self._services.events,
        )


deprecation_version = "3.7.0"
removed_version = "3.8.0"


def get_deprecation_reason(property_name: str, alternative: Optional[str] = None) -> str:
    msg = f"{property_name} is deprecated as of v{deprecation_version}. It will be removed in v{removed_version}."
    if alternative is not None:
        msg += f" Use {alternative} instead."
    msg += " See PLACEHOLDER_URL for details."
    return msg


# Deprecation docstrings template. I don't think we can implement these programmatically with
# __doc__ because the IDE won't see them.

"""
**DEPRECATED as of v3.7.0**

PROPERTY_NAME will be removed in v3.8.0. Use ALTERNATIVE instead. See PLACEHOLDER_URL for details.

OG_DOCSTRING
"""


class InvocationContext:
    """
    The `InvocationContext` provides access to various services and data for the current invocation.
    """

    def __init__(
        self,
        images: ImagesInterface,
        tensors: TensorsInterface,
        conditioning: ConditioningInterface,
        models: ModelsInterface,
        logger: LoggerInterface,
        config: ConfigInterface,
        util: UtilInterface,
        boards: BoardsInterface,
        context_data: InvocationContextData,
        services: InvocationServices,
    ) -> None:
        self.images = images
        """Provides methods to save, get and update images and their metadata."""
        self.tensors = tensors
        """Provides methods to save and get tensors, including image, noise, masks, and masked images."""
        self.conditioning = conditioning
        """Provides methods to save and get conditioning data."""
        self.models = models
        """Provides methods to check if a model exists, get a model, and get a model's info."""
        self.logger = logger
        """Provides access to the app logger."""
        self.config = config
        """Provides access to the app's config."""
        self.util = util
        """Provides utility methods."""
        self.boards = boards
        """Provides methods to interact with boards."""
        self._data = context_data
        """Provides data about the current queue item and invocation. This is an internal API and may change without warning."""
        self._services = services
        """Provides access to the full application services. This is an internal API and may change without warning."""

    @property
    @deprecated(version=deprecation_version, reason=get_deprecation_reason("`context.services`"))
    def services(self) -> InvocationServices:
        """
        **DEPRECATED as of v3.7.0**

        `context.services` will be removed in v3.8.0. See PLACEHOLDER_URL for details.

        The invocation services.
        """
        return self._services

    @property
    @deprecated(
        version=deprecation_version,
        reason=get_deprecation_reason("`context.graph_execution_state_id", "`context._data.session_id`"),
    )
    def graph_execution_state_id(self) -> str:
        """
        **DEPRECATED as of v3.7.0**

        `context.graph_execution_state_api` will be removed in v3.8.0. Use `context._data.session_id` instead. See PLACEHOLDER_URL for details.

        The ID of the session (aka graph execution state).
        """
        return self._data.session_id

    @property
    @deprecated(
        version=deprecation_version,
        reason=get_deprecation_reason("`context.queue_id`", "`context._data.queue_id`"),
    )
    def queue_id(self) -> str:
        """
        **DEPRECATED as of v3.7.0**

        `context.queue_id` will be removed in v3.8.0. Use `context._data.queue_id` instead. See PLACEHOLDER_URL for details.

        The ID of the queue.
        """
        return self._data.queue_id

    @property
    @deprecated(
        version=deprecation_version,
        reason=get_deprecation_reason("`context.queue_item_id`", "`context._data.queue_item_id`"),
    )
    def queue_item_id(self) -> int:
        """
        **DEPRECATED as of v3.7.0**

        `context.queue_item_id` will be removed in v3.8.0. Use `context._data.queue_item_id` instead. See PLACEHOLDER_URL for details.

        The ID of the queue item.
        """
        return self._data.queue_item_id

    @property
    @deprecated(
        version=deprecation_version,
        reason=get_deprecation_reason("`context.queue_batch_id`", "`context._data.batch_id`"),
    )
    def queue_batch_id(self) -> str:
        """
        **DEPRECATED as of v3.7.0**

        `context.queue_batch_id` will be removed in v3.8.0. Use `context._data.batch_id` instead. See PLACEHOLDER_URL for details.

        The ID of the batch.
        """
        return self._data.batch_id

    @property
    @deprecated(
        version=deprecation_version,
        reason=get_deprecation_reason("`context.workflow`", "`context._data.workflow`"),
    )
    def workflow(self) -> Optional[WorkflowWithoutID]:
        """
        **DEPRECATED as of v3.7.0**

        `context.workflow` will be removed in v3.8.0. Use `context._data.workflow` instead. See PLACEHOLDER_URL for details.

        The workflow associated with this queue item, if any.
        """
        return self._data.workflow


def build_invocation_context(
    services: InvocationServices,
    context_data: InvocationContextData,
) -> InvocationContext:
    """
    Builds the invocation context for a specific invocation execution.

    :param invocation_services: The invocation services to wrap.
    :param invocation_context_data: The invocation context data.
    """

    logger = LoggerInterface(services=services, context_data=context_data)
    images = ImagesInterface(services=services, context_data=context_data)
    tensors = TensorsInterface(services=services, context_data=context_data)
    models = ModelsInterface(services=services, context_data=context_data)
    config = ConfigInterface(services=services, context_data=context_data)
    util = UtilInterface(services=services, context_data=context_data)
    conditioning = ConditioningInterface(services=services, context_data=context_data)
    boards = BoardsInterface(services=services, context_data=context_data)

    ctx = InvocationContext(
        images=images,
        logger=logger,
        config=config,
        tensors=tensors,
        models=models,
        context_data=context_data,
        util=util,
        conditioning=conditioning,
        services=services,
        boards=boards,
    )

    return ctx
