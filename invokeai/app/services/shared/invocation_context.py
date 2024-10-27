from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, Union

from PIL.Image import Image
from pydantic.networks import AnyHttpUrl
from torch import Tensor

from invokeai.app.invocations.constants import IMAGE_MODES
from invokeai.app.invocations.fields import MetadataField, WithBoard, WithMetadata
from invokeai.app.services.boards.boards_common import BoardDTO
from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.images.images_common import ImageDTO
from invokeai.app.services.invocation_services import InvocationServices
from invokeai.app.services.model_records.model_records_base import UnknownModelException
from invokeai.app.services.session_processor.session_processor_common import ProgressImage
from invokeai.app.util.step_callback import flux_step_callback, stable_diffusion_step_callback
from invokeai.backend.model_manager.config import (
    AnyModel,
    AnyModelConfig,
    BaseModelType,
    ModelFormat,
    ModelType,
    SubModelType,
)
from invokeai.backend.model_manager.load.load_base import LoadedModel, LoadedModelWithoutConfig
from invokeai.backend.stable_diffusion.diffusers_pipeline import PipelineIntermediateState
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ConditioningFieldData

if TYPE_CHECKING:
    from invokeai.app.invocations.baseinvocation import BaseInvocation
    from invokeai.app.invocations.model import ModelIdentifierField
    from invokeai.app.services.session_queue.session_queue_common import SessionQueueItem

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
    queue_item: "SessionQueueItem"
    """The queue item that is being executed."""
    invocation: "BaseInvocation"
    """The invocation that is being executed."""
    source_invocation_id: str
    """The ID of the invocation from which the currently executing invocation was prepared."""


class InvocationContextInterface:
    def __init__(self, services: InvocationServices, data: InvocationContextData) -> None:
        self._services = services
        self._data = data


class BoardsInterface(InvocationContextInterface):
    def create(self, board_name: str) -> BoardDTO:
        """Creates a board.

        Args:
            board_name: The name of the board to create.

        Returns:
            The created board DTO.
        """
        return self._services.boards.create(board_name)

    def get_dto(self, board_id: str) -> BoardDTO:
        """Gets a board DTO.

        Args:
            board_id: The ID of the board to get.

        Returns:
            The board DTO.
        """
        return self._services.boards.get_dto(board_id)

    def get_all(self) -> list[BoardDTO]:
        """Gets all boards.

        Returns:
            A list of all boards.
        """
        return self._services.boards.get_all()

    def add_image_to_board(self, board_id: str, image_name: str) -> None:
        """Adds an image to a board.

        Args:
            board_id: The ID of the board to add the image to.
            image_name: The name of the image to add to the board.
        """
        return self._services.board_images.add_image_to_board(board_id, image_name)

    def get_all_image_names_for_board(self, board_id: str) -> list[str]:
        """Gets all image names for a board.

        Args:
            board_id: The ID of the board to get the image names for.

        Returns:
            A list of all image names for the board.
        """
        return self._services.board_images.get_all_board_image_names_for_board(board_id)


class LoggerInterface(InvocationContextInterface):
    def debug(self, message: str) -> None:
        """Logs a debug message.

        Args:
            message: The message to log.
        """
        self._services.logger.debug(message)

    def info(self, message: str) -> None:
        """Logs an info message.

        Args:
            message: The message to log.
        """
        self._services.logger.info(message)

    def warning(self, message: str) -> None:
        """Logs a warning message.

        Args:
            message: The message to log.
        """
        self._services.logger.warning(message)

    def error(self, message: str) -> None:
        """Logs an error message.

        Args:
            message: The message to log.
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
        """Saves an image, returning its DTO.

        If the current queue item has a workflow or metadata, it is automatically saved with the image.

        Args:
            image: The image to save, as a PIL image.
            board_id: The board ID to add the image to, if it should be added. It the invocation \
            inherits from `WithBoard`, that board will be used automatically. **Use this only if \
            you want to override or provide a board manually!**
            image_category: The category of the image. Only the GENERAL category is added \
            to the gallery.
            metadata: The metadata to save with the image, if it should have any. If the \
            invocation inherits from `WithMetadata`, that metadata will be used automatically. \
            **Use this only if you want to override or provide metadata manually!**

        Returns:
            The saved image DTO.
        """

        # If `metadata` is provided directly, use that. Else, use the metadata provided by `WithMetadata`, falling back to None.
        metadata_ = None
        if metadata:
            metadata_ = metadata.model_dump_json()
        elif isinstance(self._data.invocation, WithMetadata) and self._data.invocation.metadata:
            metadata_ = self._data.invocation.metadata.model_dump_json()

        # If `board_id` is provided directly, use that. Else, use the board provided by `WithBoard`, falling back to None.
        board_id_ = None
        if board_id:
            board_id_ = board_id
        elif isinstance(self._data.invocation, WithBoard) and self._data.invocation.board:
            board_id_ = self._data.invocation.board.board_id

        workflow_ = None
        if self._data.queue_item.workflow:
            workflow_ = self._data.queue_item.workflow.model_dump_json()

        graph_ = None
        if self._data.queue_item.session.graph:
            graph_ = self._data.queue_item.session.graph.model_dump_json()

        return self._services.images.create(
            image=image,
            is_intermediate=self._data.invocation.is_intermediate,
            image_category=image_category,
            board_id=board_id_,
            metadata=metadata_,
            image_origin=ResourceOrigin.INTERNAL,
            workflow=workflow_,
            graph=graph_,
            session_id=self._data.queue_item.session_id,
            node_id=self._data.invocation.id,
        )

    def get_pil(self, image_name: str, mode: IMAGE_MODES | None = None) -> Image:
        """Gets an image as a PIL Image object. This method returns a copy of the image.

        Args:
            image_name: The name of the image to get.
            mode: The color mode to convert the image to. If None, the original mode is used.

        Returns:
            The image as a PIL Image object.
        """
        image = self._services.images.get_pil_image(image_name)
        if mode and mode != image.mode:
            try:
                # convert makes a copy!
                image = image.convert(mode)
            except ValueError:
                self._services.logger.warning(
                    f"Could not convert image from {image.mode} to {mode}. Using original mode instead."
                )
        else:
            # copy the image to prevent the user from modifying the original
            image = image.copy()
        return image

    def get_metadata(self, image_name: str) -> Optional[MetadataField]:
        """Gets an image's metadata, if it has any.

        Args:
            image_name: The name of the image to get the metadata for.

        Returns:
            The image's metadata, if it has any.
        """
        return self._services.images.get_metadata(image_name)

    def get_dto(self, image_name: str) -> ImageDTO:
        """Gets an image as an ImageDTO object.

        Args:
            image_name: The name of the image to get.

        Returns:
            The image as an ImageDTO object.
        """
        return self._services.images.get_dto(image_name)

    def get_path(self, image_name: str, thumbnail: bool = False) -> Path:
        """Gets the internal path to an image or thumbnail.

        Args:
            image_name: The name of the image to get the path of.
            thumbnail: Get the path of the thumbnail instead of the full image

        Returns:
            The local path of the image or thumbnail.
        """
        return self._services.images.get_path(image_name, thumbnail)


class TensorsInterface(InvocationContextInterface):
    def save(self, tensor: Tensor) -> str:
        """Saves a tensor, returning its name.

        Args:
            tensor: The tensor to save.

        Returns:
            The name of the saved tensor.
        """

        name = self._services.tensors.save(obj=tensor)
        return name

    def load(self, name: str) -> Tensor:
        """Loads a tensor by name. This method returns a copy of the tensor.

        Args:
            name: The name of the tensor to load.

        Returns:
            The tensor.
        """
        return self._services.tensors.load(name).clone()


class ConditioningInterface(InvocationContextInterface):
    def save(self, conditioning_data: ConditioningFieldData) -> str:
        """Saves a conditioning data object, returning its name.

        Args:
            conditioning_data: The conditioning data to save.

        Returns:
            The name of the saved conditioning data.
        """

        name = self._services.conditioning.save(obj=conditioning_data)
        return name

    def load(self, name: str) -> ConditioningFieldData:
        """Loads conditioning data by name. This method returns a copy of the conditioning data.

        Args:
            name: The name of the conditioning data to load.

        Returns:
            The conditioning data.
        """

        return deepcopy(self._services.conditioning.load(name))


class ModelsInterface(InvocationContextInterface):
    """Common API for loading, downloading and managing models."""

    def exists(self, identifier: Union[str, "ModelIdentifierField"]) -> bool:
        """Check if a model exists.

        Args:
            identifier: The key or ModelField representing the model.

        Returns:
            True if the model exists, False if not.
        """
        if isinstance(identifier, str):
            return self._services.model_manager.store.exists(identifier)
        else:
            return self._services.model_manager.store.exists(identifier.key)

    def load(
        self, identifier: Union[str, "ModelIdentifierField"], submodel_type: Optional[SubModelType] = None
    ) -> LoadedModel:
        """Load a model.

        Args:
            identifier: The key or ModelField representing the model.
            submodel_type: The submodel of the model to get.

        Returns:
            An object representing the loaded model.
        """

        # The model manager emits events as it loads the model. It needs the context data to build
        # the event payloads.

        if isinstance(identifier, str):
            model = self._services.model_manager.store.get_model(identifier)
            return self._services.model_manager.load.load_model(model, submodel_type)
        else:
            _submodel_type = submodel_type or identifier.submodel_type
            model = self._services.model_manager.store.get_model(identifier.key)
            return self._services.model_manager.load.load_model(model, _submodel_type)

    def load_by_attrs(
        self, name: str, base: BaseModelType, type: ModelType, submodel_type: Optional[SubModelType] = None
    ) -> LoadedModel:
        """Load a model by its attributes.

        Args:
            name: Name of the model.
            base: The models' base type, e.g. `BaseModelType.StableDiffusion1`, `BaseModelType.StableDiffusionXL`, etc.
            type: Type of the model, e.g. `ModelType.Main`, `ModelType.Vae`, etc.
            submodel_type: The type of submodel to load, e.g. `SubModelType.UNet`, `SubModelType.TextEncoder`, etc. Only main
            models have submodels.

        Returns:
            An object representing the loaded model.
        """

        configs = self._services.model_manager.store.search_by_attr(model_name=name, base_model=base, model_type=type)
        if len(configs) == 0:
            raise UnknownModelException(f"No model found with name {name}, base {base}, and type {type}")

        if len(configs) > 1:
            raise ValueError(f"More than one model found with name {name}, base {base}, and type {type}")

        return self._services.model_manager.load.load_model(configs[0], submodel_type)

    def get_config(self, identifier: Union[str, "ModelIdentifierField"]) -> AnyModelConfig:
        """Get a model's config.

        Args:
            identifier: The key or ModelField representing the model.

        Returns:
            The model's config.
        """
        if isinstance(identifier, str):
            return self._services.model_manager.store.get_model(identifier)
        else:
            return self._services.model_manager.store.get_model(identifier.key)

    def search_by_path(self, path: Path) -> list[AnyModelConfig]:
        """Search for models by path.

        Args:
            path: The path to search for.

        Returns:
            A list of models that match the path.
        """
        return self._services.model_manager.store.search_by_path(path)

    def search_by_attrs(
        self,
        name: Optional[str] = None,
        base: Optional[BaseModelType] = None,
        type: Optional[ModelType] = None,
        format: Optional[ModelFormat] = None,
    ) -> list[AnyModelConfig]:
        """Search for models by attributes.

        Args:
            name: The name to search for (exact match).
            base: The base to search for, e.g. `BaseModelType.StableDiffusion1`, `BaseModelType.StableDiffusionXL`, etc.
            type: Type type of model to search for, e.g. `ModelType.Main`, `ModelType.Vae`, etc.
            format: The format of model to search for, e.g. `ModelFormat.Checkpoint`, `ModelFormat.Diffusers`, etc.

        Returns:
            A list of models that match the attributes.
        """

        return self._services.model_manager.store.search_by_attr(
            model_name=name,
            base_model=base,
            model_type=type,
            model_format=format,
        )

    def download_and_cache_model(
        self,
        source: str | AnyHttpUrl,
    ) -> Path:
        """
        Download the model file located at source to the models cache and return its Path.

        This can be used to single-file install models and other resources of arbitrary types
        which should not get registered with the database. If the model is already
        installed, the cached path will be returned. Otherwise it will be downloaded.

        Args:
            source: A URL that points to the model, or a huggingface repo_id.

        Returns:
            Path to the downloaded model
        """
        return self._services.model_manager.install.download_and_cache_model(source=source)

    def load_local_model(
        self,
        model_path: Path,
        loader: Optional[Callable[[Path], AnyModel]] = None,
    ) -> LoadedModelWithoutConfig:
        """
        Load the model file located at the indicated path

        If a loader callable is provided, it will be invoked to load the model. Otherwise,
        `safetensors.torch.load_file()` or `torch.load()` will be called to load the model.

        Be aware that the LoadedModelWithoutConfig object has no `config` attribute

        Args:
            path: A model Path
            loader: A Callable that expects a Path and returns a dict[str|int, Any]

        Returns:
            A LoadedModelWithoutConfig object.
        """
        return self._services.model_manager.load.load_model_from_path(model_path=model_path, loader=loader)

    def load_remote_model(
        self,
        source: str | AnyHttpUrl,
        loader: Optional[Callable[[Path], AnyModel]] = None,
    ) -> LoadedModelWithoutConfig:
        """
        Download, cache, and load the model file located at the indicated URL or repo_id.

        If the model is already downloaded, it will be loaded from the cache.

        If the a loader callable is provided, it will be invoked to load the model. Otherwise,
        `safetensors.torch.load_file()` or `torch.load()` will be called to load the model.

        Be aware that the LoadedModelWithoutConfig object has no `config` attribute

        Args:
            source: A URL or huggingface repoid.
            loader: A Callable that expects a Path and returns a dict[str|int, Any]

        Returns:
            A LoadedModelWithoutConfig object.
        """
        model_path = self._services.model_manager.install.download_and_cache_model(source=str(source))
        return self._services.model_manager.load.load_model_from_path(model_path=model_path, loader=loader)


class ConfigInterface(InvocationContextInterface):
    def get(self) -> InvokeAIAppConfig:
        """Gets the app's config.

        Returns:
            The app's config.
        """

        return self._services.configuration


class UtilInterface(InvocationContextInterface):
    def __init__(
        self, services: InvocationServices, data: InvocationContextData, is_canceled: Callable[[], bool]
    ) -> None:
        super().__init__(services, data)
        self._is_canceled = is_canceled

    def is_canceled(self) -> bool:
        """Checks if the current session has been canceled.

        Returns:
            True if the current session has been canceled, False if not.
        """
        return self._is_canceled()

    def sd_step_callback(self, intermediate_state: PipelineIntermediateState, base_model: BaseModelType) -> None:
        """
        The step callback emits a progress event with the current step, the total number of
        steps, a preview image, and some other internal metadata.

        This should be called after each denoising step.

        Args:
            intermediate_state: The intermediate state of the diffusion pipeline.
            base_model: The base model for the current denoising step.
        """

        stable_diffusion_step_callback(
            signal_progress=self.signal_progress,
            intermediate_state=intermediate_state,
            base_model=base_model,
            is_canceled=self.is_canceled,
        )

    def flux_step_callback(self, intermediate_state: PipelineIntermediateState) -> None:
        """
        The step callback emits a progress event with the current step, the total number of
        steps, a preview image, and some other internal metadata.

        This should be called after each denoising step.

        Args:
            intermediate_state: The intermediate state of the diffusion pipeline.
        """

        flux_step_callback(
            signal_progress=self.signal_progress,
            intermediate_state=intermediate_state,
            is_canceled=self.is_canceled,
        )

    def signal_progress(
        self,
        message: str,
        percentage: float | None = None,
        image: Image | None = None,
        image_size: tuple[int, int] | None = None,
    ) -> None:
        """Signals the progress of some long-running invocation. The progress is displayed in the UI.

        If a percentage is provided, the UI will display a progress bar and automatically append the percentage to the
        message. You should not include the percentage in the message.

        Example:
            ```py
            total_steps = 10
            for i in range(total_steps):
                percentage = i / (total_steps - 1)
                context.util.signal_progress("Doing something cool", percentage)
            ```

        If an image is provided, the UI will display it. If your image should be displayed at a different size, provide
        a tuple of `(width, height)` for the `image_size` parameter. The image will be displayed at the specified size
        in the UI.

        For example, SD denoising progress images are 1/8 the size of the original image, so you'd do this to ensure the
        image is displayed at the correct size:
            ```py
            # Calculate the output size of the image (8x the progress image's size)
            width = progress_image.width * 8
            height = progress_image.height * 8
            # Signal the progress with the image and output size
            signal_progress("Denoising", percentage, progress_image, (width, height))
            ```

        If your progress image is very large, consider downscaling it to reduce the payload size and provide the original
        size to the `image_size` parameter. The PIL `thumbnail` method is useful for this, as it maintains the aspect
        ratio of the image:
            ```py
            # `thumbnail` modifies the image in-place, so we need to first make a copy
            thumbnail_image = progress_image.copy()
            # Resize the image to a maximum of 256x256 pixels, maintaining the aspect ratio
            thumbnail_image.thumbnail((256, 256))
            # Signal the progress with the thumbnail, passing the original size
            signal_progress("Denoising", percentage, thumbnail, progress_image.size)
            ```

        Args:
            message: A message describing the current status. Do not include the percentage in this message.
            percentage: The current percentage completion for the process. Omit for indeterminate progress.
            image: An optional image to display.
            image_size: The optional size of the image to display. If omitted, the image will be displayed at its
                original size.
        """

        self._services.events.emit_invocation_progress(
            queue_item=self._data.queue_item,
            invocation=self._data.invocation,
            message=message,
            percentage=percentage,
            image=ProgressImage.build(image, image_size) if image else None,
        )


class InvocationContext:
    """Provides access to various services and data for the current invocation.

    Attributes:
        images (ImagesInterface): Methods to save, get and update images and their metadata.
        tensors (TensorsInterface): Methods to save and get tensors, including image, noise, masks, and masked images.
        conditioning (ConditioningInterface): Methods to save and get conditioning data.
        models (ModelsInterface): Methods to check if a model exists, get a model, and get a model's info.
        logger (LoggerInterface): The app logger.
        config (ConfigInterface): The app config.
        util (UtilInterface): Utility methods, including a method to check if an invocation was canceled and step callbacks.
        boards (BoardsInterface): Methods to interact with boards.
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
        data: InvocationContextData,
        services: InvocationServices,
    ) -> None:
        self.images = images
        """Methods to save, get and update images and their metadata."""
        self.tensors = tensors
        """Methods to save and get tensors, including image, noise, masks, and masked images."""
        self.conditioning = conditioning
        """Methods to save and get conditioning data."""
        self.models = models
        """Methods to check if a model exists, get a model, and get a model's info."""
        self.logger = logger
        """The app logger."""
        self.config = config
        """The app config."""
        self.util = util
        """Utility methods, including a method to check if an invocation was canceled and step callbacks."""
        self.boards = boards
        """Methods to interact with boards."""
        self._data = data
        """An internal API providing access to data about the current queue item and invocation. You probably shouldn't use this. It may change without warning."""
        self._services = services
        """An internal API providing access to all application services. You probably shouldn't use this. It may change without warning."""


def build_invocation_context(
    services: InvocationServices,
    data: InvocationContextData,
    is_canceled: Callable[[], bool],
) -> InvocationContext:
    """Builds the invocation context for a specific invocation execution.

    Args:
        services: The invocation services to wrap.
        data: The invocation context data.

    Returns:
        The invocation context.
    """

    logger = LoggerInterface(services=services, data=data)
    images = ImagesInterface(services=services, data=data)
    tensors = TensorsInterface(services=services, data=data)
    models = ModelsInterface(services=services, data=data)
    config = ConfigInterface(services=services, data=data)
    util = UtilInterface(services=services, data=data, is_canceled=is_canceled)
    conditioning = ConditioningInterface(services=services, data=data)
    boards = BoardsInterface(services=services, data=data)

    ctx = InvocationContext(
        images=images,
        logger=logger,
        config=config,
        tensors=tensors,
        models=models,
        data=data,
        util=util,
        conditioning=conditioning,
        services=services,
        boards=boards,
    )

    return ctx
