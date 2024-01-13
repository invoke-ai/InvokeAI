from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Optional

from PIL.Image import Image
from pydantic import ConfigDict
from torch import Tensor

from invokeai.app.invocations.fields import ConditioningFieldData, MetadataField, WithMetadata
from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.image_records.image_records_common import ImageCategory, ImageRecordChanges, ResourceOrigin
from invokeai.app.services.images.images_common import ImageDTO
from invokeai.app.services.invocation_services import InvocationServices
from invokeai.app.services.workflow_records.workflow_records_common import WorkflowWithoutID
from invokeai.app.util.misc import uuid_string
from invokeai.app.util.step_callback import stable_diffusion_step_callback
from invokeai.backend.model_management.model_manager import ModelInfo
from invokeai.backend.model_management.models.base import BaseModelType, ModelType, SubModelType
from invokeai.backend.stable_diffusion.diffusers_pipeline import PipelineIntermediateState

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

Note: The docstrings are in weird places, but that's where they must be to get IDEs to see them.
"""


@dataclass(frozen=True)
class InvocationContextData:
    invocation: "BaseInvocation"
    session_id: str
    queue_id: str
    source_node_id: str
    queue_item_id: int
    batch_id: str
    workflow: Optional[WorkflowWithoutID] = None


class LoggerInterface:
    def __init__(self, services: InvocationServices) -> None:
        def debug(message: str) -> None:
            """
            Logs a debug message.

            :param message: The message to log.
            """
            services.logger.debug(message)

        def info(message: str) -> None:
            """
            Logs an info message.

            :param message: The message to log.
            """
            services.logger.info(message)

        def warning(message: str) -> None:
            """
            Logs a warning message.

            :param message: The message to log.
            """
            services.logger.warning(message)

        def error(message: str) -> None:
            """
            Logs an error message.

            :param message: The message to log.
            """
            services.logger.error(message)

        self.debug = debug
        self.info = info
        self.warning = warning
        self.error = error


class ImagesInterface:
    def __init__(self, services: InvocationServices, context_data: InvocationContextData) -> None:
        def save(
            image: Image,
            board_id: Optional[str] = None,
            image_category: ImageCategory = ImageCategory.GENERAL,
            metadata: Optional[MetadataField] = None,
        ) -> ImageDTO:
            """
            Saves an image, returning its DTO.

            If the current queue item has a workflow, it is automatically saved with the image.

            :param image: The image to save, as a PIL image.
            :param board_id: The board ID to add the image to, if it should be added.
            :param image_category: The category of the image. Only the GENERAL category is added to the gallery.
            :param metadata: The metadata to save with the image, if it should have any. If the invocation inherits \
                from `WithMetadata`, that metadata will be used automatically. Provide this only if you want to \
                override or provide metadata manually.
            """

            # If the invocation inherits metadata, use that. Else, use the metadata passed in.
            metadata_ = (
                context_data.invocation.metadata if isinstance(context_data.invocation, WithMetadata) else metadata
            )

            return services.images.create(
                image=image,
                is_intermediate=context_data.invocation.is_intermediate,
                image_category=image_category,
                board_id=board_id,
                metadata=metadata_,
                image_origin=ResourceOrigin.INTERNAL,
                workflow=context_data.workflow,
                session_id=context_data.session_id,
                node_id=context_data.invocation.id,
            )

        def get_pil(image_name: str) -> Image:
            """
            Gets an image as a PIL Image object.

            :param image_name: The name of the image to get.
            """
            return services.images.get_pil_image(image_name)

        def get_metadata(image_name: str) -> Optional[MetadataField]:
            """
            Gets an image's metadata, if it has any.

            :param image_name: The name of the image to get the metadata for.
            """
            return services.images.get_metadata(image_name)

        def get_dto(image_name: str) -> ImageDTO:
            """
            Gets an image as an ImageDTO object.

            :param image_name: The name of the image to get.
            """
            return services.images.get_dto(image_name)

        def update(
            image_name: str,
            board_id: Optional[str] = None,
            is_intermediate: Optional[bool] = False,
        ) -> ImageDTO:
            """
            Updates an image, returning its updated DTO.

            It is not suggested to update images saved by earlier nodes, as this can cause confusion for users.

            If you use this method, you *must* return the image as an :class:`ImageOutput` for the gallery to
            get the updated image.

            :param image_name: The name of the image to update.
            :param board_id: The board ID to add the image to, if it should be added.
            :param is_intermediate: Whether the image is an intermediate. Intermediate images aren't added to the gallery.
            """
            if is_intermediate is not None:
                services.images.update(image_name, ImageRecordChanges(is_intermediate=is_intermediate))
            if board_id is None:
                services.board_images.remove_image_from_board(image_name)
            else:
                services.board_images.add_image_to_board(image_name, board_id)
            return services.images.get_dto(image_name)

        self.save = save
        self.get_pil = get_pil
        self.get_metadata = get_metadata
        self.get_dto = get_dto
        self.update = update


class LatentsKind(str, Enum):
    IMAGE = "image"
    NOISE = "noise"
    MASK = "mask"
    MASKED_IMAGE = "masked_image"
    OTHER = "other"


class LatentsInterface:
    def __init__(
        self,
        services: InvocationServices,
        context_data: InvocationContextData,
    ) -> None:
        def save(tensor: Tensor) -> str:
            """
            Saves a latents tensor, returning its name.

            :param tensor: The latents tensor to save.
            """
            name = f"{context_data.session_id}__{context_data.invocation.id}__{uuid_string()[:7]}"
            services.latents.save(
                name=name,
                data=tensor,
            )
            return name

        def get(latents_name: str) -> Tensor:
            """
            Gets a latents tensor by name.

            :param latents_name: The name of the latents tensor to get.
            """
            return services.latents.get(latents_name)

        self.save = save
        self.get = get


class ConditioningInterface:
    def __init__(
        self,
        services: InvocationServices,
        context_data: InvocationContextData,
    ) -> None:
        def save(conditioning_data: ConditioningFieldData) -> str:
            """
            Saves a conditioning data object, returning its name.

            :param conditioning_data: The conditioning data to save.
            """
            name = f"{context_data.session_id}__{context_data.invocation.id}__{uuid_string()[:7]}__conditioning"
            services.latents.save(
                name=name,
                data=conditioning_data,  # type: ignore [arg-type]
            )
            return name

        def get(conditioning_name: str) -> ConditioningFieldData:
            """
            Gets conditioning data by name.

            :param conditioning_name: The name of the conditioning data to get.
            """
            # TODO(sm): We are (ab)using the latents storage service as a general pickle storage
            # service, but it is typed as returning tensors, so we need to ignore the type here.
            return services.latents.get(conditioning_name) # type: ignore [return-value]

        self.save = save
        self.get = get


class ModelsInterface:
    def __init__(self, services: InvocationServices, context_data: InvocationContextData) -> None:
        def exists(model_name: str, base_model: BaseModelType, model_type: ModelType) -> bool:
            """
            Checks if a model exists.

            :param model_name: The name of the model to check.
            :param base_model: The base model of the model to check.
            :param model_type: The type of the model to check.
            """
            return services.model_manager.model_exists(model_name, base_model, model_type)

        def load(
            model_name: str, base_model: BaseModelType, model_type: ModelType, submodel: Optional[SubModelType] = None
        ) -> ModelInfo:
            """
            Loads a model, returning its `ModelInfo` object.

            :param model_name: The name of the model to get.
            :param base_model: The base model of the model to get.
            :param model_type: The type of the model to get.
            :param submodel: The submodel of the model to get.
            """
            return services.model_manager.get_model(
                model_name, base_model, model_type, submodel, context_data=context_data
            )

        def get_info(model_name: str, base_model: BaseModelType, model_type: ModelType) -> dict:
            """
            Gets a model's info, an dict-like object.

            :param model_name: The name of the model to get.
            :param base_model: The base model of the model to get.
            :param model_type: The type of the model to get.
            """
            return services.model_manager.model_info(model_name, base_model, model_type)

        self.exists = exists
        self.load = load
        self.get_info = get_info


class ConfigInterface:
    def __init__(self, services: InvocationServices) -> None:
        def get() -> InvokeAIAppConfig:
            """
            Gets the app's config.
            """
            # The config can be changed at runtime. We don't want nodes doing this, so we make a
            # frozen copy..
            config = services.configuration.get_config()
            frozen_config = config.model_copy(update={"model_config": ConfigDict(frozen=True)})
            return frozen_config

        self.get = get


class UtilInterface:
    def __init__(self, services: InvocationServices, context_data: InvocationContextData) -> None:
        def sd_step_callback(
            intermediate_state: PipelineIntermediateState,
            base_model: BaseModelType,
        ) -> None:
            """
            The step callback emits a progress event with the current step, the total number of
            steps, a preview image, and some other internal metadata.

            This should be called after each step of the diffusion process.

            :param intermediate_state: The intermediate state of the diffusion pipeline.
            :param base_model: The base model for the current denoising step.
            """
            stable_diffusion_step_callback(
                context_data=context_data,
                intermediate_state=intermediate_state,
                base_model=base_model,
                invocation_queue=services.queue,
                events=services.events,
            )

        self.sd_step_callback = sd_step_callback


class InvocationContext:
    """
    The invocation context provides access to various services and data about the current invocation.
    """

    def __init__(
        self,
        images: ImagesInterface,
        latents: LatentsInterface,
        models: ModelsInterface,
        config: ConfigInterface,
        logger: LoggerInterface,
        data: InvocationContextData,
        util: UtilInterface,
        conditioning: ConditioningInterface,
    ) -> None:
        self.images = images
        "Provides methods to save, get and update images and their metadata."
        self.logger = logger
        "Provides access to the app logger."
        self.latents = latents
        "Provides methods to save and get latents tensors, including image, noise, masks, and masked images."
        self.conditioning = conditioning
        "Provides methods to save and get conditioning data."
        self.models = models
        "Provides methods to check if a model exists, get a model, and get a model's info."
        self.config = config
        "Provides access to the app's config."
        self.data = data
        "Provides data about the current queue item and invocation."
        self.util = util
        "Provides utility methods."


def build_invocation_context(
    services: InvocationServices,
    context_data: InvocationContextData,
) -> InvocationContext:
    """
    Builds the invocation context. This is a wrapper around the invocation services that provides
    a more convenient (and less dangerous) interface for nodes to use.

    :param invocation_services: The invocation services to wrap.
    :param invocation_context_data: The invocation context data.
    """

    logger = LoggerInterface(services=services)
    images = ImagesInterface(services=services, context_data=context_data)
    latents = LatentsInterface(services=services, context_data=context_data)
    models = ModelsInterface(services=services, context_data=context_data)
    config = ConfigInterface(services=services)
    util = UtilInterface(services=services, context_data=context_data)
    conditioning = ConditioningInterface(services=services, context_data=context_data)

    ctx = InvocationContext(
        images=images,
        logger=logger,
        config=config,
        latents=latents,
        models=models,
        data=context_data,
        util=util,
        conditioning=conditioning,
    )

    return ctx
