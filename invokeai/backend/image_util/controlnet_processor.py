"""Utilities for processing images with ControlNet processors."""

from datetime import datetime
from typing import Any, Optional

from invokeai.app.invocations.fields import ImageField
from invokeai.app.services.invoker import InvocationServices
from invokeai.app.services.session_queue.session_queue_common import SessionQueueItem
from invokeai.app.services.shared.graph import Graph, GraphExecutionState
from invokeai.app.services.shared.invocation_context import InvocationContextData, build_invocation_context


def _get_processor_invocation_class(processor_type: str):
    """Get the invocation class for a processor type."""
    # Import processor invocation classes on demand
    processor_class_map = {
        "canny_image_processor": lambda: __import__(
            "invokeai.app.invocations.canny", fromlist=["CannyEdgeDetectionInvocation"]
        ).CannyEdgeDetectionInvocation,
        "hed_image_processor": lambda: __import__(
            "invokeai.app.invocations.hed", fromlist=["HEDEdgeDetectionInvocation"]
        ).HEDEdgeDetectionInvocation,
        "mlsd_image_processor": lambda: __import__(
            "invokeai.app.invocations.mlsd", fromlist=["MLSDDetectionInvocation"]
        ).MLSDDetectionInvocation,
        "depth_anything_image_processor": lambda: __import__(
            "invokeai.app.invocations.depth_anything", fromlist=["DepthAnythingDepthEstimationInvocation"]
        ).DepthAnythingDepthEstimationInvocation,
        "normalbae_image_processor": lambda: __import__(
            "invokeai.app.invocations.normal_bae", fromlist=["NormalMapInvocation"]
        ).NormalMapInvocation,
        "pidi_image_processor": lambda: __import__(
            "invokeai.app.invocations.pidi", fromlist=["PiDiNetEdgeDetectionInvocation"]
        ).PiDiNetEdgeDetectionInvocation,
        "lineart_image_processor": lambda: __import__(
            "invokeai.app.invocations.lineart", fromlist=["LineartEdgeDetectionInvocation"]
        ).LineartEdgeDetectionInvocation,
        "lineart_anime_image_processor": lambda: __import__(
            "invokeai.app.invocations.lineart_anime", fromlist=["LineartAnimeEdgeDetectionInvocation"]
        ).LineartAnimeEdgeDetectionInvocation,
        "content_shuffle_image_processor": lambda: __import__(
            "invokeai.app.invocations.content_shuffle", fromlist=["ContentShuffleInvocation"]
        ).ContentShuffleInvocation,
        "dw_openpose_image_processor": lambda: __import__(
            "invokeai.app.invocations.dw_openpose", fromlist=["DWOpenposeDetectionInvocation"]
        ).DWOpenposeDetectionInvocation,
        "mediapipe_face_processor": lambda: __import__(
            "invokeai.app.invocations.mediapipe_face", fromlist=["MediaPipeFaceDetectionInvocation"]
        ).MediaPipeFaceDetectionInvocation,
        # Note: zoe_depth_image_processor doesn't have a processor invocation implementation
        "color_map_image_processor": lambda: __import__(
            "invokeai.app.invocations.color_map", fromlist=["ColorMapInvocation"]
        ).ColorMapInvocation,
    }

    if processor_type in processor_class_map:
        return processor_class_map[processor_type]()
    return None


# Map processor type names to their default parameters
PROCESSOR_DEFAULT_PARAMS = {
    "canny_image_processor": {"low_threshold": 100, "high_threshold": 200},
    "hed_image_processor": {"scribble": False},
    "mlsd_image_processor": {"detect_resolution": 512, "thr_v": 0.1, "thr_d": 0.1},
    "depth_anything_image_processor": {"model_size": "small"},
    "normalbae_image_processor": {"detect_resolution": 512},
    "pidi_image_processor": {"detect_resolution": 512, "safe": False},
    "lineart_image_processor": {"detect_resolution": 512, "coarse": False},
    "lineart_anime_image_processor": {"detect_resolution": 512},
    "content_shuffle": {},
    "dw_openpose_image_processor": {"draw_body": True, "draw_face": True, "draw_hands": True},
    "mediapipe_face_processor": {"max_faces": 1, "min_confidence": 0.5},
    "zoe_depth_image_processor": {},
    "color_map_image_processor": {"color_map_tile_size": 64},
}


def process_controlnet_image(image_name: str, model_key: str, services: InvocationServices) -> Optional[dict[str, Any]]:
    """
    Process a controlnet image using the appropriate processor based on the model's default settings.

    Args:
        image_name: The filename of the image to process
        model_key: The model key to look up default processor settings
        services: The invocation services providing access to models and images

    Returns:
        A dictionary with the processed image data (image_name, width, height) or None if processing fails
    """
    logger = services.logger

    try:
        # Get model config to find default processor
        model_record = services.model_manager.store.get_model(model_key)
        if not model_record or not model_record.default_settings:
            logger.info(f"No default processor settings found for model {model_key}")
            return None

        preprocessor = model_record.default_settings.preprocessor
        if not preprocessor:
            logger.info(f"No preprocessor configured for model {model_key}")
            return None

        # Get the invocation class for this processor
        invocation_class = _get_processor_invocation_class(preprocessor)
        if not invocation_class:
            logger.info(f"No processor mapping found for preprocessor '{preprocessor}'")
            return None

        # Get default parameters for this processor
        default_params = PROCESSOR_DEFAULT_PARAMS.get(preprocessor, {})
        logger.info(f"Processing image {image_name} with processor {preprocessor}")

        # Create a minimal context to run the invocation
        # We need a fake queue item and session for the context
        fake_session = GraphExecutionState(graph=Graph())
        now = datetime.now()

        # Create invocation instance first so we have its ID
        invocation_params = {"image": ImageField(image_name=image_name), **default_params}
        invocation = invocation_class(**invocation_params)

        # Add the invocation ID to the session's prepared_source_mapping
        # This is required for the invocation context to emit progress events
        fake_session.prepared_source_mapping[invocation.id] = invocation.id

        fake_queue_item = SessionQueueItem(
            item_id=0,
            session_id=fake_session.id,
            queue_id="default",
            batch_id="recall_processor",
            field_values=None,
            session=fake_session,
            status="in_progress",
            created_at=now,
            updated_at=now,
            started_at=now,
            completed_at=None,
        )

        context_data = InvocationContextData(
            invocation=invocation,
            source_invocation_id=invocation.id,
            queue_item=fake_queue_item,
        )

        context = build_invocation_context(
            data=context_data,
            services=services,
            is_canceled=lambda: False,
        )

        # Invoke the processor
        output = invocation.invoke(context)

        # Get the processed image DTO
        processed_image_dto = services.images.get_dto(output.image.image_name)

        logger.info(f"Successfully processed image {image_name} -> {processed_image_dto.image_name}")

        return {
            "image_name": processed_image_dto.image_name,
            "width": processed_image_dto.width,
            "height": processed_image_dto.height,
        }

    except Exception as e:
        logger.error(f"Error processing controlnet image {image_name}: {e}", exc_info=True)
        return None
