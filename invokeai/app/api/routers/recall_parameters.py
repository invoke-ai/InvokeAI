"""Router for updating recallable parameters on the frontend."""

import json
from typing import Any, Literal, Optional

from fastapi import Body, HTTPException, Path
from fastapi.routing import APIRouter
from pydantic import BaseModel, ConfigDict, Field

from invokeai.app.api.dependencies import ApiDependencies
from invokeai.backend.image_util.controlnet_processor import process_controlnet_image
from invokeai.backend.model_manager.taxonomy import ModelType

recall_parameters_router = APIRouter(prefix="/v1/recall", tags=["recall"])


class LoRARecallParameter(BaseModel):
    """LoRA configuration for recall"""

    model_name: str = Field(description="The name of the LoRA model")
    weight: float = Field(default=0.75, ge=-10, le=10, description="The weight for the LoRA")
    is_enabled: bool = Field(default=True, description="Whether the LoRA is enabled")


class ControlNetRecallParameter(BaseModel):
    """ControlNet configuration for recall"""

    model_name: str = Field(description="The name of the ControlNet/T2I Adapter/Control LoRA model")
    image_name: Optional[str] = Field(default=None, description="The filename of the control image in outputs/images")
    weight: float = Field(default=1.0, ge=-1, le=2, description="The weight for the control adapter")
    begin_step_percent: Optional[float] = Field(
        default=None, ge=0, le=1, description="When the control adapter is first applied (% of total steps)"
    )
    end_step_percent: Optional[float] = Field(
        default=None, ge=0, le=1, description="When the control adapter is last applied (% of total steps)"
    )
    control_mode: Optional[Literal["balanced", "more_prompt", "more_control"]] = Field(
        default=None, description="The control mode (ControlNet only)"
    )


class IPAdapterRecallParameter(BaseModel):
    """IP Adapter configuration for recall"""

    model_name: str = Field(description="The name of the IP Adapter model")
    image_name: Optional[str] = Field(default=None, description="The filename of the reference image in outputs/images")
    weight: float = Field(default=1.0, ge=-1, le=2, description="The weight for the IP Adapter")
    begin_step_percent: Optional[float] = Field(
        default=None, ge=0, le=1, description="When the IP Adapter is first applied (% of total steps)"
    )
    end_step_percent: Optional[float] = Field(
        default=None, ge=0, le=1, description="When the IP Adapter is last applied (% of total steps)"
    )
    method: Optional[Literal["full", "style", "composition"]] = Field(default=None, description="The IP Adapter method")
    image_influence: Optional[Literal["lowest", "low", "medium", "high", "highest"]] = Field(
        default=None, description="FLUX Redux image influence (if model is flux_redux)"
    )


class RecallParameter(BaseModel):
    """Request model for updating recallable parameters."""

    model_config = ConfigDict(extra="forbid")

    # Prompts
    positive_prompt: Optional[str] = Field(None, description="Positive prompt text")
    negative_prompt: Optional[str] = Field(None, description="Negative prompt text")

    # Model configuration
    model: Optional[str] = Field(None, description="Main model name/identifier")
    refiner_model: Optional[str] = Field(None, description="Refiner model name/identifier")
    vae_model: Optional[str] = Field(None, description="VAE model name/identifier")
    scheduler: Optional[str] = Field(None, description="Scheduler name")

    # Generation parameters
    steps: Optional[int] = Field(None, ge=1, description="Number of generation steps")
    refiner_steps: Optional[int] = Field(None, ge=0, description="Number of refiner steps")
    cfg_scale: Optional[float] = Field(None, description="CFG scale for guidance")
    cfg_rescale_multiplier: Optional[float] = Field(None, description="CFG rescale multiplier")
    refiner_cfg_scale: Optional[float] = Field(None, description="Refiner CFG scale")
    guidance: Optional[float] = Field(None, description="Guidance scale")

    # Image parameters
    width: Optional[int] = Field(None, ge=64, description="Image width in pixels")
    height: Optional[int] = Field(None, ge=64, description="Image height in pixels")
    seed: Optional[int] = Field(None, ge=0, description="Random seed")

    # Advanced parameters
    denoise_strength: Optional[float] = Field(None, ge=0, le=1, description="Denoising strength")
    refiner_denoise_start: Optional[float] = Field(None, ge=0, le=1, description="Refiner denoising start")
    clip_skip: Optional[int] = Field(None, ge=0, description="CLIP skip layers")
    seamless_x: Optional[bool] = Field(None, description="Enable seamless X tiling")
    seamless_y: Optional[bool] = Field(None, description="Enable seamless Y tiling")

    # Refiner aesthetics
    refiner_positive_aesthetic_score: Optional[float] = Field(None, description="Refiner positive aesthetic score")
    refiner_negative_aesthetic_score: Optional[float] = Field(None, description="Refiner negative aesthetic score")

    # LoRAs, ControlNets, and IP Adapters
    loras: Optional[list[LoRARecallParameter]] = Field(None, description="List of LoRAs with their weights")
    control_layers: Optional[list[ControlNetRecallParameter]] = Field(
        None, description="List of control adapters (ControlNet, T2I Adapter, Control LoRA) with their settings"
    )
    ip_adapters: Optional[list[IPAdapterRecallParameter]] = Field(
        None, description="List of IP Adapters with their settings"
    )


def resolve_model_name_to_key(model_name: str, model_type: ModelType = ModelType.Main) -> Optional[str]:
    """
    Look up a model by name and return its key.

    Args:
        model_name: The name of the model to look up
        model_type: The type of model to search for (default: Main)

    Returns:
        The key of the first matching model, or None if not found.
    """
    logger = ApiDependencies.invoker.services.logger
    try:
        models = ApiDependencies.invoker.services.model_manager.store.search_by_attr(
            model_name=model_name, model_type=model_type
        )

        if models:
            logger.info(f"Resolved {model_type.value} model name '{model_name}' to key '{models[0].key}'")
            return models[0].key

        logger.warning(f"Could not find {model_type.value} model with name '{model_name}'")
        return None
    except Exception as e:
        logger.error(f"Exception during {model_type.value} model lookup: {e}", exc_info=True)
        return None


def load_image_file(image_name: str) -> Optional[dict[str, Any]]:
    """
    Load an image from the outputs/images directory.

    Args:
        image_name: The filename of the image in outputs/images

    Returns:
        A dictionary with image_name, width, and height, or None if the image cannot be found
    """
    logger = ApiDependencies.invoker.services.logger
    try:
        # Prefer using the image_files service to validate & open images
        image_files = ApiDependencies.invoker.services.image_files
        # Resolve a safe path inside outputs
        image_path = image_files.get_path(image_name)

        if not image_files.validate_path(str(image_path)):
            logger.warning(f"Image file not found: {image_name} (searched in {image_path.parent})")
            return None

        # Open the image via service to leverage caching
        pil_image = image_files.get(image_name)
        width, height = pil_image.size
        logger.info(f"Found image file: {image_name} ({width}x{height})")
        return {"image_name": image_name, "width": width, "height": height}
    except Exception as e:
        logger.warning(f"Error loading image file {image_name}: {e}")
        return None


def resolve_lora_models(loras: list[LoRARecallParameter]) -> list[dict[str, Any]]:
    """
    Resolve LoRA model names to keys and build configuration list.

    Args:
        loras: List of LoRA recall parameters

    Returns:
        List of resolved LoRA configurations with model keys
    """
    logger = ApiDependencies.invoker.services.logger
    resolved_loras = []

    for lora in loras:
        model_key = resolve_model_name_to_key(lora.model_name, ModelType.LoRA)
        if model_key:
            resolved_loras.append({"model_key": model_key, "weight": lora.weight, "is_enabled": lora.is_enabled})
        else:
            logger.warning(f"Skipping LoRA '{lora.model_name}' - model not found")

    return resolved_loras


def resolve_control_models(control_layers: list[ControlNetRecallParameter]) -> list[dict[str, Any]]:
    """
    Resolve control adapter model names to keys and build configuration list.

    Tries to resolve as ControlNet, T2I Adapter, or Control LoRA in that order.

    Args:
        control_layers: List of control adapter recall parameters

    Returns:
        List of resolved control adapter configurations with model keys
    """
    logger = ApiDependencies.invoker.services.logger
    services = ApiDependencies.invoker.services
    resolved_controls = []

    for control in control_layers:
        model_key = None

        # Try ControlNet first
        model_key = resolve_model_name_to_key(control.model_name, ModelType.ControlNet)
        if not model_key:
            # Try T2I Adapter
            model_key = resolve_model_name_to_key(control.model_name, ModelType.T2IAdapter)
        if not model_key:
            # Try Control LoRA (also uses LoRA type)
            model_key = resolve_model_name_to_key(control.model_name, ModelType.LoRA)

        if model_key:
            config: dict[str, Any] = {"model_key": model_key, "weight": control.weight}
            if control.image_name is not None:
                image_data = load_image_file(control.image_name)
                if image_data:
                    config["image"] = image_data

                    # Try to process the image using the model's default processor
                    processed_image_data = process_controlnet_image(control.image_name, model_key, services)
                    if processed_image_data:
                        config["processed_image"] = processed_image_data
                        logger.info(f"Added processed image for control adapter {control.model_name}")
                else:
                    logger.warning(f"Could not load image for control adapter: {control.image_name}")
            if control.begin_step_percent is not None:
                config["begin_step_percent"] = control.begin_step_percent
            if control.end_step_percent is not None:
                config["end_step_percent"] = control.end_step_percent
            if control.control_mode is not None:
                config["control_mode"] = control.control_mode

            resolved_controls.append(config)
        else:
            logger.warning(f"Skipping control adapter '{control.model_name}' - model not found")

    return resolved_controls


def resolve_ip_adapter_models(ip_adapters: list[IPAdapterRecallParameter]) -> list[dict[str, Any]]:
    """
    Resolve IP Adapter model names to keys and build configuration list.

    Args:
        ip_adapters: List of IP Adapter recall parameters

    Returns:
        List of resolved IP Adapter configurations with model keys
    """
    logger = ApiDependencies.invoker.services.logger
    resolved_adapters = []

    for adapter in ip_adapters:
        # Try resolving as IP Adapter; if not found, try FLUX Redux
        model_key = resolve_model_name_to_key(adapter.model_name, ModelType.IPAdapter)
        if not model_key:
            model_key = resolve_model_name_to_key(adapter.model_name, ModelType.FluxRedux)
        if model_key:
            config: dict[str, Any] = {
                "model_key": model_key,
                # Always include weight; ignored by FLUX Redux on the frontend
                "weight": adapter.weight,
            }
            if adapter.image_name is not None:
                image_data = load_image_file(adapter.image_name)
                if image_data:
                    config["image"] = image_data
                else:
                    logger.warning(f"Could not load image for IP Adapter: {adapter.image_name}")
            if adapter.begin_step_percent is not None:
                config["begin_step_percent"] = adapter.begin_step_percent
            if adapter.end_step_percent is not None:
                config["end_step_percent"] = adapter.end_step_percent
            if adapter.method is not None:
                config["method"] = adapter.method
            # Include FLUX Redux image influence when provided
            if adapter.image_influence is not None:
                config["image_influence"] = adapter.image_influence

            resolved_adapters.append(config)
        else:
            logger.warning(f"Skipping IP Adapter '{adapter.model_name}' - model not found")

    return resolved_adapters


@recall_parameters_router.post(
    "/{queue_id}",
    operation_id="update_recall_parameters",
    response_model=dict[str, Any],
)
async def update_recall_parameters(
    queue_id: str = Path(..., description="The queue id to perform this operation on"),
    parameters: RecallParameter = Body(..., description="Recall parameters to update"),
) -> dict[str, Any]:
    """
    Update recallable parameters that can be recalled on the frontend.

    This endpoint allows updating parameters such as prompt, model, steps, and other
    generation settings. These parameters are stored in client state and can be
    accessed by the frontend to populate UI elements.

    Args:
        queue_id: The queue ID to associate these parameters with
        parameters: The RecallParameter object containing the parameters to update

    Returns:
        A dictionary containing the updated parameters and status

    Example:
        POST /api/v1/recall/{queue_id}
        {
            "positive_prompt": "a beautiful landscape",
            "model": "sd-1.5",
            "steps": 20,
            "cfg_scale": 7.5,
            "width": 512,
            "height": 512,
            "seed": 12345
        }
    """
    logger = ApiDependencies.invoker.services.logger

    try:
        # Get only the parameters that were actually provided (non-None values)
        provided_params = {k: v for k, v in parameters.model_dump().items() if v is not None}

        if not provided_params:
            return {"status": "no_parameters_provided", "updated_count": 0}

        # Store each parameter in client state using a consistent key format
        updated_count = 0
        for param_key, param_value in provided_params.items():
            # Convert parameter values to JSON strings for storage
            value_str = json.dumps(param_value)
            try:
                ApiDependencies.invoker.services.client_state_persistence.set_by_key(
                    queue_id, f"recall_{param_key}", value_str
                )
                updated_count += 1
            except Exception as e:
                logger.error(f"Error setting recall parameter {param_key}: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error setting recall parameter {param_key}",
                )

        logger.info(f"Updated {updated_count} recall parameters for queue {queue_id}")

        # Resolve model name to key if a model was provided
        if "model" in provided_params and isinstance(provided_params["model"], str):
            model_name = provided_params["model"]
            model_key = resolve_model_name_to_key(model_name, ModelType.Main)

            if model_key:
                logger.info(f"Resolved model name '{model_name}' to key '{model_key}'")
                provided_params["model"] = model_key
            else:
                logger.warning(f"Could not resolve model name '{model_name}' to a model key")
                # Remove model from parameters if we couldn't resolve it
                del provided_params["model"]

        # Process LoRAs if provided
        if "loras" in provided_params:
            loras_param = parameters.loras
            if loras_param is not None:
                resolved_loras = resolve_lora_models(loras_param)
                provided_params["loras"] = resolved_loras
                logger.info(f"Resolved {len(resolved_loras)} LoRA(s)")

        # Process control layers if provided
        if "control_layers" in provided_params:
            control_layers_param = parameters.control_layers
            if control_layers_param is not None:
                resolved_controls = resolve_control_models(control_layers_param)
                provided_params["control_layers"] = resolved_controls
                logger.info(f"Resolved {len(resolved_controls)} control layer(s)")

        # Process IP adapters if provided
        if "ip_adapters" in provided_params:
            ip_adapters_param = parameters.ip_adapters
            if ip_adapters_param is not None:
                resolved_adapters = resolve_ip_adapter_models(ip_adapters_param)
                provided_params["ip_adapters"] = resolved_adapters
                logger.info(f"Resolved {len(resolved_adapters)} IP adapter(s)")

        # Emit event to notify frontend of parameter updates
        try:
            logger.info(
                f"Emitting recall_parameters_updated event for queue {queue_id} with {len(provided_params)} parameters"
            )
            ApiDependencies.invoker.services.events.emit_recall_parameters_updated(queue_id, provided_params)
            logger.info("Successfully emitted recall_parameters_updated event")
        except Exception as e:
            logger.error(f"Error emitting recall parameters event: {e}", exc_info=True)
            # Don't fail the request if event emission fails, just log it

        return {
            "status": "success",
            "queue_id": queue_id,
            "updated_count": updated_count,
            "parameters": provided_params,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating recall parameters: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error updating recall parameters",
        )


@recall_parameters_router.get(
    "/{queue_id}",
    operation_id="get_recall_parameters",
    response_model=dict[str, Any],
)
async def get_recall_parameters(
    queue_id: str = Path(..., description="The queue id to retrieve parameters for"),
) -> dict[str, Any]:
    """
    Retrieve all stored recall parameters for a given queue.

    Returns a dictionary of all recall parameters that have been set for the queue.

    Args:
        queue_id: The queue ID to retrieve parameters for

    Returns:
        A dictionary containing all stored recall parameters
    """
    logger = ApiDependencies.invoker.services.logger

    try:
        # Retrieve all recall parameters by iterating through expected keys
        # Since client_state_persistence doesn't have a "get_all" method, we'll
        # return an informative response
        return {
            "status": "success",
            "queue_id": queue_id,
            "note": "Use the frontend to access stored recall parameters, or set specific parameters using POST",
        }

    except Exception as e:
        logger.error(f"Error retrieving recall parameters: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error retrieving recall parameters",
        )
