import asyncio
import logging
import threading
from pathlib import Path
from typing import Optional, Union

import torch
from dynamicprompts.generators import CombinatorialPromptGenerator, RandomPromptGenerator
from dynamicprompts.wildcards import WildcardManager
from fastapi import Body, HTTPException, Query
from fastapi.routing import APIRouter
from pydantic import BaseModel, Field
from pyparsing import ParseException
from transformers import AutoProcessor, AutoTokenizer, LlavaOnevisionForConditionalGeneration, LlavaOnevisionProcessor

from invokeai.app.api.auth_dependencies import CurrentUserOrDefault
from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.services.image_files.image_files_common import ImageFileNotFoundException
from invokeai.app.services.model_records.model_records_base import UnknownModelException
from invokeai.app.util.wildcards import (
    WildcardsResponse,
    WildcardValuesResponse,
    clean_dynamic_prompt_outputs,
    find_missing_wildcard_references,
    get_wildcard_values,
    get_wildcards_path,
    index_wildcards,
)
from invokeai.backend.llava_onevision_pipeline import LlavaOnevisionPipeline
from invokeai.backend.model_manager.taxonomy import ModelType
from invokeai.backend.text_llm_pipeline import DEFAULT_SYSTEM_PROMPT, TextLLMPipeline
from invokeai.backend.util.devices import TorchDevice

logger = logging.getLogger(__name__)

utilities_router = APIRouter(prefix="/v1/utilities", tags=["utilities"])

# The underlying model loader is not thread-safe, so we serialize load_model calls.
_model_load_lock = threading.Lock()


class DynamicPromptsResponse(BaseModel):
    prompts: list[str]
    error: Optional[str] = None
    warnings: list[str] = Field(default_factory=list)
    missing_wildcards: list[str] = Field(default_factory=list)


@utilities_router.get(
    "/wildcards",
    operation_id="list_wildcards",
    responses={
        200: {"model": WildcardsResponse},
    },
)
async def list_wildcards(_: CurrentUserOrDefault) -> WildcardsResponse:
    """List local dynamic prompt wildcards from INVOKEAI_ROOT/wildcards."""
    wildcards_path = get_wildcards_path(ApiDependencies.invoker.services.configuration.root_path)
    return index_wildcards(wildcards_path)


@utilities_router.get(
    "/wildcards/values",
    operation_id="get_wildcard_values",
    responses={
        200: {"model": WildcardValuesResponse},
    },
)
async def list_wildcard_values(
    _: CurrentUserOrDefault,
    path: str = Query(description="The relative wildcard path to read values for"),
    limit: int = Query(default=200, ge=1, le=1000, description="The max number of wildcard values to return"),
) -> WildcardValuesResponse:
    """List values for a single local dynamic prompt wildcard."""
    wildcards_path = get_wildcards_path(ApiDependencies.invoker.services.configuration.root_path)
    values = get_wildcard_values(wildcards_path, path, limit)
    if values is None:
        raise HTTPException(status_code=404, detail=f"Wildcard '{path}' not found")
    return values


@utilities_router.post(
    "/dynamicprompts",
    operation_id="parse_dynamicprompts",
    responses={
        200: {"model": DynamicPromptsResponse},
    },
)
async def parse_dynamicprompts(
    _: CurrentUserOrDefault,
    prompt: str = Body(description="The prompt to parse with dynamicprompts"),
    max_prompts: int = Body(ge=1, le=10000, default=1000, description="The max number of prompts to generate"),
    combinatorial: bool = Body(default=True, description="Whether to use the combinatorial generator"),
    seed: int | None = Body(None, description="The seed to use for random generation. Only used if not combinatorial"),
) -> DynamicPromptsResponse:
    """Creates a batch process"""
    max_prompts = min(max_prompts, 10000)
    generator: Union[RandomPromptGenerator, CombinatorialPromptGenerator]
    warnings: list[str] = []
    wildcards_path = get_wildcards_path(ApiDependencies.invoker.services.configuration.root_path)
    wildcard_index = index_wildcards(wildcards_path)
    missing_wildcards = find_missing_wildcard_references(prompt, wildcard_index.wildcards)
    if wildcard_index.errors:
        warnings.append("Some wildcard files could not be indexed.")
    try:
        error: Optional[str] = None
        wildcard_manager = WildcardManager(wildcards_path) if wildcards_path.is_dir() else None
        if combinatorial:
            generator = CombinatorialPromptGenerator(wildcard_manager=wildcard_manager)
            prompts = generator.generate(prompt, max_prompts=max_prompts)
        else:
            generator = RandomPromptGenerator(wildcard_manager=wildcard_manager, seed=seed)
            prompts = generator.generate(prompt, num_images=max_prompts)
    except ParseException as e:
        prompts = [prompt]
        error = str(e)
    return DynamicPromptsResponse(
        prompts=clean_dynamic_prompt_outputs(prompts) if prompts else [""],
        error=error,
        warnings=warnings,
        missing_wildcards=missing_wildcards,
    )


# --- Expand Prompt ---


class ExpandPromptRequest(BaseModel):
    prompt: str
    model_key: str
    max_tokens: int = Field(default=300, ge=1, le=2048)
    system_prompt: str | None = None


class ExpandPromptResponse(BaseModel):
    expanded_prompt: str
    error: str | None = None


def _resolve_model_path(model_config_path: str) -> Path:
    """Resolve a model config path to an absolute path."""
    model_path = Path(model_config_path)
    if model_path.is_absolute():
        return model_path.resolve()
    base_models_path = ApiDependencies.invoker.services.configuration.models_path
    return (base_models_path / model_path).resolve()


def _run_expand_prompt(prompt: str, model_key: str, max_tokens: int, system_prompt: str | None) -> str:
    """Run text LLM inference synchronously (called from thread)."""
    model_manager = ApiDependencies.invoker.services.model_manager
    model_config = model_manager.store.get_model(model_key)

    if model_config.type != ModelType.TextLLM:
        raise ValueError(f"Model '{model_key}' is not a TextLLM model (got {model_config.type})")

    with _model_load_lock:
        loaded_model = model_manager.load.load_model(model_config)

    with torch.no_grad(), loaded_model.model_on_device() as (_, model):
        model_abs_path = _resolve_model_path(model_config.path)
        tokenizer = AutoTokenizer.from_pretrained(model_abs_path, local_files_only=True)

        pipeline = TextLLMPipeline(model, tokenizer)
        model_device = next(model.parameters()).device
        output = pipeline.run(
            prompt=prompt,
            system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT,
            max_new_tokens=max_tokens,
            device=model_device,
            dtype=TorchDevice.choose_torch_dtype(),
        )

    return output


@utilities_router.post(
    "/expand-prompt",
    operation_id="expand_prompt",
    responses={
        200: {"model": ExpandPromptResponse},
    },
)
async def expand_prompt(body: ExpandPromptRequest) -> ExpandPromptResponse:
    """Expand a brief prompt into a detailed image generation prompt using a text LLM."""
    try:
        expanded = await asyncio.to_thread(
            _run_expand_prompt,
            body.prompt,
            body.model_key,
            body.max_tokens,
            body.system_prompt,
        )
        return ExpandPromptResponse(expanded_prompt=expanded)
    except UnknownModelException:
        raise HTTPException(status_code=404, detail=f"Model '{body.model_key}' not found")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Error expanding prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- Image to Prompt ---


class ImageToPromptRequest(BaseModel):
    image_name: str
    model_key: str
    instruction: str = "Describe this image in detail for use as an AI image generation prompt."


class ImageToPromptResponse(BaseModel):
    prompt: str
    error: str | None = None


def _run_image_to_prompt(image_name: str, model_key: str, instruction: str) -> str:
    """Run LLaVA OneVision inference synchronously (called from thread)."""
    model_manager = ApiDependencies.invoker.services.model_manager
    model_config = model_manager.store.get_model(model_key)

    if model_config.type != ModelType.LlavaOnevision:
        raise ValueError(f"Model '{model_key}' is not a LLaVA OneVision model (got {model_config.type})")

    with _model_load_lock:
        loaded_model = model_manager.load.load_model(model_config)

    # Load the image from InvokeAI's image store
    image = ApiDependencies.invoker.services.images.get_pil_image(image_name)
    image = image.convert("RGB")

    with torch.no_grad(), loaded_model.model_on_device() as (_, model):
        if not isinstance(model, LlavaOnevisionForConditionalGeneration):
            raise TypeError(f"Expected LlavaOnevisionForConditionalGeneration, got {type(model).__name__}")

        model_abs_path = _resolve_model_path(model_config.path)
        processor = AutoProcessor.from_pretrained(model_abs_path, local_files_only=True)
        if not isinstance(processor, LlavaOnevisionProcessor):
            raise TypeError(f"Expected LlavaOnevisionProcessor, got {type(processor).__name__}")

        pipeline = LlavaOnevisionPipeline(model, processor)
        model_device = next(model.parameters()).device
        output = pipeline.run(
            prompt=instruction,
            images=[image],
            device=model_device,
            dtype=TorchDevice.choose_torch_dtype(),
        )

    return output


@utilities_router.post(
    "/image-to-prompt",
    operation_id="image_to_prompt",
    responses={
        200: {"model": ImageToPromptResponse},
    },
)
async def image_to_prompt(body: ImageToPromptRequest) -> ImageToPromptResponse:
    """Generate a descriptive prompt from an image using a vision-language model."""
    try:
        prompt = await asyncio.to_thread(
            _run_image_to_prompt,
            body.image_name,
            body.model_key,
            body.instruction,
        )
        return ImageToPromptResponse(prompt=prompt)
    except UnknownModelException:
        raise HTTPException(status_code=404, detail=f"Model '{body.model_key}' not found")
    except ImageFileNotFoundException:
        raise HTTPException(status_code=404, detail=f"Image '{body.image_name}' not found")
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating prompt from image: {e}")
        raise HTTPException(status_code=500, detail=str(e))
