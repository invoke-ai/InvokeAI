import asyncio
import logging
import re
import threading
from pathlib import Path
from typing import Optional, Union
from urllib.parse import quote

import torch
from dynamicprompts.generators import CombinatorialPromptGenerator, RandomPromptGenerator
from fastapi import Body, HTTPException
from fastapi.routing import APIRouter
from fontTools.ttLib import TTFont
from PIL import ImageFont
from pydantic import BaseModel, Field
from pyparsing import ParseException
from starlette.responses import FileResponse
from transformers import AutoProcessor, AutoTokenizer, LlavaOnevisionForConditionalGeneration, LlavaOnevisionProcessor

from invokeai.app.api.auth_dependencies import CurrentUserOrDefault
from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.services.image_files.image_files_common import ImageFileNotFoundException
from invokeai.app.services.model_records.model_records_base import UnknownModelException
from invokeai.backend.llava_onevision_pipeline import LlavaOnevisionPipeline
from invokeai.backend.model_manager.taxonomy import ModelType
from invokeai.backend.text_llm_pipeline import DEFAULT_SYSTEM_PROMPT, TextLLMPipeline
from invokeai.backend.util.devices import TorchDevice

logger = logging.getLogger(__name__)

utilities_router = APIRouter(prefix="/v1/utilities", tags=["utilities"])
SUPPORTED_FONT_EXTENSIONS = {".ttf", ".otf", ".woff", ".woff2"}

# The underlying model loader is not thread-safe, so we serialize load_model calls.
_model_load_lock = threading.Lock()


class DynamicPromptsResponse(BaseModel):
    prompts: list[str]
    error: Optional[str] = None


class UserFontFace(BaseModel):
    path: str
    url: str
    weight: int
    style: str


class UserFont(BaseModel):
    id: str
    family: str
    label: str
    path: str
    url: str
    faces: list[UserFontFace]


class UserFontsResponse(BaseModel):
    fonts: list[UserFont]


@utilities_router.post(
    "/dynamicprompts",
    operation_id="parse_dynamicprompts",
    responses={
        200: {"model": DynamicPromptsResponse},
    },
)
async def parse_dynamicprompts(
    prompt: str = Body(description="The prompt to parse with dynamicprompts"),
    max_prompts: int = Body(ge=1, le=10000, default=1000, description="The max number of prompts to generate"),
    combinatorial: bool = Body(default=True, description="Whether to use the combinatorial generator"),
    seed: int | None = Body(None, description="The seed to use for random generation. Only used if not combinatorial"),
) -> DynamicPromptsResponse:
    """Creates a batch process"""
    max_prompts = min(max_prompts, 10000)
    generator: Union[RandomPromptGenerator, CombinatorialPromptGenerator]
    try:
        error: Optional[str] = None
        if combinatorial:
            generator = CombinatorialPromptGenerator()
            prompts = generator.generate(prompt, max_prompts=max_prompts)
        else:
            generator = RandomPromptGenerator(seed=seed)
            prompts = generator.generate(prompt, num_images=max_prompts)
    except ParseException as e:
        prompts = [prompt]
        error = str(e)
    return DynamicPromptsResponse(prompts=prompts if prompts else [""], error=error)


def _get_fonts_dir() -> Path:
    root = ApiDependencies.invoker.services.configuration.root_path
    return root / "Fonts"


def _path_has_symlink_component(path: Path, boundary: Path) -> bool:
    current = path
    boundary = boundary.absolute()
    try:
        current.absolute().relative_to(boundary)
    except ValueError:
        return True

    while True:
        if current.exists() and current.is_symlink():
            return True
        if current == boundary:
            return False
        current = current.parent


def _get_name_table_value(font: TTFont, name_ids: tuple[int, ...]) -> str | None:
    if "name" not in font:
        return None

    def _sort_key(record: object) -> tuple[int, int]:
        platform_id = getattr(record, "platformID", -1)
        lang_id = getattr(record, "langID", -1)
        if platform_id == 3 and lang_id in (0x0409, 0):
            return (0, 0)
        if platform_id == 3:
            return (1, 0)
        if platform_id == 0:
            return (2, 0)
        return (3, lang_id)

    records = font["name"].names
    for name_id in name_ids:
        for record in sorted((record for record in records if record.nameID == name_id), key=_sort_key):
            try:
                value = record.toUnicode().strip()
            except Exception:
                continue
            if value:
                return value
    return None


def _normalize_variant_text(value: str) -> str:
    value = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", value)
    value = value.replace("_", " ").replace("-", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip().lower()


def _infer_font_weight(style_name: str, file_stem: str, weight_class: int | None) -> int:
    if isinstance(weight_class, int) and 1 <= weight_class <= 1000:
        return weight_class

    combined = _normalize_variant_text(f"{style_name} {file_stem}")
    # Ordering matters: more specific keywords must be matched before "bold".
    weight_keywords = [
        (("thin", "hairline"), 100),
        (("extra light", "ultra light", "extralight", "ultralight"), 200),
        (("light",), 300),
        (("normal", "regular", "roman", "book"), 400),
        (("medium",), 500),
        (("semi bold", "semibold", "demi bold", "demibold"), 600),
        (("extra bold", "ultra bold", "extrabold", "ultrabold"), 800),
        (("black", "heavy"), 900),
        (("bold",), 700),
    ]
    for keywords, weight in weight_keywords:
        if any(keyword in combined for keyword in keywords):
            return weight

    return 400


def _infer_font_style(style_name: str, file_stem: str, italic_flag: bool) -> str:
    if italic_flag:
        return "italic"

    combined = _normalize_variant_text(f"{style_name} {file_stem}")
    if "italic" in combined or "oblique" in combined:
        return "italic"

    return "normal"


def _get_font_metadata(font_file: Path) -> tuple[str, str, int, str]:
    fallback_family = font_file.stem
    fallback_label = font_file.stem.replace("_", " ").replace("-", " ").strip() or font_file.stem

    try:
        with TTFont(font_file.as_posix(), lazy=True) as font:
            family_name = (_get_name_table_value(font, (16, 1)) or "").strip()
            style_name = (_get_name_table_value(font, (17, 2)) or "").strip()
            os2_table = font["OS/2"] if "OS/2" in font else None
            head_table = font["head"] if "head" in font else None
            post_table = font["post"] if "post" in font else None
            weight_class = getattr(os2_table, "usWeightClass", None)
            italic_flag = bool(getattr(os2_table, "fsSelection", 0) & 0x01) or bool(
                getattr(head_table, "macStyle", 0) & 0x02
            )
            if post_table is not None:
                italic_flag = italic_flag or bool(getattr(post_table, "italicAngle", 0))
            if family_name:
                return (
                    family_name,
                    family_name,
                    _infer_font_weight(style_name, font_file.stem, weight_class),
                    _infer_font_style(style_name, font_file.stem, italic_flag),
                )
    except Exception:
        pass

    try:
        font = ImageFont.truetype(font_file.as_posix(), size=16)
        family_name, style_name = font.getname()
        family_name = family_name.strip()
        style_name = style_name.strip()
        if not family_name:
            return (
                fallback_family,
                fallback_label,
                _infer_font_weight(style_name, font_file.stem, None),
                _infer_font_style(style_name, font_file.stem, False),
            )
        return (
            family_name,
            family_name,
            _infer_font_weight(style_name, font_file.stem, None),
            _infer_font_style(style_name, font_file.stem, False),
        )
    except Exception:
        return (
            fallback_family,
            fallback_label,
            _infer_font_weight("", font_file.stem, None),
            _infer_font_style("", font_file.stem, False),
        )


def _resolve_font_request_path(font_path: str) -> Path:
    fonts_dir = _get_fonts_dir()
    if not fonts_dir.exists() or not fonts_dir.is_dir():
        raise HTTPException(status_code=404, detail="Font file not found")

    requested = (fonts_dir / font_path).absolute()
    if _path_has_symlink_component(requested, fonts_dir):
        raise HTTPException(status_code=400, detail="Invalid font path")

    resolved_fonts_dir = fonts_dir.resolve()
    resolved_requested = requested.resolve()

    try:
        resolved_requested.relative_to(resolved_fonts_dir)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid font path") from e

    return resolved_requested


@utilities_router.get(
    "/fonts",
    operation_id="list_user_fonts",
    responses={200: {"model": UserFontsResponse}},
)
async def list_user_fonts(_current_user: CurrentUserOrDefault) -> UserFontsResponse:
    fonts_dir = _get_fonts_dir()
    if not fonts_dir.exists() or not fonts_dir.is_dir() or fonts_dir.is_symlink():
        return UserFontsResponse(fonts=[])

    family_candidates: dict[str, list[tuple[Path, str, str, int, str]]] = {}
    # key -> [(font_file, relative, family, weight, style)]
    for font_file in sorted(fonts_dir.rglob("*")):
        if _path_has_symlink_component(font_file.absolute(), fonts_dir):
            continue
        if not font_file.is_file() or font_file.suffix.lower() not in SUPPORTED_FONT_EXTENSIONS:
            continue
        relative = font_file.relative_to(fonts_dir).as_posix()
        family, _label, weight, style = _get_font_metadata(font_file)
        family_key = family.strip().lower()
        family_candidates.setdefault(family_key, []).append((font_file, relative, family, weight, style))

    def _candidate_score(weight: int, style: str, path: Path) -> tuple[int, int, int]:
        """Lower score is better. Prefer regular/normal faces, then shorter names."""
        return (0 if style == "normal" else 1, abs(weight - 400), len(path.stem))

    fonts: list[UserFont] = []
    for _, candidates in sorted(family_candidates.items(), key=lambda kv: kv[0]):
        _, selected_relative, selected_family, _, _ = min(candidates, key=lambda c: _candidate_score(c[3], c[4], c[0]))
        faces_by_variant: dict[tuple[int, str], tuple[Path, str, str, int, str]] = {}
        for candidate in candidates:
            variant_key = (candidate[3], candidate[4])
            current = faces_by_variant.get(variant_key)
            if current is None or _candidate_score(candidate[3], candidate[4], candidate[0]) < _candidate_score(
                current[3], current[4], current[0]
            ):
                faces_by_variant[variant_key] = candidate

        faces = [
            UserFontFace(
                path=relative,
                url=f"/api/v1/utilities/fonts/{quote(relative)}",
                weight=weight,
                style=style,
            )
            for (weight, style), (_, relative, _, _, _) in sorted(
                faces_by_variant.items(), key=lambda item: (item[0][1] != "normal", abs(item[0][0] - 400), item[0][0])
            )
        ]

        fonts.append(
            UserFont(
                id=f"user:{selected_relative}",
                family=selected_family,
                label=selected_family,
                path=selected_relative,
                url=f"/api/v1/utilities/fonts/{quote(selected_relative)}",
                faces=faces,
            )
        )

    return UserFontsResponse(fonts=fonts)


@utilities_router.get(
    "/fonts/{font_path:path}",
    operation_id="get_user_font_file",
)
async def get_user_font_file(font_path: str, _current_user: CurrentUserOrDefault) -> FileResponse:
    requested = _resolve_font_request_path(font_path)
    if not requested.exists() or not requested.is_file():
        raise HTTPException(status_code=404, detail="Font file not found")

    if requested.suffix.lower() not in SUPPORTED_FONT_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported font format")

    return FileResponse(path=requested)


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
