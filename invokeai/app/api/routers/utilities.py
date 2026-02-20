from pathlib import Path
from typing import Optional, Union
from urllib.parse import quote

from dynamicprompts.generators import CombinatorialPromptGenerator, RandomPromptGenerator
from fastapi import Body, HTTPException
from fastapi.routing import APIRouter
from PIL import ImageFont
from pydantic import BaseModel
from pyparsing import ParseException
from starlette.responses import FileResponse

from invokeai.app.api.dependencies import ApiDependencies

utilities_router = APIRouter(prefix="/v1/utilities", tags=["utilities"])
SUPPORTED_FONT_EXTENSIONS = {".ttf", ".otf", ".woff", ".woff2"}


class DynamicPromptsResponse(BaseModel):
    prompts: list[str]
    error: Optional[str] = None


class UserFont(BaseModel):
    id: str
    family: str
    label: str
    path: str
    url: str


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
    return (root / "Fonts").resolve()


def _get_font_display_names(font_file: Path) -> tuple[str, str]:
    fallback_family = font_file.stem
    fallback_label = font_file.stem.replace("_", " ").replace("-", " ").strip() or font_file.stem

    try:
        font = ImageFont.truetype(font_file.as_posix(), size=16)
        family_name, style_name = font.getname()
        family_name = family_name.strip()
        style_name = style_name.strip()
        if not family_name:
            return fallback_family, fallback_label
        if style_name and style_name.lower() not in {"regular", "normal", "roman", "book"}:
            return family_name, f"{family_name} {style_name}"
        return family_name, family_name
    except Exception:
        return fallback_family, fallback_label


@utilities_router.get(
    "/fonts",
    operation_id="list_user_fonts",
    responses={200: {"model": UserFontsResponse}},
)
async def list_user_fonts() -> UserFontsResponse:
    fonts_dir = _get_fonts_dir()
    if not fonts_dir.exists() or not fonts_dir.is_dir():
        return UserFontsResponse(fonts=[])

    family_candidates: dict[str, list[tuple[Path, str, str, str]]] = {}
    # key -> [(font_file, relative, family, label)]
    for font_file in sorted(fonts_dir.rglob("*")):
        if not font_file.is_file() or font_file.suffix.lower() not in SUPPORTED_FONT_EXTENSIONS:
            continue
        relative = font_file.relative_to(fonts_dir).as_posix()
        family, label = _get_font_display_names(font_file)
        family_key = family.strip().lower()
        family_candidates.setdefault(family_key, []).append((font_file, relative, family, label))

    def _candidate_score(label_text: str, path: Path) -> tuple[int, int]:
        """Lower score is better. Prefer regular/normal names, then shorter names."""
        combined = f"{label_text} {path.stem}".lower()
        preferred = any(token in combined for token in (" regular", " normal", " roman", " book"))
        has_variant = any(token in combined for token in (" bold", " italic", " oblique", " black", " light", " medium"))
        if preferred:
            tier = 0
        elif has_variant:
            tier = 2
        else:
            tier = 1
        return (tier, len(path.stem))

    fonts: list[UserFont] = []
    seen_ids: set[str] = set()
    for _, candidates in sorted(family_candidates.items(), key=lambda kv: kv[0]):
        selected_file, selected_relative, selected_family, _selected_label = min(
            candidates, key=lambda c: _candidate_score(c[3], c[0])
        )
        base_id = f"user:{selected_relative.lower()}"
        font_id = base_id
        i = 1
        while font_id in seen_ids:
            i += 1
            font_id = f"{base_id}:{i}"
        seen_ids.add(font_id)
        fonts.append(
            UserFont(
                id=font_id,
                family=selected_family,
                label=selected_family,
                path=selected_relative,
                url=f"/api/v1/utilities/fonts/{quote(selected_relative)}",
            )
        )

    return UserFontsResponse(fonts=fonts)


@utilities_router.get(
    "/fonts/{font_path:path}",
    operation_id="get_user_font_file",
)
async def get_user_font_file(font_path: str) -> FileResponse:
    fonts_dir = _get_fonts_dir()
    requested = (fonts_dir / font_path).resolve()

    try:
        requested.relative_to(fonts_dir)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid font path") from e

    if not requested.exists() or not requested.is_file():
        raise HTTPException(status_code=404, detail="Font file not found")

    if requested.suffix.lower() not in SUPPORTED_FONT_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported font format")

    return FileResponse(path=requested)
