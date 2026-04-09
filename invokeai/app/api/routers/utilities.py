import re
from pathlib import Path
from typing import Optional, Union
from urllib.parse import quote

from dynamicprompts.generators import CombinatorialPromptGenerator, RandomPromptGenerator
from fastapi import Body, HTTPException
from fastapi.routing import APIRouter
from fontTools.ttLib import TTFont
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
    return (root / "Fonts").resolve()


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


@utilities_router.get(
    "/fonts",
    operation_id="list_user_fonts",
    responses={200: {"model": UserFontsResponse}},
)
async def list_user_fonts() -> UserFontsResponse:
    fonts_dir = _get_fonts_dir()
    if not fonts_dir.exists() or not fonts_dir.is_dir():
        return UserFontsResponse(fonts=[])

    family_candidates: dict[str, list[tuple[Path, str, str, int, str]]] = {}
    # key -> [(font_file, relative, family, weight, style)]
    for font_file in sorted(fonts_dir.rglob("*")):
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
        _, selected_relative, selected_family, _, _ = min(
            candidates, key=lambda c: _candidate_score(c[3], c[4], c[0])
        )
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
                id=f"user:{selected_family.strip().lower()}",
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
