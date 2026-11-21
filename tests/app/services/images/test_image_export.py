import io
import json
from pathlib import Path
from typing import Optional

import pytest
from PIL import Image, ImageCms
from PIL.PngImagePlugin import PngInfo

from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.images.image_export import (
    ImageExportFormat,
    ImageExportMetadata,
    ImageExportOptions,
    encode_image,
    resolve_export_file_names,
)
from invokeai.app.services.images.images_common import ImageDTO
from invokeai.app.services.invoker import Invoker

INVOKE_CHUNKS = {
    "invokeai_metadata": '{"seed": 7, "marker": "only-in-metadata"}',
    "invokeai_workflow": '{"marker": "only-in-workflow"}',
    "invokeai_graph": '{"marker": "only-in-graph"}',
    "invokeai_resolved_inputs": '{"marker": "only-in-resolved-inputs"}',
}
SRGB = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def _stored_png(tmp_path: Path, image: Image.Image, icc_profile: Optional[bytes] = None) -> Path:
    pnginfo = PngInfo()
    for key, value in INVOKE_CHUNKS.items():
        pnginfo.add_text(key, value)
    path = tmp_path / "stored.png"
    image.save(path, "PNG", pnginfo=pnginfo, icc_profile=icc_profile)
    return path


def _decode(data: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(data))
    image.load()
    return image


@pytest.mark.parametrize(
    ("format", "pil_format"), [("png", "PNG"), ("jpeg", "JPEG"), ("webp", "WEBP"), ("avif", "AVIF")]
)
def test_every_format_decodes_as_itself_and_keeps_the_icc_profile(
    tmp_path: Path, format: ImageExportFormat, pil_format: str
) -> None:
    path = _stored_png(tmp_path, Image.new("RGB", (40, 30), (200, 10, 10)), icc_profile=SRGB)

    exported = _decode(encode_image(path, ImageExportOptions(format=format)))

    assert exported.format == pil_format
    assert exported.size == (40, 30)
    assert exported.info.get("icc_profile") == SRGB


def test_long_edge_cap_shrinks_and_keeps_aspect(tmp_path: Path) -> None:
    path = _stored_png(tmp_path, Image.new("RGB", (400, 200)))

    assert _decode(encode_image(path, ImageExportOptions(format="webp", long_edge=100))).size == (100, 50)
    assert _decode(encode_image(path, ImageExportOptions(format="png", long_edge=3))).size == (3, 2)


def test_long_edge_cap_above_the_source_never_enlarges(tmp_path: Path) -> None:
    path = _stored_png(tmp_path, Image.new("RGB", (400, 200)))

    assert _decode(encode_image(path, ImageExportOptions(format="jpeg", long_edge=1000))).size == (400, 200)


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        ("all", set(INVOKE_CHUNKS)),
        ("parameters", {"invokeai_metadata"}),
        ("none", set()),
    ],
)
def test_png_metadata_level_selects_invoke_chunks(
    tmp_path: Path, metadata: ImageExportMetadata, expected: set[str]
) -> None:
    path = _stored_png(tmp_path, Image.new("RGBA", (8, 8)))

    exported = _decode(encode_image(path, ImageExportOptions(format="png", metadata=metadata, long_edge=4)))

    assert {key for key in exported.info if key.startswith("invokeai_")} == expected
    assert all(exported.info[key] == INVOKE_CHUNKS[key] for key in expected)


@pytest.mark.parametrize("format", ["jpeg", "webp", "avif"])
def test_lossy_formats_carry_no_invoke_metadata(tmp_path: Path, format: ImageExportFormat) -> None:
    path = _stored_png(tmp_path, Image.new("RGB", (8, 8)))

    data = encode_image(path, ImageExportOptions(format=format, metadata="all"))

    assert b"only-in-" not in data
    assert b"invokeai_" not in data


def test_transparent_source_becomes_a_jpeg_on_white(tmp_path: Path) -> None:
    source = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
    source.putpixel((0, 0), (255, 0, 0, 255))
    path = _stored_png(tmp_path, source)

    exported = _decode(encode_image(path, ImageExportOptions(format="jpeg", quality=100)))

    assert exported.mode == "RGB"
    assert all(abs(a - b) <= 8 for a, b in zip(exported.getpixel((7, 7)), (255, 255, 255), strict=True))


def test_sixteen_bit_source_keeps_its_tones_in_a_lossy_format(tmp_path: Path) -> None:
    source = Image.new("I;16", (8, 8), 32768)
    path = _stored_png(tmp_path, source)

    exported = _decode(encode_image(path, ImageExportOptions(format="webp", quality=100)))

    assert all(abs(channel - 128) <= 3 for channel in exported.getpixel((4, 4)))


# File names


def _dto(
    image_name: str = "abc.png", board_id: Optional[str] = None, created_at: str = "2026-03-04 05:06:07.000"
) -> ImageDTO:
    return ImageDTO(
        image_name=image_name,
        board_id=board_id,
        image_url="unused",
        thumbnail_url="unused",
        width=8,
        height=8,
        image_origin=ResourceOrigin.INTERNAL,
        image_category=ImageCategory.GENERAL,
        created_at=created_at,
        updated_at=created_at,
        starred=False,
        has_workflow=False,
        is_intermediate=False,
    )


def _save_record(mock_invoker: Invoker, image_name: str, metadata: Optional[dict] = None) -> None:
    mock_invoker.services.image_records.save(
        image_name=image_name,
        image_origin=ResourceOrigin.INTERNAL,
        image_category=ImageCategory.GENERAL,
        width=8,
        height=8,
        has_workflow=False,
        metadata=json.dumps(metadata) if metadata is not None else None,
    )


def test_tokens_resolve_from_board_metadata_date_and_position(mock_invoker: Invoker) -> None:
    board = mock_invoker.services.board_records.save("Portraits", "system")
    _save_record(mock_invoker, "first.png", {"seed": 42, "model": {"name": "SDXL Base"}})
    _save_record(mock_invoker, "second.png", {})
    images = [_dto("first.png", board_id=board.board_id), _dto("second.png")]

    names = resolve_export_file_names(mock_invoker.services, images, "{board}_{model}_{seed}_{date}_{n}_{name}", "jpeg")

    assert names == [
        "Portraits_SDXL Base_42_2026-03-04_001_first.jpg",
        "uncategorized_2026-03-04_002_second.jpg",
    ]


@pytest.mark.parametrize(
    ("pattern", "expected"),
    [
        ("{unknown}-{name}", "{unknown}-abc.png"),
        ("{board} - {seed} - {name}", "uncategorized - abc.png"),
        ("{seed}__{name}..{model}", "abc.png"),
        ('a:b*c?"d<e>f|g\x07h', "a-b-c--d-e-f-g-h.png"),
        (" .-{seed}- . ", "abc.png"),
        ("ä" * 100, "ä" * 60 + ".png"),
    ],
    ids=["unknown-token", "empty-token-spaced", "empty-tokens-at-ends", "unsafe-characters", "fallback", "byte-cut"],
)
def test_pattern_is_rendered_into_a_safe_stem(mock_invoker: Invoker, pattern: str, expected: str) -> None:
    _save_record(mock_invoker, "abc.png")

    assert resolve_export_file_names(mock_invoker.services, [_dto()], pattern, "png") == [expected]


def test_repeated_names_get_numbered_suffixes_ignoring_case(mock_invoker: Invoker) -> None:
    images = [_dto("A.png"), _dto("a.png"), _dto("b.png"), _dto("a-2.png")]

    names = resolve_export_file_names(mock_invoker.services, images, "{name}", "webp")

    assert names == ["A.webp", "a-2.webp", "b.webp", "a-2-2.webp"]


def test_position_padding_widens_past_three_digits(mock_invoker: Invoker) -> None:
    images = [_dto(f"{index}.png") for index in range(1000)]

    names = resolve_export_file_names(mock_invoker.services, images, "{n}", "avif")

    assert names[0] == "0001.avif"
    assert names[-1] == "1000.avif"


def test_hostile_board_name_and_pattern_cannot_form_a_path(mock_invoker: Invoker) -> None:
    board = mock_invoker.services.board_records.save("../../etc\\passwd", "system")

    [name] = resolve_export_file_names(
        mock_invoker.services, [_dto(board_id=board.board_id)], "../{board}/..\\{name}", "png"
    )

    assert name == "etc-passwd-..-abc.png"
    assert "/" not in name and "\\" not in name
