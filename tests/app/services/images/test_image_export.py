import io
from pathlib import Path

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from invokeai.app.services.board_access.board_access_common import BoardViewer
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
    "invokeai_metadata": '{"seed": 7}',
    "invokeai_workflow": "{}",
    "invokeai_graph": "{}",
    "invokeai_resolved_inputs": "{}",
}


def _stored_png(tmp_path: Path, image: Image.Image) -> Path:
    pnginfo = PngInfo()
    for key, value in INVOKE_CHUNKS.items():
        pnginfo.add_text(key, value)
    path = tmp_path / "stored.png"
    image.save(path, "PNG", pnginfo=pnginfo)
    return path


def _decode(data: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(data))
    image.load()
    return image


@pytest.mark.parametrize("format", ["png", "jpeg", "webp", "avif"])
def test_export_decodes_as_its_format_capped_without_enlarging(tmp_path: Path, format: ImageExportFormat) -> None:
    path = _stored_png(tmp_path, Image.new("RGB", (400, 200)))

    capped = encode_image(path, ImageExportOptions(format=format, long_edge=100, metadata="all"))
    uncapped = _decode(encode_image(path, ImageExportOptions(format=format, long_edge=1000)))

    assert (_decode(capped).format, _decode(capped).size) == (format.upper(), (100, 50))
    assert uncapped.size == (400, 200)
    if format != "png":
        assert b"invokeai_" not in capped


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [("all", set(INVOKE_CHUNKS)), ("parameters", {"invokeai_metadata"}), ("none", set())],
)
def test_png_metadata_level_selects_invoke_chunks(
    tmp_path: Path, metadata: ImageExportMetadata, expected: set[str]
) -> None:
    path = _stored_png(tmp_path, Image.new("RGBA", (8, 8)))

    exported = _decode(encode_image(path, ImageExportOptions(format="png", metadata=metadata, long_edge=4)))

    assert {key for key in exported.info if key.startswith("invokeai_")} == expected


def test_hostile_board_name_and_pattern_cannot_form_a_path(mock_invoker: Invoker) -> None:
    board = mock_invoker.services.board_records.save("../../etc\\passwd", "system")
    image = ImageDTO(
        image_name="abc.png",
        board_id=board.board_id,
        image_url="unused",
        thumbnail_url="unused",
        width=8,
        height=8,
        image_origin=ResourceOrigin.INTERNAL,
        image_category=ImageCategory.GENERAL,
        created_at="2026-03-04 05:06:07.000",
        updated_at="2026-03-04 05:06:07.000",
        starred=False,
        has_workflow=False,
        is_intermediate=False,
    )

    [name] = resolve_export_file_names(
        mock_invoker.services, [image], "../{board}/..\\{name}", "png", BoardViewer(user_id="system", is_admin=False)
    )

    assert name == "etc-passwd-..-abc.png"
