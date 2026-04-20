"""Tests for SaveImageToFileInvocation."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image
from pydantic import ValidationError

from invokeai.app.invocations.image import SaveImageToFileInvocation


def _make_context(tmp_path: Path, pil_image: Image.Image, gallery_uuid: str = "abc123") -> MagicMock:
    context = MagicMock()
    context.config.get.return_value.outputs_path = tmp_path
    context.images.get_pil.return_value = pil_image
    image_dto = MagicMock()
    image_dto.image_name = f"{gallery_uuid}.png"
    image_dto.width = pil_image.width
    image_dto.height = pil_image.height
    context.images.save.return_value = image_dto
    return context


def _build_node(**overrides) -> SaveImageToFileInvocation:
    defaults = {
        "id": "test",
        "image": {"image_name": "input.png"},
    }
    defaults.update(overrides)
    return SaveImageToFileInvocation(**defaults)


class TestSaveImageToFileInvocation:
    def test_saves_to_gallery(self, tmp_path):
        img = Image.new("RGB", (8, 8), (255, 0, 0))
        ctx = _make_context(tmp_path, img)
        node = _build_node()

        node.invoke(ctx)

        ctx.images.save.assert_called_once()
        assert ctx.images.save.call_args.kwargs["image"] is img

    def test_default_directory_is_outputs_root(self, tmp_path):
        img = Image.new("RGB", (8, 8))
        ctx = _make_context(tmp_path, img, gallery_uuid="uuid-1")
        node = _build_node()

        node.invoke(ctx)

        assert (tmp_path / "uuid-1.png").exists()

    def test_relative_subdirectory_created(self, tmp_path):
        img = Image.new("RGB", (8, 8))
        ctx = _make_context(tmp_path, img, gallery_uuid="uuid-2")
        node = _build_node(output_directory="my-exports")

        node.invoke(ctx)

        assert (tmp_path / "my-exports" / "uuid-2.png").exists()

    def test_nested_relative_path(self, tmp_path):
        img = Image.new("RGB", (8, 8))
        ctx = _make_context(tmp_path, img, gallery_uuid="uuid-3")
        node = _build_node(output_directory="exports/2026/hero")

        node.invoke(ctx)

        assert (tmp_path / "exports" / "2026" / "hero" / "uuid-3.png").exists()

    def test_prefix_and_suffix_applied(self, tmp_path):
        img = Image.new("RGB", (8, 8))
        ctx = _make_context(tmp_path, img, gallery_uuid="xyz")
        node = _build_node(prefix="hero_", suffix="_final")

        node.invoke(ctx)

        assert (tmp_path / "hero_xyz_final.png").exists()

    def test_prefix_only(self, tmp_path):
        img = Image.new("RGB", (8, 8))
        ctx = _make_context(tmp_path, img, gallery_uuid="u")
        node = _build_node(prefix="p_")

        node.invoke(ctx)

        assert (tmp_path / "p_u.png").exists()

    def test_suffix_only(self, tmp_path):
        img = Image.new("RGB", (8, 8))
        ctx = _make_context(tmp_path, img, gallery_uuid="u")
        node = _build_node(suffix="_s")

        node.invoke(ctx)

        assert (tmp_path / "u_s.png").exists()

    def test_filename_uses_gallery_uuid_not_input_uuid(self, tmp_path):
        """The exported filename must use the UUID from the new gallery entry,
        not the UUID of the input image_name."""
        img = Image.new("RGB", (8, 8))
        ctx = _make_context(tmp_path, img, gallery_uuid="new-uuid")
        node = _build_node(image={"image_name": "old-uuid.png"})

        node.invoke(ctx)

        assert (tmp_path / "new-uuid.png").exists()
        assert not (tmp_path / "old-uuid.png").exists()

    @pytest.mark.parametrize(
        "bad_path",
        [
            "D:/Pictures/Invoke",
            "C:/Windows",
            "/etc/passwd",
            "/tmp/foo",
        ],
    )
    def test_absolute_paths_rejected(self, tmp_path, bad_path):
        img = Image.new("RGB", (8, 8))
        ctx = _make_context(tmp_path, img)
        node = _build_node(output_directory=bad_path)

        with pytest.raises(ValueError, match="[Aa]bsolute"):
            node.invoke(ctx)

    @pytest.mark.parametrize(
        "traversal_path",
        [
            "../outside",
            "subdir/../../outside",
            "..",
        ],
    )
    def test_path_traversal_rejected(self, tmp_path, traversal_path):
        img = Image.new("RGB", (8, 8))
        ctx = _make_context(tmp_path, img)
        node = _build_node(output_directory=traversal_path)

        with pytest.raises(ValueError):
            node.invoke(ctx)

    def test_png_format(self, tmp_path):
        img = Image.new("RGBA", (8, 8), (10, 20, 30, 128))
        ctx = _make_context(tmp_path, img, gallery_uuid="u")
        node = _build_node(file_format="png")

        node.invoke(ctx)

        path = tmp_path / "u.png"
        assert path.exists()
        with Image.open(path) as saved:
            assert saved.format == "PNG"
            assert saved.mode == "RGBA"

    def test_jpg_format_converts_rgba_to_rgb(self, tmp_path):
        img = Image.new("RGBA", (8, 8), (10, 20, 30, 128))
        ctx = _make_context(tmp_path, img, gallery_uuid="u")
        node = _build_node(file_format="jpg", quality=80)

        node.invoke(ctx)

        path = tmp_path / "u.jpg"
        assert path.exists()
        with Image.open(path) as saved:
            assert saved.format == "JPEG"
            assert saved.mode == "RGB"

    def test_webp_format(self, tmp_path):
        img = Image.new("RGB", (8, 8))
        ctx = _make_context(tmp_path, img, gallery_uuid="u")
        node = _build_node(file_format="webp", quality=75)

        node.invoke(ctx)

        path = tmp_path / "u.webp"
        assert path.exists()
        with Image.open(path) as saved:
            assert saved.format == "WEBP"

    def test_quality_bounds_enforced_by_pydantic(self):
        with pytest.raises(ValidationError):
            _build_node(quality=0)
        with pytest.raises(ValidationError):
            _build_node(quality=101)

    def test_output_is_pass_through_of_gallery_dto(self, tmp_path):
        img = Image.new("RGB", (16, 24))
        ctx = _make_context(tmp_path, img, gallery_uuid="uuid-out")
        node = _build_node()

        result = node.invoke(ctx)

        assert result.image.image_name == "uuid-out.png"
        assert result.width == 16
        assert result.height == 24
