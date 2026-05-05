import numpy as np
from PIL import Image, ImageDraw

from invokeai.app.invocations.detailer import paste_detailer_crop, prepare_detailer_crop, select_bounding_box
from invokeai.app.invocations.fields import BoundingBoxField
from invokeai.app.services.shared.graph import Graph


def _bbox(x_min: int, y_min: int, x_max: int, y_max: int, score: float | None = None) -> BoundingBoxField:
    return BoundingBoxField(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max, score=score)


def _mask(size: tuple[int, int], box: tuple[int, int, int, int] | None) -> Image.Image:
    mask = Image.new("L", size, color=0)
    if box is not None:
        draw = ImageDraw.Draw(mask)
        draw.rectangle((box[0], box[1], box[2] - 1, box[3] - 1), fill=255)
    return mask


def test_select_bounding_box_highest_score() -> None:
    boxes = [
        _bbox(0, 0, 10, 10, 0.2),
        _bbox(20, 0, 30, 10, 0.9),
        _bbox(40, 0, 50, 10, 0.4),
    ]

    assert select_bounding_box(boxes, "highest_score") == [boxes[1]]


def test_select_bounding_box_largest_area() -> None:
    boxes = [
        _bbox(0, 0, 10, 10, 0.9),
        _bbox(20, 0, 50, 20, 0.1),
        _bbox(40, 0, 50, 10, 0.8),
    ]

    assert select_bounding_box(boxes, "largest_area") == [boxes[1]]


def test_select_bounding_box_index_uses_left_to_right_order() -> None:
    left = _bbox(0, 0, 10, 10, 0.1)
    middle = _bbox(20, 0, 30, 10, 0.1)
    right = _bbox(40, 0, 50, 10, 0.1)

    assert select_bounding_box([right, left, middle], "index", 1) == [middle]


def test_select_bounding_box_invalid_index_or_empty_collection_returns_empty_selection() -> None:
    assert select_bounding_box([], "highest_score") == []
    assert select_bounding_box([_bbox(0, 0, 10, 10)], "index", 2) == []


def test_prepare_detailer_crop_no_mask_is_noop() -> None:
    image = Image.new("RGBA", (512, 512), color=(10, 20, 30, 255))
    result = prepare_detailer_crop(image=image, mask=_mask((512, 512), None))

    assert result.detected is False
    assert result.x == 0
    assert result.y == 0
    assert result.original_width == 64
    assert result.original_height == 64
    assert result.processed_width == 64
    assert result.processed_height == 64
    assert set(np.array(result.paste_alpha_mask).ravel()) == {0}
    assert set(np.array(result.denoise_mask).ravel()) == {255}


def test_prepare_detailer_crop_close_up_face_caps_processing_size() -> None:
    image = Image.new("RGBA", (1024, 1024), color=(10, 20, 30, 255))
    result = prepare_detailer_crop(
        image=image,
        mask=_mask((1024, 1024), (120, 120, 904, 904)),
        padding=96,
        target_size=512,
        max_upscale=4,
        max_process_size=768,
    )

    assert result.detected is True
    assert result.original_width == 976
    assert result.original_height == 976
    assert result.processed_width == 768
    assert result.processed_height == 768
    assert result.original_width % 8 == 0


def test_prepare_detailer_crop_tiny_face_upscales_toward_target() -> None:
    image = Image.new("RGBA", (1024, 1024), color=(10, 20, 30, 255))
    result = prepare_detailer_crop(
        image=image,
        mask=_mask((1024, 1024), (502, 120, 522, 144)),
        padding=64,
        target_size=512,
        max_upscale=4,
        max_process_size=768,
    )

    assert result.detected is True
    assert result.original_width == 152
    assert result.original_height == 152
    assert result.processed_width == 512
    assert result.processed_height == 512


def test_prepare_detailer_crop_edge_clamps_to_image_bounds() -> None:
    image = Image.new("RGBA", (512, 512), color=(10, 20, 30, 255))
    result = prepare_detailer_crop(
        image=image,
        mask=_mask((512, 512), (0, 8, 40, 48)),
        padding=64,
        target_size=512,
        max_upscale=4,
        max_process_size=768,
    )

    assert result.detected is True
    assert result.x == 0
    assert result.y == 0
    assert result.original_width % 8 == 0
    assert result.original_height % 8 == 0


def test_prepare_detailer_crop_mask_polarity_splits_denoise_and_paste_alpha() -> None:
    image = Image.new("RGBA", (256, 256), color=(10, 20, 30, 255))
    result = prepare_detailer_crop(
        image=image,
        mask=_mask((256, 256), (96, 96, 160, 160)),
        padding=32,
        denoise_mask_expand=0,
        denoise_mask_feather=0,
        paste_mask_expand=0,
        paste_mask_feather=0,
        target_size=64,
        max_upscale=1,
        max_process_size=64,
    )

    denoise_mask = np.array(result.denoise_mask)
    paste_alpha_mask = np.array(result.paste_alpha_mask)

    assert denoise_mask[32, 32] == 0
    assert denoise_mask[0, 0] == 255
    assert paste_alpha_mask[64, 64] == 255
    assert paste_alpha_mask[0, 0] == 0


def test_prepare_detailer_crop_uses_separate_denoise_and_paste_mask_settings() -> None:
    image = Image.new("RGBA", (256, 256), color=(10, 20, 30, 255))
    result = prepare_detailer_crop(
        image=image,
        mask=_mask((256, 256), (96, 96, 160, 160)),
        padding=32,
        denoise_mask_expand=16,
        denoise_mask_feather=0,
        paste_mask_expand=0,
        paste_mask_feather=0,
        target_size=128,
        max_upscale=1,
        max_process_size=128,
    )

    denoise_mask = np.array(result.denoise_mask)
    paste_alpha_mask = np.array(result.paste_alpha_mask)

    assert denoise_mask[17, 64] == 0
    assert paste_alpha_mask[17, 64] == 0


def test_prepare_detailer_crop_does_not_pre_feather_denoise_mask() -> None:
    image = Image.new("RGBA", (256, 256), color=(10, 20, 30, 255))
    result = prepare_detailer_crop(
        image=image,
        mask=_mask((256, 256), (96, 96, 160, 160)),
        padding=32,
        denoise_mask_expand=0,
        denoise_mask_feather=24,
        paste_mask_expand=0,
        paste_mask_feather=0,
        target_size=256,
        max_upscale=2,
        max_process_size=256,
    )

    assert set(np.array(result.denoise_mask).ravel()) == {0, 255}


def test_prepare_detailer_crop_paste_mask_feather_can_stay_soft() -> None:
    image = Image.new("RGBA", (256, 256), color=(10, 20, 30, 255))
    result = prepare_detailer_crop(
        image=image,
        mask=_mask((256, 256), (96, 96, 160, 160)),
        padding=32,
        paste_mask_expand=0,
        paste_mask_feather=12,
        target_size=128,
        max_upscale=1,
        max_process_size=128,
    )

    values = set(np.array(result.paste_alpha_mask).ravel())
    assert any(0 < value < 255 for value in values)


def test_paste_detailer_crop_alpha_mask_only_changes_masked_region() -> None:
    base = Image.new("RGBA", (128, 128), color=(10, 20, 30, 255))
    detail = Image.new("RGBA", (64, 64), color=(200, 10, 20, 255))
    alpha = _mask((64, 64), (16, 16, 48, 48))

    result = paste_detailer_crop(
        base_image=base,
        image=detail,
        paste_alpha_mask=alpha,
        x=32,
        y=32,
        original_width=64,
        original_height=64,
    )

    pixels = np.array(result)
    assert tuple(pixels[63, 63]) == (200, 10, 20, 255)
    assert tuple(pixels[32, 32]) == (10, 20, 30, 255)
    assert tuple(pixels[0, 0]) == (10, 20, 30, 255)


def test_paste_detailer_crop_semitransparent_alpha_uses_normal_compositing() -> None:
    base = Image.new("RGBA", (1, 1), color=(255, 255, 255, 255))
    detail = Image.new("RGBA", (1, 1), color=(0, 0, 0, 255))
    alpha = Image.new("L", (1, 1), color=128)

    result = paste_detailer_crop(
        base_image=base,
        image=detail,
        paste_alpha_mask=alpha,
        x=0,
        y=0,
        original_width=1,
        original_height=1,
    )

    assert list(result.getdata()) == list(Image.composite(detail, base, alpha).getdata())


def test_paste_detailer_crop_empty_alpha_mask_is_noop() -> None:
    base = Image.new("RGBA", (128, 128), color=(10, 20, 30, 255))
    detail = Image.new("RGBA", (64, 64), color=(200, 10, 20, 255))
    alpha = Image.new("L", (64, 64), color=0)

    result = paste_detailer_crop(
        base_image=base,
        image=detail,
        paste_alpha_mask=alpha,
        x=32,
        y=32,
        original_width=64,
        original_height=64,
    )

    assert list(result.getdata()) == list(base.getdata())


def test_detailer_invocations_validate_in_graph_schema() -> None:
    graph = Graph.model_validate(
        {
            "id": "detailer-validation",
            "nodes": {
                "select": {
                    "id": "select",
                    "type": "select_bounding_box",
                    "selection_mode": "highest_score",
                    "index": 0,
                },
                "crop": {
                    "id": "crop",
                    "type": "detailer_crop_from_mask",
                    "image": {"image_name": "base.png"},
                    "mask": {"image_name": "mask.png"},
                },
                "paste": {
                    "id": "paste",
                    "type": "detailer_paste_crop",
                    "base_image": {"image_name": "base.png"},
                    "image": {"image_name": "detail.png"},
                    "paste_alpha_mask": {"image_name": "mask.png"},
                },
            },
            "edges": [],
        }
    )

    assert graph.nodes["select"].type == "select_bounding_box"
    assert graph.nodes["crop"].type == "detailer_crop_from_mask"
    assert graph.nodes["paste"].type == "detailer_paste_crop"
