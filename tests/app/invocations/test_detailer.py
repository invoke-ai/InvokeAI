import numpy as np
from PIL import Image, ImageDraw

from invokeai.app.invocations.detailer import (
    _format_selected_bbox,
    _get_detail_delta_lines,
    _get_scale_mode,
    create_detailer_debug_collage,
    paste_detailer_crop,
    prepare_detailer_crop,
    select_bounding_box,
)
from invokeai.app.invocations.fields import BoundingBoxField
from invokeai.app.services.shared.graph import Graph


def _bbox(
    x_min: int, y_min: int, x_max: int, y_max: int, score: float | None = None, label: str | None = None
) -> BoundingBoxField:
    return BoundingBoxField(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max, score=score, label=label)


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


def test_select_bounding_box_label_priority_filters_before_selection() -> None:
    face_low = _bbox(0, 0, 10, 10, 0.2, "face")
    face_high = _bbox(20, 0, 30, 10, 0.7, "face")
    head_highest = _bbox(40, 0, 50, 10, 0.95, "head")

    assert select_bounding_box([head_highest, face_low, face_high], "highest_score", label_priority="face | head") == [
        face_high
    ]


def test_select_bounding_box_label_priority_falls_back_when_no_label_matches() -> None:
    face = _bbox(0, 0, 10, 10, 0.2, "face")
    head = _bbox(20, 0, 30, 10, 0.9, "head")

    assert select_bounding_box([face, head], "highest_score", label_priority="hand") == [head]


def test_select_bounding_box_empty_label_priority_keeps_default_behavior() -> None:
    face = _bbox(0, 0, 10, 10, 0.2, "face")
    head = _bbox(20, 0, 50, 20, 0.1, "head")

    assert select_bounding_box([face, head], "largest_area", label_priority="") == [head]


def test_debug_bbox_text_uses_width_height_and_score() -> None:
    assert _format_selected_bbox(_bbox(96, 80, 160, 144, 0.8123)) == "bbox 96,80 64x64 score 0.81"
    assert _format_selected_bbox(_bbox(96, 80, 160, 144, 0.8123, "face")) == ("bbox 96,80 64x64 score 0.81 label face")
    assert _format_selected_bbox(None) == "bbox none"


def test_debug_scale_mode_reports_scale_direction() -> None:
    assert _get_scale_mode(processed_width=1024, original_width=512) == "upscale"
    assert _get_scale_mode(processed_width=512, original_width=1024) == "downscale"
    assert _get_scale_mode(processed_width=1024, original_width=1024) == "no-scale"


def test_debug_detail_delta_reports_luma_and_rgb_change_in_edit_region() -> None:
    processed_crop = Image.new("RGBA", (4, 4), color=(10, 20, 30, 255))
    detailed_crop = Image.new("RGBA", (4, 4), color=(20, 30, 40, 255))
    denoise_mask = Image.new("L", (4, 4), color=0)

    assert _get_detail_delta_lines(
        processed_crop=processed_crop,
        detailed_crop=detailed_crop,
        denoise_mask=denoise_mask,
    ) == ["luma delta +10.0", "rgb delta +10.0,+10.0,+10.0"]


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


def test_prepare_detailer_crop_balanced_and_high_preserve_1024_body_crop() -> None:
    image = Image.new("RGBA", (1024, 1024), color=(10, 20, 30, 255))
    mask = _mask((1024, 1024), (0, 0, 1024, 1024))
    balanced = prepare_detailer_crop(
        image=image,
        mask=mask,
        padding=0,
        target_size=768,
        max_upscale=8,
        max_process_size=1024,
    )
    high = prepare_detailer_crop(
        image=image,
        mask=mask,
        padding=0,
        target_size=1024,
        max_upscale=12,
        max_process_size=1024,
    )

    assert balanced.original_width == 1024
    assert balanced.processed_width == 1024
    assert high.original_width == 1024
    assert high.processed_width == 1024


def test_prepare_detailer_crop_fast_still_caps_1024_body_crop_at_512() -> None:
    image = Image.new("RGBA", (1024, 1024), color=(10, 20, 30, 255))
    result = prepare_detailer_crop(
        image=image,
        mask=_mask((1024, 1024), (0, 0, 1024, 1024)),
        padding=0,
        target_size=512,
        max_upscale=4,
        max_process_size=512,
    )

    assert result.original_width == 1024
    assert result.processed_width == 512


def test_prepare_detailer_crop_prevent_downscale_still_respects_max_process_cap() -> None:
    image = Image.new("RGBA", (1024, 1024), color=(10, 20, 30, 255))
    result = prepare_detailer_crop(
        image=image,
        mask=_mask((1024, 1024), (0, 0, 1024, 1024)),
        padding=0,
        target_size=512,
        max_upscale=4,
        max_process_size=512,
        prevent_downscale=True,
    )

    assert result.original_width == 1024
    assert result.processed_width == 512


def test_prepare_detailer_crop_prevent_downscale_keeps_crop_when_under_max_process_cap() -> None:
    image = Image.new("RGBA", (1024, 1024), color=(10, 20, 30, 255))
    result = prepare_detailer_crop(
        image=image,
        mask=_mask((1024, 1024), (0, 0, 1024, 1024)),
        padding=0,
        target_size=512,
        max_upscale=4,
        max_process_size=1024,
        prevent_downscale=True,
    )

    assert result.original_width == 1024
    assert result.processed_width == 1024


def test_prepare_detailer_crop_person_high_can_supersample_1024_crop_to_1536() -> None:
    image = Image.new("RGBA", (1024, 1024), color=(10, 20, 30, 255))
    result = prepare_detailer_crop(
        image=image,
        mask=_mask((1024, 1024), (0, 0, 1024, 1024)),
        padding=0,
        target_size=1536,
        max_upscale=12,
        max_process_size=1536,
        prevent_downscale=True,
    )

    assert result.original_width == 1024
    assert result.processed_width == 1536


def test_prepare_detailer_crop_oversized_crop_caps_at_max_process_size() -> None:
    image = Image.new("RGBA", (1536, 1536), color=(10, 20, 30, 255))
    result = prepare_detailer_crop(
        image=image,
        mask=_mask((1536, 1536), (0, 0, 1536, 1536)),
        padding=0,
        target_size=1024,
        max_upscale=12,
        max_process_size=1024,
    )

    assert result.original_width == 1536
    assert result.processed_width == 1024


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


def test_prepare_detailer_crop_mask_contract_protects_silhouette_edges() -> None:
    image = Image.new("RGBA", (128, 128), color=(10, 20, 30, 255))
    result = prepare_detailer_crop(
        image=image,
        mask=_mask((128, 128), (32, 32, 96, 96)),
        padding=16,
        denoise_mask_expand=0,
        denoise_mask_contract=4,
        paste_mask_expand=0,
        paste_mask_contract=2,
        paste_mask_feather=0,
        target_size=96,
        max_upscale=1,
        max_process_size=96,
    )

    denoise_mask = np.array(result.denoise_mask)
    paste_alpha_mask = np.array(result.paste_alpha_mask)

    assert denoise_mask[16, 48] == 255
    assert denoise_mask[20, 48] == 0
    assert paste_alpha_mask[16, 48] == 0
    assert paste_alpha_mask[18, 48] == 255


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


def test_create_detailer_debug_collage_detected_path() -> None:
    base = Image.new("RGBA", (256, 256), color=(10, 20, 30, 255))
    mask = _mask((256, 256), (96, 96, 160, 160))
    crop = Image.new("RGBA", (128, 128), color=(40, 50, 60, 255))
    detail = Image.new("RGBA", (128, 128), color=(200, 210, 220, 255))
    final = paste_detailer_crop(
        base_image=base,
        image=detail,
        paste_alpha_mask=_mask((64, 64), (8, 8, 56, 56)),
        x=80,
        y=80,
        original_width=64,
        original_height=64,
    )

    collage = create_detailer_debug_collage(
        base_image=base,
        selected_bounding_boxes=[_bbox(96, 96, 160, 160, 0.8)],
        mask=mask,
        processed_crop=crop,
        denoise_mask=_mask((128, 128), (32, 32, 96, 96)),
        paste_alpha_mask=_mask((64, 64), (8, 8, 56, 56)),
        detailed_crop=detail,
        final_image=final,
        x=80,
        y=80,
        original_width=64,
        original_height=64,
        processed_width=128,
        processed_height=128,
        detected=True,
        target_prompt="face",
        detector_prompt="face | head",
        sam_model="segment-anything-2-large",
    )

    assert collage.size == (1260, 660)
    assert collage.getbbox() is not None


def test_create_detailer_debug_collage_no_target_path() -> None:
    base = Image.new("RGBA", (128, 128), color=(10, 20, 30, 255))
    noop_mask = Image.new("L", (128, 128), color=0)
    noop_crop = base.crop((0, 0, 64, 64))

    collage = create_detailer_debug_collage(
        base_image=base,
        selected_bounding_boxes=[],
        mask=noop_mask,
        processed_crop=noop_crop,
        denoise_mask=Image.new("L", (64, 64), color=255),
        paste_alpha_mask=Image.new("L", (64, 64), color=0),
        detailed_crop=noop_crop,
        final_image=base,
        x=0,
        y=0,
        original_width=64,
        original_height=64,
        processed_width=64,
        processed_height=64,
        detected=False,
        target_prompt="belt",
    )

    assert collage.size == (1260, 660)
    assert collage.mode == "RGBA"


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
                    "label_priority": "face | head",
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
                "debug": {
                    "id": "debug",
                    "type": "detailer_debug_collage",
                    "base_image": {"image_name": "base.png"},
                    "selected_bounding_boxes": [
                        {"x_min": 0, "y_min": 0, "x_max": 32, "y_max": 32, "score": 0.8, "label": "face"}
                    ],
                    "mask": {"image_name": "mask.png"},
                    "processed_crop": {"image_name": "crop.png"},
                    "denoise_mask": {"image_name": "denoise-mask.png"},
                    "paste_alpha_mask": {"image_name": "paste-mask.png"},
                    "detailed_crop": {"image_name": "detail.png"},
                    "final_image": {"image_name": "final.png"},
                    "x": 0,
                    "y": 0,
                    "original_width": 64,
                    "original_height": 64,
                    "processed_width": 128,
                    "processed_height": 128,
                    "detected": True,
                    "target_prompt": "face",
                    "detector_prompt": "face | head",
                    "sam_model": "segment-anything-2-large",
                },
            },
            "edges": [],
        }
    )

    assert graph.nodes["select"].type == "select_bounding_box"
    assert graph.nodes["crop"].type == "detailer_crop_from_mask"
    assert graph.nodes["paste"].type == "detailer_paste_crop"
    assert graph.nodes["debug"].type == "detailer_debug_collage"
