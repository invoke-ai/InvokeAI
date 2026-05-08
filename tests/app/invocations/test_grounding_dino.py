from invokeai.app.invocations.grounding_dino import (
    detection_result_to_bounding_box,
    normalize_grounding_dino_label,
    parse_grounding_dino_labels,
)
from invokeai.backend.image_util.grounding_dino.detection_result import BoundingBox, DetectionResult


def test_parse_grounding_dino_labels_splits_supported_delimiters() -> None:
    assert parse_grounding_dino_labels("face") == ["face"]
    assert parse_grounding_dino_labels("face | head") == ["face", "head"]
    assert parse_grounding_dino_labels("face, head") == ["face", "head"]
    assert parse_grounding_dino_labels("face. head") == ["face", "head"]


def test_parse_grounding_dino_labels_strips_empty_labels_and_duplicates() -> None:
    assert parse_grounding_dino_labels(" face || head, face. ") == ["face", "head"]


def test_normalize_grounding_dino_label_strips_punctuation_and_case() -> None:
    assert normalize_grounding_dino_label(" Face. ") == "face"


def test_detection_result_to_bounding_box_preserves_normalized_label() -> None:
    bbox = detection_result_to_bounding_box(
        DetectionResult(score=0.82, label="Face.", box=BoundingBox(xmin=1, ymin=2, xmax=11, ymax=22))
    )

    assert bbox.x_min == 1
    assert bbox.y_min == 2
    assert bbox.x_max == 11
    assert bbox.y_max == 22
    assert bbox.score == 0.82
    assert bbox.label == "face"
