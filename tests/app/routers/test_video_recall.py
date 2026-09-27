"""Tests for the video recall router (`/api/v1/recall/video`).

Media records live in the real SQLite stores; the model manager is replaced by a small in-memory
store because building installable model configs is beside the point here. Each test reads the
`video_recall_requested` event the router emits, which is the contract the frontend consumes.
"""

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api.routers import videos
from invokeai.app.services.board_records.board_records_common import BoardChanges, BoardVisibility
from invokeai.app.services.events.events_common import VideoRecallRequestedEvent, VideoUploadedEvent
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.model_records.model_records_base import UnknownModelException
from invokeai.app.services.users.users_common import UserCreateRequest
from invokeai.app.services.video_records.video_records_common import VideoRecordNotFoundException
from invokeai.app.services.videos.videos_common import VideoDTO
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType

MP4_BYTES = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 12


class MockApiDependencies(ApiDependencies):
    invoker: Invoker

    def __init__(self, invoker: Invoker) -> None:
        self.invoker = invoker


def _model(key: str, name: str, base: BaseModelType, model_type: ModelType) -> SimpleNamespace:
    return SimpleNamespace(key=key, hash=f"blake3:{key}", name=name, base=base, type=model_type)


class FakeModelStore:
    """The two lookups the router makes, over a fixed catalog in install order."""

    def __init__(self, models: list[SimpleNamespace]) -> None:
        self._models = models

    def get_model(self, key: str) -> SimpleNamespace:
        for model in self._models:
            if model.key == key:
                return model
        raise UnknownModelException(key)

    def search_by_attr(self, model_name: str, model_type: ModelType) -> list[SimpleNamespace]:
        return [model for model in self._models if model.name == model_name and model.type == model_type]


CATALOG = [
    # Installed first, so an unguided name lookup would pick the SDXL LoRA over the Wan one.
    _model("sdxl-lora", "detail", BaseModelType.StableDiffusionXL, ModelType.LoRA),
    _model("wan-lora", "detail", BaseModelType.Wan, ModelType.LoRA),
    _model("wan-main", "Wan 2.2 I2V", BaseModelType.Wan, ModelType.Main),
    _model("wan-t5", "UMT5 XXL", BaseModelType.Any, ModelType.WanT5Encoder),
    _model("h3-ref2va", "MiniMax H3 Ref2VA", BaseModelType.MiniMaxH3, ModelType.Main),
    _model("h3-qwen", "Qwen3-VL", BaseModelType.MiniMaxH3, ModelType.Qwen3VLEncoder),
    # Generic encoders install as base "any"; the H3 and LTX-2 slots must not take them.
    _model("any-qwen", "Qwen3-VL Generic", BaseModelType.Any, ModelType.Qwen3VLEncoder),
    _model("any-gemma", "Gemma Generic", BaseModelType.Any, ModelType.Gemma4Encoder),
    _model("ltx-main", "LTX-2.3", BaseModelType.LTX2, ModelType.Main),
    _model("ltx-gemma", "Gemma 4", BaseModelType.LTX2, ModelType.Gemma4Encoder),
    _model("sdxl-vae", "fp16 VAE", BaseModelType.StableDiffusionXL, ModelType.VAE),
    _model("wan-vae", "Wan VAE", BaseModelType.Wan, ModelType.VAE),
    _model("sdxl-main", "SDXL Base", BaseModelType.StableDiffusionXL, ModelType.Main),
    # Two mains sharing a name: the SDXL one is installed first.
    _model("sdxl-shared", "Shared", BaseModelType.StableDiffusionXL, ModelType.Main),
    _model("ltx-shared", "Shared", BaseModelType.LTX2, ModelType.Main),
]


def _video_dto(video_name: str, **update: Any) -> VideoDTO:
    return VideoDTO(
        video_name=video_name,
        video_origin=ResourceOrigin.EXTERNAL,
        video_category=ImageCategory.GENERAL,
        width=832,
        height=480,
        duration=5.0,
        fps=16.0,
        created_at="None",
        updated_at="None",
        is_intermediate=False,
        starred=False,
        has_workflow=False,
        video_url=f"/videos/{video_name}",
        thumbnail_url=f"/thumbnails/{video_name}.webp",
        board_id=None,
    ).model_copy(update=update)


def _save_image(invoker: Invoker, image_name: str, user_id: str = "system") -> None:
    invoker.services.image_records.save(
        image_name=image_name,
        image_origin=ResourceOrigin.INTERNAL,
        image_category=ImageCategory.GENERAL,
        width=64,
        height=64,
        has_workflow=False,
        user_id=user_id,
    )


def _save_video(invoker: Invoker, video_name: str, user_id: str = "system") -> None:
    invoker.services.video_records.save(
        video_name=video_name,
        video_origin=ResourceOrigin.INTERNAL,
        video_category=ImageCategory.GENERAL,
        width=832,
        height=480,
        duration=5.0,
        fps=16.0,
        has_workflow=False,
        user_id=user_id,
    )


@pytest.fixture
def invoker(monkeypatch: Any, mock_invoker: Invoker) -> Invoker:
    mock_invoker.services.model_manager = SimpleNamespace(store=FakeModelStore(CATALOG))  # type: ignore[assignment]
    mock_invoker.services.image_moves = MagicMock()
    mock_invoker.services.image_moves.is_maintenance_active.return_value = False

    def get_dto(video_name: str) -> VideoDTO:
        if not mock_invoker.services.video_records.exists(video_name):
            raise VideoRecordNotFoundException(video_name)
        return _video_dto(video_name)

    mock_invoker.services.videos = MagicMock()
    mock_invoker.services.videos.get_dto.side_effect = get_dto

    dependencies = MockApiDependencies(mock_invoker)
    for target in (
        "invokeai.app.api.routers.video_recall.ApiDependencies",
        "invokeai.app.api.routers.auth.ApiDependencies",
        "invokeai.app.api.routers.videos.ApiDependencies",
        "invokeai.app.api.routers._access.ApiDependencies",
        "invokeai.app.api.routers.image_move_maintenance.ApiDependencies",
        "invokeai.app.api.auth_dependencies.ApiDependencies",
        "invokeai.app.api_app.ApiDependencies",
    ):
        monkeypatch.setattr(target, dependencies)
    return mock_invoker


def _recall_events(invoker: Invoker) -> list[VideoRecallRequestedEvent]:
    return [
        event
        for event in invoker.services.events.events  # type: ignore[attr-defined]
        if isinstance(event, VideoRecallRequestedEvent)
    ]


def _only_recall_event(invoker: Invoker) -> VideoRecallRequestedEvent:
    events = _recall_events(invoker)
    assert len(events) == 1
    return events[0]


def _identifier(key: str) -> dict[str, Any]:
    model = next(model for model in CATALOG if model.key == key)
    return {"key": key, "hash": f"blake3:{key}", "name": model.name, "base": model.base.value, "type": model.type.value}


class TestParameterRecall:
    def test_models_resolve_to_the_identifiers_a_metadata_record_carries(
        self, invoker: Invoker, client: TestClient
    ) -> None:
        response = client.post(
            "/api/v1/recall/video/default",
            json={
                "positive_prompt": "a heron takes flight",
                "seed": 42,
                "num_frames": 81,
                "model": "Wan 2.2 I2V",
                "wan_t5_encoder_model": "UMT5 XXL",
                "loras": [{"model_name": "detail", "weight": 0.6}],
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success"
        assert body["skipped"] == []
        event = _only_recall_event(invoker)
        assert event.action == "parameters"
        assert event.mode == "recall"
        assert event.strict is False
        assert event.user_id == "system"
        assert event.parameters == {
            "positive_prompt": "a heron takes flight",
            "seed": 42,
            "num_frames": 81,
            "model": _identifier("wan-main"),
            "wan_t5_encoder_model": _identifier("wan-t5"),
            # The Wan LoRA wins over the same-named SDXL LoRA installed before it.
            "loras": [{"model": _identifier("wan-lora"), "weight": 0.6}],
        }
        assert body["parameters"] == event.parameters

    def test_a_model_key_is_accepted_in_place_of_a_name(self, invoker: Invoker, client: TestClient) -> None:
        response = client.post("/api/v1/recall/video/default", json={"model": "h3-ref2va"})

        assert response.status_code == 200
        assert _only_recall_event(invoker).parameters == {"model": _identifier("h3-ref2va")}

    def test_a_key_of_the_wrong_model_type_does_not_fill_the_slot(self, invoker: Invoker, client: TestClient) -> None:
        response = client.post("/api/v1/recall/video/default", json={"model": "wan-lora", "positive_prompt": "p"})

        assert response.json()["skipped"] == ["model"]
        assert _only_recall_event(invoker).parameters == {"positive_prompt": "p"}

    def test_unresolvable_models_are_skipped_and_the_rest_applied(self, invoker: Invoker, client: TestClient) -> None:
        response = client.post(
            "/api/v1/recall/video/default",
            json={
                "positive_prompt": "p",
                "model": "Not Installed",
                "loras": [{"model_name": "missing"}, {"model_name": "detail", "weight": 1.2}],
            },
        )

        body = response.json()
        assert body["status"] == "success"
        assert body["skipped"] == ["model", "loras[0]"]
        assert _only_recall_event(invoker).parameters == {
            "positive_prompt": "p",
            "loras": [{"model": _identifier("wan-lora"), "weight": 1.2}],
        }

    @pytest.mark.parametrize(
        ("slot", "value", "expected_key"),
        [
            ("model", "Shared", "ltx-shared"),
            ("vae", "Wan VAE", "wan-vae"),
            ("minimax_h3_text_encoder_model", "Qwen3-VL", "h3-qwen"),
            ("minimax_h3_component_source", "MiniMax H3 Ref2VA", "h3-ref2va"),
            ("ltx2_component_source", "LTX-2.3", "ltx-main"),
            ("ltx2_text_encoder_model", "Gemma 4", "ltx-gemma"),
        ],
    )
    def test_each_slot_resolves_a_model_of_its_own_family(
        self, invoker: Invoker, client: TestClient, slot: str, value: str, expected_key: str
    ) -> None:
        response = client.post("/api/v1/recall/video/default", json={slot: value})

        assert response.json()["skipped"] == []
        assert _only_recall_event(invoker).parameters == {slot: _identifier(expected_key)}

    @pytest.mark.parametrize(
        ("slot", "value"),
        [
            ("model", "SDXL Base"),
            ("model", "sdxl-main"),
            ("vae", "fp16 VAE"),
            ("wan_component_source", "LTX-2.3"),
            ("minimax_h3_hybrid_base_model", "Wan 2.2 I2V"),
            ("ltx2_component_source", "ltx-shared-missing"),
            ("wan_t5_encoder_model", "Gemma 4"),
            ("minimax_h3_text_encoder_model", "Qwen3-VL Generic"),
            ("ltx2_text_encoder_model", "Gemma Generic"),
            ("ltx2_component_source", "MiniMax H3 Ref2VA"),
            ("wan_transformer_low_noise", "LTX-2.3"),
        ],
        ids=[
            "non-video-main",
            "non-video-main-key",
            "sdxl-vae",
            "ltx-as-wan-source",
            "wan-as-h3-base",
            "missing",
            "wrong-type",
            "generic-qwen-as-h3-encoder",
            "generic-gemma-as-ltx2-encoder",
            "h3-as-ltx2-source",
            "ltx2-as-wan-low-noise",
        ],
    )
    def test_a_model_from_another_family_is_skipped_not_delivered(
        self, invoker: Invoker, client: TestClient, slot: str, value: str
    ) -> None:
        response = client.post("/api/v1/recall/video/default", json={slot: value, "positive_prompt": "p"})

        assert response.json()["skipped"] == [slot]
        assert _only_recall_event(invoker).parameters == {"positive_prompt": "p"}

    def test_a_lora_of_another_family_is_skipped(self, invoker: Invoker, client: TestClient) -> None:
        response = client.post(
            "/api/v1/recall/video/default", json={"loras": [{"model_name": "sdxl-lora"}, {"model_name": "wan-lora"}]}
        )

        assert response.json()["skipped"] == ["loras[0]"]
        assert _only_recall_event(invoker).parameters == {"loras": [{"model": _identifier("wan-lora"), "weight": 1.0}]}

    @pytest.mark.parametrize(
        "field",
        [
            {"loras": [{"model_name": "missing"}]},
            {"minimax_h3_references": [{"kind": "image", "image_name": "gone.png"}]},
        ],
        ids=["loras", "references"],
    )
    def test_a_list_whose_every_entry_was_dropped_leaves_the_panel_list_alone(
        self, invoker: Invoker, client: TestClient, field: dict[str, Any]
    ) -> None:
        """Asking to add a missing LoRA must not turn into "clear every LoRA"."""
        response = client.post("/api/v1/recall/video/default", json={"positive_prompt": "p", **field})

        assert response.json()["skipped"] == [f"{next(iter(field))}[0]"]
        assert _only_recall_event(invoker).parameters == {"positive_prompt": "p"}

    def test_an_explicit_null_negative_prompt_is_delivered_to_turn_it_off(
        self, invoker: Invoker, client: TestClient
    ) -> None:
        client.post("/api/v1/recall/video/default", json={"negative_prompt": None})

        assert _only_recall_event(invoker).parameters == {"negative_prompt": None}

    @pytest.mark.parametrize("field", ["loras", "minimax_h3_references"])
    def test_an_empty_list_is_delivered_as_a_request_for_none(
        self, invoker: Invoker, client: TestClient, field: str
    ) -> None:
        client.post("/api/v1/recall/video/default", json={field: []})

        assert _only_recall_event(invoker).parameters == {field: []}

    def test_remix_never_delivers_the_seed(self, invoker: Invoker, client: TestClient) -> None:
        response = client.post(
            "/api/v1/recall/video/default", params={"mode": "remix"}, json={"positive_prompt": "p", "seed": 7}
        )

        assert response.json()["parameters"] == {"positive_prompt": "p"}
        event = _only_recall_event(invoker)
        assert event.mode == "remix"
        assert event.parameters == {"positive_prompt": "p"}

    def test_strict_is_carried_to_the_frontend(self, invoker: Invoker, client: TestClient) -> None:
        client.post("/api/v1/recall/video/default", params={"strict": True}, json={"positive_prompt": "p"})

        assert _only_recall_event(invoker).strict is True

    def test_gallery_media_is_delivered_and_missing_media_skipped_with_its_qualifiers(
        self, invoker: Invoker, client: TestClient
    ) -> None:
        _save_image(invoker, "last.png")
        _save_image(invoker, "ref.png")
        _save_video(invoker, "ref.mp4")

        response = client.post(
            "/api/v1/recall/video/default",
            json={
                "first_frame_image": {"image_name": "gone.png"},
                "last_frame_image": {"image_name": "last.png"},
                "source_video": {"video_name": "gone.mp4"},
                "source_video_start_frame": 0,
                "source_video_end_frame": 40,
                "minimax_h3_references": [
                    {
                        "kind": "video",
                        "video_name": "ref.mp4",
                        "conditioning": "video",
                        "start_frame": 0,
                        "end_frame": 8,
                    },
                    {"kind": "image", "image_name": "gone.png"},
                    {"kind": "image", "image_name": "ref.png", "detail": "max"},
                ],
            },
        )

        assert response.json()["skipped"] == [
            "first_frame_image",
            "source_video",
            "minimax_h3_references[1]",
            "source_video_start_frame",
            "source_video_end_frame",
        ]
        # References replace the frame slots, so the last frame is reported as overridden rather than sent.
        assert response.json()["overridden"] == {"last_frame_image": "minimax_h3_references"}
        assert _only_recall_event(invoker).parameters == {
            "minimax_h3_references": [
                {"kind": "video", "video_name": "ref.mp4", "conditioning": "video", "start_frame": 0, "end_frame": 8},
                {"kind": "image", "image_name": "ref.png", "detail": "max"},
            ],
        }

    @pytest.mark.parametrize(
        "body",
        [
            {"source_video": {"video_name": "clip.mp4"}, "source_video_start_frame": 8, "source_video_end_frame": 40},
            {"ltx2_conditioning_video": {"video_name": "clip.mp4"}, "ltx2_conditioning_role": "audio"},
        ],
        ids=["source-video-trim", "conditioning-role"],
    )
    def test_trims_and_roles_travel_with_their_media(
        self, invoker: Invoker, client: TestClient, body: dict[str, Any]
    ) -> None:
        _save_video(invoker, "clip.mp4")

        response = client.post("/api/v1/recall/video/default", json=body)

        assert response.json()["skipped"] == []
        assert response.json()["overridden"] == {}
        assert _only_recall_event(invoker).parameters == body

    def test_a_missing_conditioning_video_takes_its_role_with_it(self, invoker: Invoker, client: TestClient) -> None:
        response = client.post(
            "/api/v1/recall/video/default",
            json={
                "positive_prompt": "p",
                "ltx2_conditioning_video": {"video_name": "gone.mp4"},
                "ltx2_conditioning_role": "video",
            },
        )

        assert response.json()["skipped"] == ["ltx2_conditioning_video", "ltx2_conditioning_role"]
        assert _only_recall_event(invoker).parameters == {"positive_prompt": "p"}

    def test_trims_sent_without_their_video_are_reported_as_skipped(self, invoker: Invoker, client: TestClient) -> None:
        response = client.post(
            "/api/v1/recall/video/default", json={"source_video_start_frame": 0, "source_video_end_frame": 8}
        )

        assert response.json()["status"] == "nothing_resolved"
        assert response.json()["skipped"] == ["source_video_start_frame", "source_video_end_frame"]
        assert _recall_events(invoker) == []

    def test_the_event_carries_the_requested_queue(self, invoker: Invoker, client: TestClient) -> None:
        client.post("/api/v1/recall/video/queue-b", json={"positive_prompt": "p"})

        assert _only_recall_event(invoker).queue_id == "queue-b"

    @pytest.mark.parametrize(
        ("body", "sent", "overridden"),
        [
            (
                {"first_frame_image": {"image_name": "first.png"}, "source_video": {"video_name": "clip.mp4"}},
                {"source_video"},
                {"first_frame_image": "source_video"},
            ),
            (
                {
                    "first_frame_image": {"image_name": "first.png"},
                    "last_frame_image": {"image_name": "last.png"},
                    "source_video": {"video_name": "clip.mp4"},
                    "minimax_h3_references": [{"kind": "image", "image_name": "ref.png"}],
                },
                {"source_video", "minimax_h3_references"},
                {"first_frame_image": "minimax_h3_references", "last_frame_image": "minimax_h3_references"},
            ),
            (
                {
                    "first_frame_image": {"image_name": "first.png"},
                    "source_video": {"video_name": "clip.mp4"},
                    "source_video_start_frame": 0,
                    "source_video_end_frame": 8,
                    "minimax_h3_references": [{"kind": "image", "image_name": "ref.png"}],
                    "ltx2_conditioning_video": {"video_name": "song.mp4"},
                    "ltx2_conditioning_role": "audio",
                },
                {"ltx2_conditioning_video", "ltx2_conditioning_role"},
                {
                    "first_frame_image": "ltx2_conditioning_video",
                    "source_video": "ltx2_conditioning_video",
                    "source_video_start_frame": "ltx2_conditioning_video",
                    "source_video_end_frame": "ltx2_conditioning_video",
                    "minimax_h3_references": "ltx2_conditioning_video",
                },
            ),
            (
                {"first_frame_image": {"image_name": "first.png"}, "minimax_h3_references": []},
                {"first_frame_image", "minimax_h3_references"},
                {},
            ),
            (
                {
                    "ltx2_conditioning_video": {"video_name": "song.mp4"},
                    "ltx2_conditioning_role": "audio",
                    "minimax_h3_references": [],
                },
                {"ltx2_conditioning_video", "ltx2_conditioning_role", "minimax_h3_references"},
                {},
            ),
        ],
        ids=[
            "initial-video-over-first-frame",
            "references-over-frames",
            "conditioning-clip-over-all",
            "empty-list-beside-a-frame",
            "empty-list-beside-a-clip",
        ],
    )
    def test_media_a_higher_precedence_medium_excludes_is_reported_not_sent(
        self,
        invoker: Invoker,
        client: TestClient,
        body: dict[str, Any],
        sent: set[str],
        overridden: dict[str, str],
    ) -> None:
        """The Video panel reads media in this precedence; sending a loser would be silently ignored there."""
        for image_name in ("first.png", "last.png", "ref.png"):
            _save_image(invoker, image_name)
        for video_name in ("clip.mp4", "song.mp4"):
            _save_video(invoker, video_name)

        response = client.post("/api/v1/recall/video/default", json=body)

        assert response.json()["overridden"] == overridden
        assert response.json()["skipped"] == []
        assert set(_only_recall_event(invoker).parameters) == sent

    def test_a_missing_winner_overrides_nothing(self, invoker: Invoker, client: TestClient) -> None:
        _save_image(invoker, "first.png")

        response = client.post(
            "/api/v1/recall/video/default",
            json={"first_frame_image": {"image_name": "first.png"}, "source_video": {"video_name": "gone.mp4"}},
        )

        assert response.json()["skipped"] == ["source_video"]
        assert response.json()["overridden"] == {}
        assert _only_recall_event(invoker).parameters == {"first_frame_image": {"image_name": "first.png"}}

    def test_every_swagger_example_is_a_valid_request(self) -> None:
        from invokeai.app.api.routers.video_recall import VIDEO_RECALL_EXAMPLES, VideoRecallParameter

        for example in VIDEO_RECALL_EXAMPLES.values():
            VideoRecallParameter.model_validate(example["value"])

        # The reference example names every field the request accepts, so it cannot fall behind the model.
        assert set(VIDEO_RECALL_EXAMPLES["every_field"]["value"]) == set(VideoRecallParameter.model_fields)

    def test_an_empty_request_emits_nothing(self, invoker: Invoker, client: TestClient) -> None:
        response = client.post("/api/v1/recall/video/default", json={})

        assert response.status_code == 200
        assert response.json()["status"] == "no_parameters_provided"
        assert _recall_events(invoker) == []

    def test_a_request_whose_every_field_was_dropped_emits_nothing(self, invoker: Invoker, client: TestClient) -> None:
        response = client.post("/api/v1/recall/video/default", json={"model": "Not Installed"})

        assert response.json() == {
            "status": "nothing_resolved",
            "queue_id": "default",
            "parameters": {},
            "skipped": ["model"],
            "overridden": {},
        }
        assert _recall_events(invoker) == []

    @pytest.mark.parametrize(
        "body",
        [
            {"cfg_rescale_multiplier": 0.7},
            {"minimax_h3_references": [{"kind": "audio", "video_name": "x.mp4"}]},
            {"minimax_h3_references": [{"kind": "image", "video_name": "x.mp4"}]},
            {"minimax_h3_references": [{"kind": "image", "image_name": f"{i}.png"} for i in range(10)]},
            {"minimax_h3_references": [{"kind": "video", "video_name": f"{i}.mp4"} for i in range(4)]},
            {"minimax_h3_references": [{"kind": "video", "video_name": "x.mp4", "start_frame": 0}]},
            {"source_video": {"video_name": "x.mp4"}, "source_video_start_frame": 9, "source_video_end_frame": 3},
            {"ltx2_conditioning_video": {"video_name": "x.mp4"}},
            {"model": ""},
            {"loras": [{"model_name": "detail"}] * 33},
            {"seed": 2**32},
            {"cfg_scale": 0.5},
            {"fps": 240},
        ],
        ids=[
            "unknown-field",
            "unknown-reference-kind",
            "reference-kind-mismatch",
            "too-many-reference-images",
            "too-many-reference-videos",
            "half-a-reference-trim",
            "inverted-source-trim",
            "conditioning-video-without-role",
            "empty-model-name",
            "too-many-loras",
            "seed-too-large",
            "cfg-below-the-panel-floor",
            "fps-above-the-panel-ceiling",
        ],
    )
    def test_malformed_requests_are_rejected(self, invoker: Invoker, client: TestClient, body: dict[str, Any]) -> None:
        response = client.post("/api/v1/recall/video/default", json=body)

        assert response.status_code == 422
        assert _recall_events(invoker) == []

    def test_blocked_during_image_move_maintenance(self, invoker: Invoker, client: TestClient) -> None:
        invoker.services.image_moves.is_maintenance_active.return_value = True  # type: ignore[attr-defined]

        response = client.post("/api/v1/recall/video/default", json={"positive_prompt": "p"})

        assert response.status_code == 409
        assert _recall_events(invoker) == []


class TestGalleryVideoPlacement:
    @pytest.mark.parametrize(
        ("path", "action"),
        [("initial-video", "initial_video"), ("reference-video", "reference_video")],
    )
    def test_a_gallery_video_is_delivered_with_the_facts_the_panel_needs(
        self, invoker: Invoker, client: TestClient, path: str, action: str
    ) -> None:
        _save_video(invoker, "clip.mp4")

        response = client.post(f"/api/v1/recall/video/queue-b/{path}", params={"video_name": "clip.mp4"})

        assert response.status_code == 200
        body = response.json()
        assert body["queue_id"] == "queue-b"
        assert body["action"] == action
        assert body["uploaded"] is False
        assert body["video"]["video_name"] == "clip.mp4"
        event = _only_recall_event(invoker)
        assert event.action == action
        assert event.queue_id == "queue-b"
        assert event.parameters is None
        assert event.video is not None
        assert event.video.model_dump() == {
            "video_name": "clip.mp4",
            "width": 832,
            "height": 480,
            "duration": 5.0,
            "fps": 16.0,
            "media_origin": None,
        }

    def test_the_media_origin_travels_with_the_video(self, invoker: Invoker, client: TestClient) -> None:
        """`audio_upload` tells the panel to default the reference to audio-only conditioning."""
        _save_video(invoker, "song.mp4")
        invoker.services.videos.get_dto.side_effect = lambda name: _video_dto(name, media_origin="audio_upload")  # type: ignore[attr-defined]

        client.post("/api/v1/recall/video/default/reference-video", params={"video_name": "song.mp4"})

        event = _only_recall_event(invoker)
        assert event.video is not None and event.video.media_origin == "audio_upload"

    def test_an_unknown_video_is_a_404(self, invoker: Invoker, client: TestClient) -> None:
        response = client.post("/api/v1/recall/video/default/initial-video", params={"video_name": "nope.mp4"})

        assert response.status_code == 404
        assert _recall_events(invoker) == []


def _create_board(invoker: Invoker, owner_id: str, visibility: BoardVisibility) -> str:
    board = invoker.services.board_records.save("Board", owner_id)
    invoker.services.board_records.update(board.board_id, BoardChanges(board_visibility=visibility))
    return board.board_id


@pytest.fixture
def upload_ready(monkeypatch: Any, invoker: Invoker) -> MagicMock:
    """Stub the ffmpeg-backed probes; returns the `videos.create` mock."""
    monkeypatch.setattr(
        videos, "probe_media_streams", lambda path: SimpleNamespace(video_codec="h264", audio_codec="aac")
    )
    monkeypatch.setattr(videos, "_probe_decodable_video", lambda path: ((832, 480, 5.0, 16.0), None))
    create = invoker.services.videos.create  # type: ignore[attr-defined]
    create.return_value = _video_dto("uploaded.mp4")
    return create


class TestUploadedVideoPlacement:
    def test_an_upload_lands_in_the_gallery_on_the_board_and_is_then_placed(
        self, invoker: Invoker, client: TestClient, upload_ready: MagicMock
    ) -> None:
        board_id = _create_board(invoker, "system", BoardVisibility.Private)
        upload_ready.return_value = _video_dto("uploaded.mp4", board_id=board_id)

        response = client.post(
            "/api/v1/recall/video/default/reference-video/upload",
            params={"board_id": board_id},
            files={"file": ("clip.mp4", MP4_BYTES, "video/mp4")},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["uploaded"] is True
        assert body["video"]["video_name"] == "uploaded.mp4"
        created = upload_ready.call_args.kwargs
        assert created["board_id"] == board_id
        assert created["video_category"] == ImageCategory.GENERAL
        assert created["is_intermediate"] is False
        events = invoker.services.events.events  # type: ignore[attr-defined]
        uploaded = [event for event in events if isinstance(event, VideoUploadedEvent)]
        assert [event.video_name for event in uploaded] == ["uploaded.mp4"]
        recall = _only_recall_event(invoker)
        assert recall.action == "reference_video"
        assert recall.video is not None and recall.video.video_name == "uploaded.mp4"
        # The gallery hears about the video before the panel is asked to use it.
        assert events.index(uploaded[0]) < events.index(recall)

    def test_an_upload_without_a_board_goes_to_uncategorized(
        self, invoker: Invoker, client: TestClient, upload_ready: MagicMock
    ) -> None:
        response = client.post(
            "/api/v1/recall/video/default/initial-video/upload",
            files={"file": ("clip.mp4", MP4_BYTES, "video/mp4")},
        )

        assert response.status_code == 200
        assert upload_ready.call_args.kwargs["board_id"] is None
        assert _only_recall_event(invoker).action == "initial_video"

    def test_a_rejected_upload_places_nothing(
        self, invoker: Invoker, client: TestClient, upload_ready: MagicMock
    ) -> None:
        response = client.post(
            "/api/v1/recall/video/default/initial-video/upload",
            files={"file": ("notes.txt", b"hello", "text/plain")},
        )

        assert response.status_code == 415
        upload_ready.assert_not_called()
        assert _recall_events(invoker) == []


# --- Multi-user ---------------------------------------------------------------------------------


@pytest.fixture
def multiuser(setup_jwt_secret: None, invoker: Invoker, client: TestClient) -> dict[str, Any]:
    """Enable multi-user mode; returns each user's id and bearer headers."""
    invoker.services.configuration.multiuser = True
    users: dict[str, Any] = {}
    for email, is_admin in (("admin@test.com", True), ("owner@test.com", False), ("other@test.com", False)):
        user = invoker.services.users.create(
            UserCreateRequest(email=email, display_name=email, password="TestPass123", is_admin=is_admin)
        )
        login = client.post(
            "/api/v1/auth/login", json={"email": email, "password": "TestPass123", "remember_me": False}
        )
        assert login.status_code == 200
        users[email.split("@")[0]] = {
            "id": user.user_id,
            "headers": {"Authorization": f"Bearer {login.json()['token']}"},
        }
    return users


class TestMultiUserAccess:
    @pytest.mark.parametrize(
        ("method_path", "kwargs"),
        [
            ("/api/v1/recall/video/default", {"json": {"positive_prompt": "p"}}),
            ("/api/v1/recall/video/default/initial-video", {"params": {"video_name": "clip.mp4"}}),
            ("/api/v1/recall/video/default/reference-video", {"params": {"video_name": "clip.mp4"}}),
            ("/api/v1/recall/video/default/initial-video/upload", {"files": {"file": ("c.mp4", MP4_BYTES)}}),
            ("/api/v1/recall/video/default/reference-video/upload", {"files": {"file": ("c.mp4", MP4_BYTES)}}),
        ],
        ids=["parameters", "initial-video", "reference-video", "initial-upload", "reference-upload"],
    )
    def test_every_route_requires_authentication(
        self, invoker: Invoker, client: TestClient, multiuser: dict[str, Any], method_path: str, kwargs: dict
    ) -> None:
        response = client.post(method_path, **kwargs)

        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        assert _recall_events(invoker) == []

    def test_another_users_private_video_cannot_be_placed(
        self, invoker: Invoker, client: TestClient, multiuser: dict[str, Any]
    ) -> None:
        _save_video(invoker, "private.mp4", multiuser["owner"]["id"])

        response = client.post(
            "/api/v1/recall/video/default/reference-video",
            params={"video_name": "private.mp4"},
            headers=multiuser["other"]["headers"],
        )

        assert response.status_code == status.HTTP_403_FORBIDDEN
        assert _recall_events(invoker) == []

    @pytest.mark.parametrize(
        "body",
        [
            {"source_video": {"video_name": "private.mp4"}},
            {"first_frame_image": {"image_name": "private.png"}},
            {"last_frame_image": {"image_name": "private.png"}},
            {"ltx2_conditioning_video": {"video_name": "private.mp4"}, "ltx2_conditioning_role": "audio"},
            {"minimax_h3_references": [{"kind": "video", "video_name": "private.mp4"}]},
            {"minimax_h3_references": [{"kind": "image", "image_name": "private.png"}]},
        ],
        ids=["source-video", "first-frame", "last-frame", "conditioning-video", "reference-video", "reference-image"],
    )
    def test_another_users_private_media_cannot_be_recalled(
        self, invoker: Invoker, client: TestClient, multiuser: dict[str, Any], body: dict[str, Any]
    ) -> None:
        _save_video(invoker, "private.mp4", multiuser["owner"]["id"])
        _save_image(invoker, "private.png", multiuser["owner"]["id"])

        response = client.post("/api/v1/recall/video/default", json=body, headers=multiuser["other"]["headers"])

        assert response.status_code == status.HTTP_403_FORBIDDEN
        assert _recall_events(invoker) == []

    def test_a_video_on_a_shared_board_can_be_placed_by_another_user(
        self, invoker: Invoker, client: TestClient, multiuser: dict[str, Any]
    ) -> None:
        _save_video(invoker, "shared.mp4", multiuser["owner"]["id"])
        board_id = _create_board(invoker, multiuser["owner"]["id"], BoardVisibility.Shared)
        invoker.services.board_video_records.add_video_to_board(board_id, "shared.mp4")

        response = client.post(
            "/api/v1/recall/video/default/initial-video",
            params={"video_name": "shared.mp4"},
            headers=multiuser["other"]["headers"],
        )

        assert response.status_code == 200
        assert _only_recall_event(invoker).user_id == multiuser["other"]["id"]

    def test_an_upload_to_another_users_board_is_refused_before_ingest(
        self, invoker: Invoker, client: TestClient, multiuser: dict[str, Any], upload_ready: MagicMock
    ) -> None:
        # Shared boards are visible to everyone but accept uploads only from their owner.
        board_id = _create_board(invoker, multiuser["owner"]["id"], BoardVisibility.Shared)

        response = client.post(
            "/api/v1/recall/video/default/initial-video/upload",
            params={"board_id": board_id},
            files={"file": ("clip.mp4", MP4_BYTES, "video/mp4")},
            headers=multiuser["other"]["headers"],
        )

        assert response.status_code == status.HTTP_403_FORBIDDEN
        upload_ready.assert_not_called()
        assert _recall_events(invoker) == []

    def test_the_event_is_addressed_to_the_caller(
        self, invoker: Invoker, client: TestClient, multiuser: dict[str, Any]
    ) -> None:
        client.post(
            "/api/v1/recall/video/default", json={"positive_prompt": "p"}, headers=multiuser["other"]["headers"]
        )

        assert _only_recall_event(invoker).user_id == multiuser["other"]["id"]
