"""Router-level tests for /api/v1/utilities.

Covers:
- Auth gating (CurrentUserOrDefault on all three utility routes).
- image-to-prompt: image read-access check must fire BEFORE the model is loaded,
  so non-owners can't probe stored images.
- image-to-prompt: a missing image surfaces as 404, not 500.
- image-to-prompt: LLaVA OneVision runs as before; a Text LLM that reads images (by role too) runs the Text LLM
  pipeline on the image and the instruction; any other model is refused before anything loads.
- expand-prompt with an image: the same read-access check, and a text-only model is refused before loading.
"""

import shutil
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from fastapi import status
from fastapi.testclient import TestClient
from fontTools.ttLib import TTFont
from PIL import Image

from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.invoker import Invoker
from invokeai.backend.util.host_platform import is_windows_arm64


def _save_image(mock_invoker: Invoker, image_name: str, user_id: str) -> None:
    mock_invoker.services.image_records.save(
        image_name=image_name,
        image_origin=ResourceOrigin.INTERNAL,
        image_category=ImageCategory.GENERAL,
        width=100,
        height=100,
        has_workflow=False,
        user_id=user_id,
    )


def _create_extra_user(mock_invoker: Invoker, email: str) -> str:
    from invokeai.app.services.users.users_common import UserCreateRequest

    user = mock_invoker.services.users.create(
        UserCreateRequest(email=email, display_name=email, password="TestPass123", is_admin=False)
    )
    return user.user_id


@pytest.fixture
def font_root(mock_invoker: Invoker, invokeai_root_dir: Path) -> Path:
    mock_invoker.services.configuration._root = invokeai_root_dir
    return invokeai_root_dir


# ----------------------------- Auth gating -----------------------------


@pytest.mark.parametrize(
    "path,body",
    [
        ("/api/v1/utilities/dynamicprompts", {"prompt": "hi"}),
        ("/api/v1/utilities/expand-prompt", {"prompt": "hi", "model_key": "m"}),
        ("/api/v1/utilities/image-to-prompt", {"image_name": "img-1", "model_key": "m"}),
    ],
)
def test_routes_require_auth(enable_multiuser: Any, client: TestClient, mock_invoker: Invoker, path: str, body: dict):
    r = client.post(path, json=body)
    assert r.status_code == status.HTTP_401_UNAUTHORIZED
    mock_invoker.services.model_manager.store.get_model.assert_not_called()


def test_dynamicprompts_works_for_user(client: TestClient, user1_token: str):
    r = client.post(
        "/api/v1/utilities/dynamicprompts",
        json={"prompt": "a {b|c}"},
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert r.status_code == status.HTTP_200_OK
    body = r.json()
    assert "prompts" in body


def test_dynamicprompts_unknown_wildcard_returns_error_without_hanging(client: TestClient, user1_token: str):
    """An unknown wildcard used as a variant value would otherwise loop forever in the combinatorial generator.

    The endpoint must instead return promptly with a clear error and the original prompt echoed back.
    """
    r = client.post(
        "/api/v1/utilities/dynamicprompts",
        json={"prompt": "{__random__8chan|fenster|stuff}"},
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert r.status_code == status.HTTP_200_OK
    body = r.json()
    assert body["error"] is not None
    assert "random" in body["error"]
    assert body["prompts"] == ["{__random__8chan|fenster|stuff}"]


def test_dynamicprompts_bare_unknown_wildcard_returns_error(client: TestClient, user1_token: str):
    """A wildcard outside a variant does not hang, but it does not generate usefully either.

    The combinatorial generator emits `max_prompts` copies of one prompt and drops every other
    variant's values, so a caller that generated anyway would queue N identical results. This case
    was previously allowed through with `error: None`; it must report the unresolvable name instead.
    """
    r = client.post(
        "/api/v1/utilities/dynamicprompts",
        json={"prompt": "a {red|green} photo, __my_style__"},
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert r.status_code == status.HTTP_200_OK
    body = r.json()
    assert body["error"] is not None
    assert "my_style" in body["error"]
    assert body["prompts"] == ["a {red|green} photo, __my_style__"]


def test_dynamicprompts_random_generator_ignores_unknown_wildcard(client: TestClient, user1_token: str):
    """The random generator handles unknown wildcards gracefully, so the guard must not fire for it."""
    r = client.post(
        "/api/v1/utilities/dynamicprompts",
        json={"prompt": "{__random__8chan|fenster|stuff}", "combinatorial": False, "seed": 0},
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert r.status_code == status.HTTP_200_OK
    body = r.json()
    assert body["error"] is None
    assert body["prompts"]  # non-empty; the random generator expanded the variant


# ----------------------------- image_to_prompt: ownership / read-access -----------------------------


def test_image_to_prompt_forbidden_for_non_owner(
    client: TestClient, user1_token: str, user2_token: str, mock_invoker: Invoker
):
    """A second user must not be able to read a private image via image-to-prompt."""
    # Need to discover user1's id, then save an image under that id.
    user1 = mock_invoker.services.users.get_by_email("user1@test.com")
    assert user1 is not None
    _save_image(mock_invoker, "private-img.png", user1.user_id)

    r = client.post(
        "/api/v1/utilities/image-to-prompt",
        json={"image_name": "private-img.png", "model_key": "some-key"},
        headers={"Authorization": f"Bearer {user2_token}"},
    )
    assert r.status_code == status.HTTP_403_FORBIDDEN
    # The model must not have been loaded — ownership must fire BEFORE inference.
    mock_invoker.services.model_manager.store.get_model.assert_not_called()


def test_image_to_prompt_owner_reaches_model_load(client: TestClient, user1_token: str, mock_invoker: Invoker):
    """The owner passes the read-access check and the model load is attempted.
    We force an UnknownModelException to keep the test light and assert 404."""
    from invokeai.app.services.model_records.model_records_base import UnknownModelException

    user1 = mock_invoker.services.users.get_by_email("user1@test.com")
    assert user1 is not None
    _save_image(mock_invoker, "owned-img.png", user1.user_id)

    mock_invoker.services.model_manager.store.get_model = MagicMock(side_effect=UnknownModelException("no such model"))

    r = client.post(
        "/api/v1/utilities/image-to-prompt",
        json={"image_name": "owned-img.png", "model_key": "missing-model"},
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert r.status_code == status.HTTP_404_NOT_FOUND
    mock_invoker.services.model_manager.store.get_model.assert_called_once()


def test_image_to_prompt_admin_can_access_any_image(
    client: TestClient, admin_token: str, user1_token: str, mock_invoker: Invoker
):
    from invokeai.app.services.model_records.model_records_base import UnknownModelException

    user1 = mock_invoker.services.users.get_by_email("user1@test.com")
    assert user1 is not None
    _save_image(mock_invoker, "user1-img.png", user1.user_id)

    mock_invoker.services.model_manager.store.get_model = MagicMock(side_effect=UnknownModelException("no model"))

    r = client.post(
        "/api/v1/utilities/image-to-prompt",
        json={"image_name": "user1-img.png", "model_key": "x"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    # Admin passes the read-access check; model loading then fails with 404.
    assert r.status_code == status.HTTP_404_NOT_FOUND


def test_list_user_fonts_requires_auth(enable_multiuser: Any, font_root: Path, client: TestClient) -> None:
    fonts_dir = font_root / "fonts"
    fonts_dir.mkdir(parents=True, exist_ok=True)
    (fonts_dir / "MyFont.ttf").write_bytes(b"not-a-real-font")

    r = client.get("/api/v1/utilities/fonts")

    assert r.status_code == status.HTTP_401_UNAUTHORIZED


def test_get_user_font_file_requires_auth(enable_multiuser: Any, font_root: Path, client: TestClient) -> None:
    fonts_dir = font_root / "fonts"
    fonts_dir.mkdir(parents=True, exist_ok=True)
    (fonts_dir / "MyFont.ttf").write_bytes(b"not-a-real-font")

    r = client.get("/api/v1/utilities/fonts/MyFont.ttf")

    assert r.status_code == status.HTTP_401_UNAUTHORIZED


def test_user_fonts_support_real_font_files_and_configured_directory(
    admin_token: str, client: TestClient, font_root: Path, mock_invoker: Invoker
) -> None:
    assert mock_invoker.services.configuration.root_path == font_root
    mock_invoker.services.configuration.fonts_dir = Path("custom-fonts")
    fonts_dir = mock_invoker.services.configuration.fonts_path
    fonts_dir.mkdir(parents=True, exist_ok=True)
    source_font = Path(__file__).parents[3] / "invokeai" / "assets" / "fonts" / "inter" / "Inter-Regular.ttf"
    shutil.copyfile(source_font, fonts_dir / "Inter-Regular.ttf")

    r = client.get("/api/v1/utilities/fonts", headers={"Authorization": f"Bearer {admin_token}"})

    assert r.status_code == status.HTTP_200_OK
    body = r.json()
    assert len(body["fonts"]) == 1
    assert body["fonts"][0]["family"] == "Inter"
    assert body["fonts"][0]["url"] == "api/v1/utilities/fonts/Inter-Regular.ttf"

    font_response = client.get(
        "/api/v1/utilities/fonts/Inter-Regular.ttf",
        headers={"Authorization": f"Bearer {admin_token}"},
    )

    assert font_response.status_code == status.HTTP_200_OK
    assert font_response.headers["content-type"] == "font/ttf"
    assert font_response.headers["cache-control"] == "private, no-cache"
    assert font_response.headers["content-disposition"].startswith('inline; filename="Inter-Regular.ttf"')
    assert font_response.content == source_font.read_bytes()


def test_list_user_fonts_reads_real_woff2_file(
    admin_token: str, client: TestClient, mock_invoker: Invoker, tmp_path: Path
) -> None:
    mock_invoker.services.configuration.fonts_dir = tmp_path / "fonts"
    fonts_dir = mock_invoker.services.configuration.fonts_path
    fonts_dir.mkdir(parents=True, exist_ok=True)
    source_font = Path(__file__).parents[3] / "invokeai" / "assets" / "fonts" / "inter" / "Inter-Regular.ttf"
    if is_windows_arm64():
        # fontTools encodes woff2 through brotli, which has no Windows ARM64 wheel (the `woff` extra is dropped there).
        pytest.importorskip("brotli", reason="woff2 encoding needs brotli, which has no Windows ARM64 wheel")
    font = TTFont(source_font)
    try:
        font.flavor = "woff2"
        font.save(fonts_dir / "Inter-Regular.woff2")
    finally:
        font.close()

    r = client.get("/api/v1/utilities/fonts", headers={"Authorization": f"Bearer {admin_token}"})

    assert r.status_code == status.HTTP_200_OK
    body = r.json()
    assert len(body["fonts"]) == 1
    assert body["fonts"][0]["family"] == "Inter"
    assert body["fonts"][0]["faces"][0]["path"] == "Inter-Regular.woff2"


def test_list_user_fonts_allows_authenticated_access(
    admin_token: str, client: TestClient, font_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fonts_dir = font_root / "fonts"
    fonts_dir.mkdir(parents=True, exist_ok=True)
    (fonts_dir / "MyFont.ttf").write_bytes(b"not-a-real-font")
    monkeypatch.setattr(
        "invokeai.app.api.routers.utilities._get_font_metadata",
        lambda _font_file: ("My Font", "My Font", 400, "normal"),
    )

    r = client.get("/api/v1/utilities/fonts", headers={"Authorization": f"Bearer {admin_token}"})

    assert r.status_code == status.HTTP_200_OK
    body = r.json()
    assert len(body["fonts"]) == 1
    assert body["fonts"][0]["id"] == "user:my font"


def test_list_user_fonts_id_is_stable_when_preferred_face_changes(
    admin_token: str,
    client: TestClient,
    mock_invoker: Invoker,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mock_invoker.services.configuration.fonts_dir = tmp_path / "fonts"
    fonts_dir = mock_invoker.services.configuration.fonts_path
    fonts_dir.mkdir(parents=True, exist_ok=True)
    (fonts_dir / "MyFont-Bold.ttf").write_bytes(b"not-a-real-font")

    def get_metadata(font_file: Path) -> tuple[str, str, int, str]:
        weight = 400 if "Regular" in font_file.stem else 700
        return ("My Font", "My Font", weight, "normal")

    monkeypatch.setattr("invokeai.app.api.routers.utilities._get_font_metadata", get_metadata)

    first_response = client.get("/api/v1/utilities/fonts", headers={"Authorization": f"Bearer {admin_token}"})
    first_font = first_response.json()["fonts"][0]
    assert first_font["id"] == "user:my font"
    assert first_font["path"] == "MyFont-Bold.ttf"

    (fonts_dir / "MyFont-Regular.ttf").write_bytes(b"not-a-real-font")
    second_response = client.get("/api/v1/utilities/fonts", headers={"Authorization": f"Bearer {admin_token}"})
    second_font = second_response.json()["fonts"][0]
    assert second_font["id"] == first_font["id"]
    assert second_font["path"] == "MyFont-Regular.ttf"


def test_list_user_fonts_skips_malformed_fonts_and_logs_warning(
    admin_token: str,
    client: TestClient,
    font_root: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    fonts_dir = font_root / "fonts"
    fonts_dir.mkdir(parents=True, exist_ok=True)
    (fonts_dir / "BrokenFont.ttf").write_bytes(b"not-a-real-font")

    with caplog.at_level("WARNING"):
        r = client.get("/api/v1/utilities/fonts", headers={"Authorization": f"Bearer {admin_token}"})

    assert r.status_code == status.HTTP_200_OK
    assert r.json()["fonts"] == []
    assert "Skipping font file" in caplog.text


def test_get_user_font_file_rejects_symlink(
    admin_token: str, client: TestClient, font_root: Path, tmp_path: Path
) -> None:
    fonts_dir = font_root / "fonts"
    fonts_dir.mkdir(parents=True, exist_ok=True)
    outside_file = tmp_path / "outside.ttf"
    outside_file.write_bytes(b"outside-font")
    symlink_path = fonts_dir / "linked.ttf"

    try:
        symlink_path.symlink_to(outside_file)
    except (NotImplementedError, OSError):
        pytest.skip("Symlinks are not available in this test environment")

    r = client.get("/api/v1/utilities/fonts/linked.ttf", headers={"Authorization": f"Bearer {admin_token}"})

    assert r.status_code == status.HTTP_400_BAD_REQUEST


def test_list_user_fonts_skips_symlinked_files(
    admin_token: str, client: TestClient, font_root: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    fonts_dir = font_root / "fonts"
    fonts_dir.mkdir(parents=True, exist_ok=True)
    outside_dir = tmp_path / "outside-fonts"
    outside_dir.mkdir()
    (outside_dir / "outside.ttf").write_bytes(b"outside-font")
    symlink_path = fonts_dir / "linked-dir"

    try:
        symlink_path.symlink_to(outside_dir, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("Symlinks are not available in this test environment")

    with caplog.at_level("WARNING"):
        r = client.get("/api/v1/utilities/fonts", headers={"Authorization": f"Bearer {admin_token}"})

    assert r.status_code == status.HTTP_200_OK
    assert r.json()["fonts"] == []
    assert "Skipping font path" in caplog.text


# ----------------------------- expand_prompt with an image -----------------------------


def test_expand_prompt_image_forbidden_for_non_owner(
    client: TestClient, user1_token: str, user2_token: str, mock_invoker: Invoker
):
    """Attaching an image must not let a second user read a private image through the rewrite."""
    user1 = mock_invoker.services.users.get_by_email("user1@test.com")
    assert user1 is not None
    _save_image(mock_invoker, "private-frame.png", user1.user_id)

    r = client.post(
        "/api/v1/utilities/expand-prompt",
        json={"prompt": "she waves", "model_key": "some-key", "image_name": "private-frame.png"},
        headers={"Authorization": f"Bearer {user2_token}"},
    )
    assert r.status_code == status.HTTP_403_FORBIDDEN
    mock_invoker.services.model_manager.store.get_model.assert_not_called()


def _text_only_llm() -> Any:
    from invokeai.backend.model_manager.taxonomy import ModelType

    return MagicMock(type=ModelType.TextLLM, supports_images=False)


def _qwen3_vl_encoder_serving_as_text_llm() -> Any:
    from tests.fixtures.model_roles import qwen3_vl_encoder_file

    return qwen3_vl_encoder_file("vl")


@pytest.mark.parametrize("record", [_text_only_llm, _qwen3_vl_encoder_serving_as_text_llm])
def test_expand_prompt_image_with_text_only_model_is_rejected_before_loading(
    client: TestClient, user1_token: str, mock_invoker: Invoker, record: Callable[[], Any]
):
    user1 = mock_invoker.services.users.get_by_email("user1@test.com")
    assert user1 is not None
    _save_image(mock_invoker, "frame.png", user1.user_id)
    mock_invoker.services.model_manager.store.get_model = MagicMock(return_value=record())
    mock_invoker.services.model_manager.load.load_model = MagicMock()

    r = client.post(
        "/api/v1/utilities/expand-prompt",
        json={"prompt": "she waves", "model_key": "qwen", "image_name": "frame.png"},
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert r.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
    assert "cannot read images" in r.json()["detail"]
    mock_invoker.services.model_manager.load.load_model.assert_not_called()


def test_expand_prompt_missing_image_is_404_for_an_admin(client: TestClient, admin_token: str, mock_invoker: Invoker):
    """Admins skip the ownership lookup, so a missing record surfaces on the read and must still be a 404."""
    from invokeai.backend.model_manager.taxonomy import ModelType

    mock_invoker.services.model_manager.store.get_model = MagicMock(
        return_value=MagicMock(type=ModelType.TextLLM, supports_images=True)
    )
    mock_invoker.services.model_manager.load.load_model = MagicMock()

    r = client.post(
        "/api/v1/utilities/expand-prompt",
        json={"prompt": "she waves", "model_key": "e2b", "image_name": "deleted-frame.png"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert r.status_code == status.HTTP_404_NOT_FOUND
    mock_invoker.services.model_manager.load.load_model.assert_not_called()


# ----------------------------- image_to_prompt: which models run, and how -----------------------------


class _Generation:
    """Stands in for the model loads and the pipelines: records what reaches them, answers with a canned prompt."""

    ANSWER = "a red fox in fresh snow"

    def __init__(self, monkeypatch: pytest.MonkeyPatch, mock_invoker: Invoker) -> None:
        from invokeai.app.api.routers import utilities

        self.loads: list[tuple[Any, Any]] = []
        self.text_llm_runs: list[dict[str, Any]] = []
        self.llava_runs: list[dict[str, Any]] = []
        self.with_processor: list[bool] = []
        self.image = Image.new("RGB", (32, 24), "red")
        generation = self

        class TextLLMPipeline:
            def __init__(self, model: Any, tokenizer: Any, processor: Any) -> None:
                self.processor = processor

            def run(self, **kwargs: Any) -> str:
                generation.text_llm_runs.append({**kwargs, "processor": self.processor})
                if kwargs["progress_callback"] is not None:
                    kwargs["progress_callback"](1, kwargs["max_new_tokens"])
                return generation.ANSWER

        class LlavaOnevisionPipeline:
            def __init__(self, model: Any, processor: Any) -> None:
                pass

            def run(self, **kwargs: Any) -> str:
                generation.llava_runs.append(kwargs)
                return generation.ANSWER

        def tokenizer_and_processor(load: Callable[[Any], Any], *, with_processor: bool) -> tuple[Any, Any]:
            from invokeai.backend.model_manager.taxonomy import SubModelType

            self.with_processor.append(with_processor)
            load(SubModelType.Processor if with_processor else SubModelType.Tokenizer)
            return "tokenizer", "processor" if with_processor else None

        monkeypatch.setattr(utilities, "TextLLMPipeline", TextLLMPipeline)
        monkeypatch.setattr(utilities, "LlavaOnevisionPipeline", LlavaOnevisionPipeline)
        monkeypatch.setattr(utilities, "load_text_llm_tokenizer_and_processor", tokenizer_and_processor)
        monkeypatch.setattr(mock_invoker.services.images, "get_pil_image", lambda _name: self.image)
        mock_invoker.services.model_manager.load.load_model = MagicMock(side_effect=self._load)

    def _load(self, config: Any, submodel: Any = None, user_id: str | None = None) -> Any:
        from transformers import LlavaOnevisionForConditionalGeneration

        self.loads.append((config, submodel))
        model = torch.nn.Linear(1, 1)
        if getattr(config, "type", None) == "llava_onevision":
            model = MagicMock(spec=LlavaOnevisionForConditionalGeneration)
            model.parameters.side_effect = lambda: iter([torch.zeros(1)])

        @contextmanager
        def on_device() -> Iterator[tuple[None, Any]]:
            yield None, model

        return MagicMock(model_on_device=on_device)


def _image_to_prompt(client: TestClient, token: str, mock_invoker: Invoker, record: Any) -> Any:
    user1 = mock_invoker.services.users.get_by_email("user1@test.com")
    assert user1 is not None
    _save_image(mock_invoker, "photo.png", user1.user_id)
    mock_invoker.services.model_manager.store.get_model = MagicMock(return_value=record)
    return client.post(
        "/api/v1/utilities/image-to-prompt",
        json={"image_name": "photo.png", "model_key": "vision", "instruction": "Describe it.", "task_id": "t1"},
        headers={"Authorization": f"Bearer {token}"},
    )


def test_image_to_prompt_runs_a_text_llm_that_reads_images_by_role_on_the_image_and_the_instruction(
    client: TestClient, user1_token: str, mock_invoker: Invoker, monkeypatch: pytest.MonkeyPatch
):
    from invokeai.app.api.routers.utilities import IMAGE_TO_PROMPT_SYSTEM_PROMPT
    from invokeai.app.services.events.events_common import LLMTaskCompleteEvent, LLMTaskProgressEvent
    from invokeai.backend.model_manager.taxonomy import ModelType, SubModelType
    from tests.fixtures.model_roles import qwen3_vl_encoder_file

    generation = _Generation(monkeypatch, mock_invoker)

    r = _image_to_prompt(client, user1_token, mock_invoker, qwen3_vl_encoder_file("vl", reads_images=True))

    assert r.status_code == status.HTTP_200_OK, r.text
    assert r.json()["prompt"] == _Generation.ANSWER
    # The encoder record loads in its Text LLM role, the processor first, as for Expand Prompt.
    assert [(config.type, submodel) for config, submodel in generation.loads] == [
        (ModelType.TextLLM, SubModelType.Processor),
        (ModelType.TextLLM, None),
    ]
    assert generation.with_processor == [True]
    (run,) = generation.text_llm_runs
    assert (run["image"], run["prompt"], run["system_prompt"]) == (
        generation.image,
        "Describe it.",
        IMAGE_TO_PROMPT_SYSTEM_PROMPT,
    )
    assert run["processor"] == "processor"
    assert generation.llava_runs == []
    events = mock_invoker.services.events.events
    # The same progress contract as LLaVA's: loading, then generating against the request's cap, then complete.
    assert [(event.phase, event.total_tokens) for event in events if isinstance(event, LLMTaskProgressEvent)] == [
        ("loading_model", None),
        ("generating", 400),
    ]
    assert [event.task_id for event in events if isinstance(event, LLMTaskCompleteEvent)] == ["t1"]


def test_image_to_prompt_still_runs_llava_onevision_on_its_own_pipeline(
    client: TestClient, user1_token: str, mock_invoker: Invoker, monkeypatch: pytest.MonkeyPatch
):
    from transformers import LlavaOnevisionProcessor

    from invokeai.app.api.routers import utilities
    from invokeai.backend.model_manager.taxonomy import ModelType

    generation = _Generation(monkeypatch, mock_invoker)
    monkeypatch.setattr(
        utilities.AutoProcessor, "from_pretrained", lambda *_args, **_kwargs: MagicMock(spec=LlavaOnevisionProcessor)
    )

    r = _image_to_prompt(
        client, user1_token, mock_invoker, MagicMock(type=ModelType.LlavaOnevision, path="/models/llava")
    )

    assert r.status_code == status.HTTP_200_OK, r.text
    assert r.json()["prompt"] == _Generation.ANSWER
    (run,) = generation.llava_runs
    assert (run["prompt"], run["images"]) == ("Describe it.", [generation.image])
    assert generation.text_llm_runs == []


def test_expand_prompt_conditions_on_the_image_with_a_text_llm_that_reads_images_by_role(
    client: TestClient, user1_token: str, mock_invoker: Invoker, monkeypatch: pytest.MonkeyPatch
):
    from invokeai.backend.model_manager.taxonomy import ModelType
    from invokeai.backend.text_llm_pipeline import DEFAULT_SYSTEM_PROMPT
    from tests.fixtures.model_roles import qwen3_vl_encoder_file

    generation = _Generation(monkeypatch, mock_invoker)
    user1 = mock_invoker.services.users.get_by_email("user1@test.com")
    assert user1 is not None
    _save_image(mock_invoker, "frame.png", user1.user_id)
    mock_invoker.services.model_manager.store.get_model = MagicMock(
        return_value=qwen3_vl_encoder_file("vl", reads_images=True)
    )

    r = client.post(
        "/api/v1/utilities/expand-prompt",
        json={"prompt": "she waves", "model_key": "vl", "image_name": "frame.png", "seed": 7},
        headers={"Authorization": f"Bearer {user1_token}"},
    )

    assert r.status_code == status.HTTP_200_OK, r.text
    assert r.json() == {"expanded_prompt": _Generation.ANSWER, "seed": 7, "error": None}
    assert {config.type for config, _ in generation.loads} == {ModelType.TextLLM}
    (run,) = generation.text_llm_runs
    assert (run["image"], run["prompt"], run["system_prompt"], run["seed"]) == (
        generation.image,
        "she waves",
        DEFAULT_SYSTEM_PROMPT,
        7,
    )


def _main_model() -> Any:
    from invokeai.backend.model_manager.taxonomy import ModelType

    return MagicMock(type=ModelType.Main, roles=[])


@pytest.mark.parametrize(
    ("record", "detail"),
    [
        (_text_only_llm, "cannot read images"),
        (_qwen3_vl_encoder_serving_as_text_llm, "Re-identify it to check whether its file carries a vision tower"),
        (_main_model, "neither a LLaVA OneVision model nor a Text LLM"),
    ],
    ids=["text-only Text LLM", "role recorded before it said", "not a vision model"],
)
def test_image_to_prompt_refuses_a_model_that_cannot_read_images_before_loading(
    client: TestClient,
    user1_token: str,
    mock_invoker: Invoker,
    monkeypatch: pytest.MonkeyPatch,
    record: Callable[[], Any],
    detail: str,
):
    generation = _Generation(monkeypatch, mock_invoker)

    r = _image_to_prompt(client, user1_token, mock_invoker, record())

    assert r.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
    assert detail in r.json()["detail"]
    assert generation.loads == []
