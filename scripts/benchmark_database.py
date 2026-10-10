#!/usr/bin/env python
"""Times representative database operations through the services, to compare two versions of the code.

Run it once per version and compare the results, e.g. a branch against its base:

    PYTHONPATH=<checkout of the base>   python scripts/benchmark_database.py --json base.json
    PYTHONPATH=<checkout of the branch> python scripts/benchmark_database.py --json head.json
    python scripts/benchmark_database.py --compare base.json head.json

The database is a SQLite file in a temporary directory, built by the real migrations and seeded through
the services, so what is timed is the production path including its transaction handling. Each operation
runs once to warm up and then repeatedly; the median and p95 time per call are reported, and the number
of SQL statements per call, counted in a separate pass with statement logging on so that the logging
does not distort the timings. Timings vary between runs on a busy machine: compare runs made back to
back, and repeat a comparison before trusting a small difference.

A change that ports a service to the database layer renames the storage this script constructs, so the base of
such a change runs its own copy of the script: `python <checkout of the base>/scripts/benchmark_database.py`.
Operations only the branch times show without a base value.
"""

import argparse
import json
import logging
import random
import statistics
import sys
import tempfile
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest import mock

# A median may grow by this fraction or by this many milliseconds, whichever is larger. The floor is the fixed
# cost of a call through the query layer: SQLAlchemy Core and an explicit transaction make a point read on
# SQLite about 11 microseconds slower than a raw cursor did (measured). Calls that do real work stay bound by
# the fraction.
BUDGET_FRACTION = 0.10
BUDGET_FLOOR_MS = 0.05

ACCOUNTS = 200
CLIENT_STATE_KEYS = 200
PROJECTS = 50
WORKFLOWS = 300
# One account's library: its own style presets beside the bundled and the shared ones, its system prompts, and the
# wildcards its prompts draw from.
STYLE_PRESETS = 50
SYSTEM_PROMPTS = 20
WILDCARDS = 25
MODELS = 500
VIDEOS = 2_000
WORKFLOW_TAGS = ["sdxl", "flux", "upscale", "inpaint", "video", "portrait", "landscape", "controlnet"]


class _StatementCounter(logging.Handler):
    """Counts the statements SQLite reports through the trace callback a verbose database installs."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.count = 0

    def handle(self, record: logging.LogRecord) -> bool:
        self.count += 1
        return True


class Services:
    def __init__(self, db: Any) -> None:
        from invokeai.app.services.board_image_records.board_image_records_default import (
            BoardImageRecordStorage,
        )
        from invokeai.app.services.board_records.board_records_default import BoardRecordStorage
        from invokeai.app.services.board_video_records.board_video_records_default import BoardVideoRecordStorage
        from invokeai.app.services.client_state_persistence.client_state_persistence_default import (
            ClientStatePersistence,
        )
        from invokeai.app.services.gallery.gallery_default import GalleryService
        from invokeai.app.services.image_records.image_records_default import ImageRecordStorage
        from invokeai.app.services.model_records import ModelRecordServiceSQL
        from invokeai.app.services.project_records.project_records_default import ProjectRecordsStorage
        from invokeai.app.services.style_preset_records.style_preset_records_default import StylePresetRecordsStorage
        from invokeai.app.services.system_prompt_records.system_prompt_records_default import (
            SystemPromptRecordsStorage,
        )
        from invokeai.app.services.users.users_default import UserService
        from invokeai.app.services.video_records.video_records_default import VideoRecordStorage
        from invokeai.app.services.wildcard_records.wildcard_records_default import WildcardRecordsStorage
        from invokeai.app.services.workflow_records.workflow_records_default import WorkflowRecordsStorage

        # A `Database`, or the cursor facade of a base from before it was removed.
        self.database = getattr(db, "database", db)
        self.image_records = ImageRecordStorage(self.database)
        self.video_records = VideoRecordStorage(self.database)
        self.board_records = BoardRecordStorage(self.database)
        self.board_image_records = BoardImageRecordStorage(self.database)
        self.board_video_records = BoardVideoRecordStorage(self.database)
        self.gallery = GalleryService(self.database)
        self.users = UserService(self.database)
        self.client_state = ClientStatePersistence(self.database)
        self.project_records = ProjectRecordsStorage(self.database)
        self.workflow_records = WorkflowRecordsStorage(self.database)
        self.style_preset_records = StylePresetRecordsStorage(self.database)
        self.system_prompt_records = SystemPromptRecordsStorage(self.database)
        self.wildcard_records = WildcardRecordsStorage(self.database)
        self.model_records = ModelRecordServiceSQL(self.database, logging.getLogger("benchmark_database.quiet"))
        # Listing gallery items builds their URLs through the invoker's URL service.
        from invokeai.app.services.urls.urls_default import LocalUrlService

        invoker = mock.Mock()
        invoker.services.urls = LocalUrlService()
        invoker.services.logger = logging.getLogger("benchmark_database.quiet")
        self.gallery.start(invoker)
        # Starting the library stores the bundled workflows and style presets.
        self.workflow_records.start(invoker)
        self.style_preset_records.start(invoker)


def _metadata(rng: random.Random) -> str:
    words = ["portrait", "landscape", "cinematic", "volumetric", "light", "detailed", "film", "grain", "neon"]
    prompt = " ".join(rng.choice(words) for _ in range(120))
    return json.dumps(
        {
            "generation_mode": "txt2img",
            "positive_prompt": prompt,
            "negative_prompt": prompt[:400],
            "model": {"key": str(uuid.UUID(int=rng.getrandbits(128))), "name": "Some Model", "base": "flux"},
            "seed": rng.getrandbits(32),
            "steps": 30,
            "cfg_scale": 3.5,
            "width": 1024,
            "height": 1024,
            "scheduler": "euler",
            "loras": [{"key": str(uuid.UUID(int=rng.getrandbits(128))), "weight": 0.75} for _ in range(3)],
        }
    )


def _project_document(rng: random.Random, names: list[str]) -> dict[str, Any]:
    """A canvas of 20 layers, each an image and five brush strokes: about 25 KB of JSON referencing 20 images."""
    layers = [
        {
            "id": f"layer_{i}",
            "type": "raster_layer",
            "isEnabled": True,
            "opacity": 1.0,
            "position": {"x": rng.randrange(-512, 512), "y": rng.randrange(-512, 512)},
            "objects": [
                {"id": f"image_{i}", "type": "image", "image": {"image_name": rng.choice(names), "width": 1024}},
                *(
                    {
                        "id": f"line_{i}_{j}",
                        "type": "brush_line",
                        "strokeWidth": 50,
                        "color": {"r": rng.randrange(256), "g": rng.randrange(256), "b": rng.randrange(256), "a": 1},
                        "points": [rng.randrange(1024) for _ in range(40)],
                    }
                    for j in range(5)
                ),
            ],
        }
        for i in range(20)
    ]
    return {"canvas": {"layers": layers, "bbox": {"x": 0, "y": 0, "width": 1024, "height": 1024}}}


def _workflow_template() -> Any:
    """The bundled workflow of median size, as an account's workflow."""
    from invokeai.app.services.workflow_records import workflow_records_common

    bundled = Path(workflow_records_common.__file__).parent / "default_workflows"
    paths = sorted(bundled.glob("*.json"), key=lambda path: path.stat().st_size)
    workflow = json.loads(paths[len(paths) // 2].read_text(encoding="utf-8"))
    workflow.pop("id", None)
    workflow["meta"]["category"] = "user"
    return workflow_records_common.WorkflowWithoutIDValidator.validate_python(workflow)


def _model_config(i: int) -> Any:
    """A main model, a VAE or an embedding, by turns."""
    from invokeai.backend.model_manager.configs.main import Main_Diffusers_SDXL_Config
    from invokeai.backend.model_manager.configs.textual_inversion import TI_File_SD1_Config
    from invokeai.backend.model_manager.configs.vae import VAE_Diffusers_SD1_Config
    from invokeai.backend.model_manager.taxonomy import (
        BaseModelType,
        ModelRepoVariant,
        ModelSourceType,
        ModelType,
        ModelVariantType,
        SchedulerPredictionType,
    )

    common = {
        "key": str(uuid.UUID(int=i)),
        "path": f"/models/{i}/model",
        "name": f"Model {i}",
        "hash": f"blake3:{i:064x}",
        "file_size": 2_000_000 + i,
        "source": f"https://example.com/models/{i}",
        "source_type": ModelSourceType.Url,
    }
    if i % 3 == 0:
        return Main_Diffusers_SDXL_Config(
            **common,
            base=BaseModelType.StableDiffusionXL,
            type=ModelType.Main,
            variant=ModelVariantType.Normal,
            prediction_type=SchedulerPredictionType.Epsilon,
            repo_variant=ModelRepoVariant.Default,
        )
    if i % 3 == 1:
        return VAE_Diffusers_SD1_Config(
            **common, base=BaseModelType.StableDiffusion1, type=ModelType.VAE, repo_variant=ModelRepoVariant.Default
        )
    return TI_File_SD1_Config(**common, base=BaseModelType.StableDiffusion1, type=ModelType.TextualInversion)


def _seed(services: Services, images: int, boards: int, rng: random.Random) -> list[str]:
    from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
    from invokeai.app.services.style_preset_records.style_preset_records_common import (
        PresetData,
        PresetType,
        StylePresetWithoutId,
    )
    from invokeai.app.services.system_prompt_records.system_prompt_records_common import SystemPromptWithoutId
    from invokeai.app.services.wildcard_records.wildcard_records_common import WildcardWithoutId

    board_ids = [services.board_records.save(board_name=f"Board {i}", user_id="system").board_id for i in range(boards)]
    names: list[str] = []
    for i in range(images):
        name = f"{uuid.UUID(int=rng.getrandbits(128))}.png"
        services.image_records.save(
            image_name=name,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            width=1024,
            height=1024,
            has_workflow=False,
            is_intermediate=i % 20 == 0,
            starred=i % 50 == 0,
            metadata=_metadata(rng),
            user_id="system",
        )
        names.append(name)
        if i % 2 == 0:
            services.board_image_records.add_image_to_board(board_id=board_ids[i % boards], image_name=name)
    for i in range(VIDEOS):
        name = f"{uuid.UUID(int=rng.getrandbits(128))}.mp4"
        services.video_records.save(
            video_name=name,
            video_origin=ResourceOrigin.INTERNAL,
            video_category=ImageCategory.GENERAL,
            width=1280,
            height=720,
            duration=5.0,
            fps=24.0,
            has_workflow=False,
            is_intermediate=i % 20 == 0,
            starred=i % 50 == 0,
            # Every row's listing reads the media origin out of its metadata.
            metadata=json.dumps({"media_origin": "audio_upload"}) if i % 10 == 0 else _metadata(rng),
            user_id="system",
        )
        if i % 2 == 0:
            services.board_video_records.add_video_to_board(board_id=board_ids[i % boards], video_name=name)
    # Accounts are inserted directly: the service would hash a password for each, which takes most of a second.
    with services.database.queries.transaction() as q:
        for i in range(ACCOUNTS):
            q.users.insert(
                user_id=f"user-{i}",
                email=f"user{i}@example.com",
                display_name=f"User {i}",
                password_hash="-",
                is_admin=False,
            )
    for i in range(CLIENT_STATE_KEYS):
        services.client_state.set_by_key(
            "system", f"canvas_snapshot:{i}", json.dumps({"imageName": names[i % len(names)]})
        )
    # Each project claims one of the oldest boards, so the boards stay as many as asked for and the newest one,
    # whose board operations are timed, belongs to no project.
    for i in range(min(PROJECTS, boards - 1)):
        services.project_records.create("system", f"Project {i}", _project_document(rng, names), board_id=board_ids[i])
    template = _workflow_template()
    for i in range(WORKFLOWS):
        tags = ", ".join(rng.sample(WORKFLOW_TAGS, rng.randint(1, 3)))
        workflow = template.model_copy(update={"name": f"Workflow {i}", "tags": tags})
        services.workflow_records.create(workflow, user_id="system", is_public=i % 10 == 0)
    for i in range(STYLE_PRESETS):
        data = PresetData(positive_prompt=f"style {i}, {{prompt}}, highly detailed", negative_prompt="blurry, lowres")
        preset = StylePresetWithoutId(name=f"Preset {i}", preset_data=data, type=PresetType.User, is_public=i % 5 == 0)
        services.style_preset_records.create(preset, user_id="user-0" if i % 2 else "user-1")
    for i in range(SYSTEM_PROMPTS):
        prompt = SystemPromptWithoutId(name=f"Prompt {i}", content="Expand the prompt. " * 20)
        services.system_prompt_records.create(prompt, user_id="user-0", is_public=i % 4 == 0)
    values = [f"value {i}" for i in range(20)]
    for i in range(WILDCARDS):
        services.wildcard_records.create(WildcardWithoutId(name=f"set{i}", values=values), user_id="user-0")
    for i in range(MODELS):
        services.model_records.add_model(_model_config(i))
    return names


def _operations(
    services: Services, names: list[str], rng: random.Random
) -> dict[str, tuple[Callable[[], object], int]]:
    """Operation name -> (one call, how many calls to time)."""
    from invokeai.app.services.board_records.board_records_common import BoardRecordOrderBy
    from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
    from invokeai.app.services.model_records import ModelRecordChanges

    try:
        from invokeai.app.services.shared.pagination import SQLiteDirection
    except ImportError:  # A base from before it moved there.
        from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection  # type: ignore[no-redef]
    from invokeai.app.services.style_preset_records.style_preset_records_common import StylePresetChanges
    from invokeai.app.services.system_prompt_records.system_prompt_records_common import SystemPromptChanges
    from invokeai.app.services.wildcard_records.wildcard_records_common import WildcardWithoutId
    from invokeai.app.services.workflow_records.workflow_records_common import (
        Workflow,
        WorkflowCategory,
        WorkflowRecordOrderBy,
    )

    sample = iter(rng.choices(names, k=100_000))
    a_board = services.board_records.get_all(
        user_id="system", is_admin=True, order_by=BoardRecordOrderBy.CreatedAt, direction=SQLiteDirection.Descending
    )[0].board_id

    def board_dto_queries() -> object:
        # The queries of `BoardService.get_dto`: the board with the project claiming it, and its counts.
        return (
            services.board_records.get_with_project_id(a_board),
            services.board_image_records.get_counts_for_board(a_board),
            services.board_video_records.get_counts_for_board(a_board),
        )

    project_ids = [summary.project_id for summary in services.project_records.list("system")]
    autosave_documents = [_project_document(rng, names) for _ in range(10)]
    revision: list[int] = []

    def autosave() -> object:
        # One project saved again and again, each time from the revision the last save returned. The timed and the
        # counted pass save the same project: each reads its revision on its first call, the warm-up.
        if not revision:
            revision.append(services.project_records.get("system", project_ids[0]).revision)
        record = services.project_records.update(
            "system", project_ids[0], revision[0], "Autosaved", autosave_documents[revision[0] % 10]
        )
        revision[0] = record.revision
        return record

    library = [WorkflowCategory.User]
    workflows = services.workflow_records.get_many(
        WorkflowRecordOrderBy.CreatedAt, SQLiteDirection.Ascending, library, user_id="system"
    ).items
    workflow_ids = [workflow.workflow_id for workflow in workflows]
    edited = services.workflow_records.get(workflow_ids[0]).workflow
    edits = [Workflow(**edited.model_copy(update={"notes": f"Edit {i}"}).model_dump()) for i in range(10)]
    saves = iter(range(10**9))

    def library_page() -> object:
        # What `GET /v1/workflows` asks of the storage for a page of 50: the page, then each workflow, which the
        # route checks for being callable.
        page = services.workflow_records.get_many(
            WorkflowRecordOrderBy.UpdatedAt, SQLiteDirection.Descending, library, page=0, per_page=50
        )
        return [services.workflow_records.get(item.workflow_id) for item in page.items]

    own_presets = [
        preset.id for preset in services.style_preset_records.get_many(user_id="user-0") if preset.user_id == "user-0"
    ]
    own_prompts = [
        prompt.id for prompt in services.system_prompt_records.get_many(user_id="user-0") if prompt.user_id == "user-0"
    ]
    wildcard_values = [f"value {i}" for i in range(20)]
    model_keys = [str(uuid.UUID(int=i)) for i in range(MODELS)]

    general = [ImageCategory.GENERAL]
    video_names = services.video_records.get_video_names(user_id="system", is_admin=True).video_names

    def save_image() -> object:
        return services.image_records.save(
            image_name=f"{uuid.uuid4()}.png",
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            width=1024,
            height=1024,
            has_workflow=False,
            metadata=_metadata(rng),
            user_id="system",
        )

    return {
        "image_records.get": (lambda: services.image_records.get(next(sample)), 1000),
        "image_records.get_many(page of 100)": (
            lambda: services.image_records.get_many(
                limit=100, categories=general, is_intermediate=False, user_id="system", is_admin=True
            ),
            50,
        ),
        "image_records.get_image_names(all)": (
            lambda: services.image_records.get_image_names(
                categories=general, is_intermediate=False, user_id="system", is_admin=True
            ),
            10,
        ),
        # A non-admin's "all" view: images on no board that it owns, and those on boards it may read.
        "image_records.get_image_names(all, account)": (
            lambda: services.image_records.get_image_names(
                categories=general, is_intermediate=False, board_id="all", user_id="user-1", is_admin=False
            ),
            10,
        ),
        "video_records.get": (lambda: services.video_records.get(rng.choice(video_names)), 1000),
        "video_records.get_many(page of 100)": (
            lambda: services.video_records.get_many(
                limit=100, categories=general, is_intermediate=False, user_id="system", is_admin=True
            ),
            50,
        ),
        "video_records.get_video_names(all)": (
            lambda: services.video_records.get_video_names(
                categories=general, is_intermediate=False, user_id="system", is_admin=True
            ),
            20,
        ),
        "gallery.get_item_names(all)": (
            lambda: services.gallery.get_item_names(
                categories=general, is_intermediate=False, user_id="system", is_admin=True
            ),
            10,
        ),
        "gallery.list_items(page of 100)": (
            lambda: services.gallery.list_items(
                limit=100, categories=general, is_intermediate=False, user_id="system", is_admin=True
            ),
            50,
        ),
        "board_records.get_all": (
            lambda: services.board_records.get_all(
                user_id="system",
                is_admin=True,
                order_by=BoardRecordOrderBy.CreatedAt,
                direction=SQLiteDirection.Descending,
            ),
            50,
        ),
        "boards.get_dto (record and counts)": (board_dto_queries, 500),
        "image_records.save": (save_image, 200),
        # Every authenticated request reads its account.
        "users.get": (lambda: services.users.get(f"user-{rng.randrange(ACCOUNTS)}"), 1000),
        "users.list_users(page of 100)": (lambda: services.users.list_users(limit=100), 50),
        "client_state.get_keys_by_prefix(all)": (
            lambda: services.client_state.get_keys_by_prefix("system", "canvas_snapshot:"),
            200,
        ),
        "client_state.set_by_key": (
            lambda: services.client_state.set_by_key("system", "canvas", json.dumps({"imageName": next(sample)})),
            200,
        ),
        "projects.get (25 KB document)": (lambda: services.project_records.get("system", rng.choice(project_ids)), 500),
        "projects.list": (lambda: services.project_records.list("system"), 200),
        "projects.update (autosave)": (autosave, 200),
        "workflows.get": (lambda: services.workflow_records.get(rng.choice(workflow_ids)), 500),
        "workflows.library page (list + get each)": (library_page, 20),
        "workflows.get_many(page of 50)": (
            lambda: services.workflow_records.get_many(
                WorkflowRecordOrderBy.Name, SQLiteDirection.Ascending, library, page=0, per_page=50, user_id="system"
            ),
            50,
        ),
        "workflows.update": (lambda: services.workflow_records.update(edits[next(saves) % 10]), 200),
        "workflows.counts_by_tag(5)": (
            lambda: services.workflow_records.counts_by_tag(WORKFLOW_TAGS[:5], library, user_id="system"),
            50,
        ),
        # A prompt-template node reads its preset, a text-LLM node its system prompt, and every prompt expansion
        # reads the account's wildcards.
        "style_presets.get (prompt template)": (
            lambda: services.style_preset_records.get(own_presets[0]),
            1000,
        ),
        "style_presets.get_many (account)": (
            lambda: services.style_preset_records.get_many(user_id="user-0"),
            200,
        ),
        "style_presets.update": (
            lambda: services.style_preset_records.update(
                own_presets[1], StylePresetChanges(name=f"Preset {next(saves)}", type=None)
            ),
            200,
        ),
        "system_prompts.get (text LLM)": (lambda: services.system_prompt_records.get(own_prompts[0]), 1000),
        "system_prompts.update": (
            lambda: services.system_prompt_records.update(
                own_prompts[1], SystemPromptChanges(content=f"Expand {next(saves)}"), user_id="user-0"
            ),
            200,
        ),
        "wildcards.get_many (dynamic prompts)": (lambda: services.wildcard_records.get_many("user-0"), 500),
        "wildcards.create": (
            lambda: services.wildcard_records.create(
                WildcardWithoutId(name=f"new{uuid.uuid4().hex}", values=wildcard_values), user_id="user-0"
            ),
            200,
        ),
        # Loading a model reads its config, and the model manager lists every model.
        "models.get_model": (lambda: services.model_records.get_model(rng.choice(model_keys)), 1000),
        "models.exists": (lambda: services.model_records.exists(rng.choice(model_keys)), 1000),
        "models.search_by_attr(all)": (lambda: services.model_records.search_by_attr(), 20),
        "models.update_model": (
            lambda: services.model_records.update_model(
                model_keys[0], ModelRecordChanges(description=f"Edit {next(saves)}")
            ),
            200,
        ),
    }


def _time(call: Callable[[], object], repeats: int) -> tuple[float, float]:
    call()
    durations: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        call()
        durations.append((time.perf_counter_ns() - started) / 1e6)
    durations.sort()
    return statistics.median(durations), durations[min(len(durations) - 1, int(len(durations) * 0.95))]


def run(images: int, boards: int, seed: int) -> dict[str, Any]:
    import invokeai.app
    from invokeai.app.services.config.config_default import InvokeAIAppConfig
    from invokeai.app.services.image_files.image_files_base import ImageFileStorageBase

    try:
        from invokeai.app.services.shared.database.database import Database
        from invokeai.app.services.shared.database.startup import init_database

        def open_database(path: Path, logger: logging.Logger, verbose: bool = False) -> Any:
            return Database.open_sqlite(path, logger, verbose=verbose)

    except ImportError:  # A base from before the cursor facade was removed.
        from invokeai.app.services.shared.sqlite.sqlite_database import (  # type: ignore[no-redef]
            SqliteDatabase as open_database,
        )
        from invokeai.app.services.shared.sqlite.sqlite_util import init_db as init_database

    quiet = logging.getLogger("benchmark_database.quiet")
    quiet.setLevel(logging.WARNING)
    counting = logging.getLogger("benchmark_database.statements")
    counting.setLevel(logging.DEBUG)
    counting.propagate = False
    counter = _StatementCounter()
    counting.addHandler(counter)

    # Code from before the database layer cannot close its connection, and Windows refuses to delete an
    # open database file: such a run leaves its temporary directory behind.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        # Seeding commits tens of thousands of times; `normal` keeps that short. The timed connection
        # below uses the default `full`, so commits are timed as an install makes them.
        config = InvokeAIAppConfig(db_dir=Path(tmp), db_synchronous="normal")
        # The migrations clean up files under the root (legacy caches and models): never a real install's.
        config._root = Path(tmp)
        seeding_db = init_database(config=config, logger=quiet, image_files=mock.Mock(spec=ImageFileStorageBase))
        rng = random.Random(seed)
        started = time.perf_counter()
        names = _seed(Services(seeding_db), images, boards, rng)
        seeded_in = time.perf_counter() - started

        timed_db = open_database(config.db_path, quiet)
        counted_db = open_database(config.db_path, counting, verbose=True)
        timed = _operations(Services(timed_db), names, random.Random(seed + 1))
        counted = _operations(Services(counted_db), names, random.Random(seed + 1))

        results: dict[str, dict[str, float]] = {}
        for name, (call, repeats) in timed.items():
            median_ms, p95_ms = _time(call, repeats)
            count_call, _ = counted[name]
            count_call()  # warm up the same way the timed pass did
            counter.count = 0
            count_call()
            results[name] = {"median_ms": median_ms, "p95_ms": p95_ms, "statements": counter.count, "calls": repeats}

        for db in (seeding_db, timed_db, counted_db):
            database = getattr(db, "database", db)
            if hasattr(database, "dispose"):
                database.dispose()

    return {
        # `invokeai.app`, because an editable install leaves the top-level package without a `__file__`.
        "code": str(Path(invokeai.app.__file__).parent.parent),
        "python": sys.version.split()[0],
        "images": images,
        "boards": boards,
        "seeded_in_s": round(seeded_in, 1),
        "operations": results,
    }


def compare(base: dict[str, Any], head: dict[str, Any]) -> int:
    print(f"base: {base['code']}\nhead: {head['code']}\n")
    print(f"{'operation':40} {'base ms':>9} {'head ms':>9} {'change':>8} {'stmts':>9}  budget")
    over_budget = 0
    for name, after in head["operations"].items():
        before = base["operations"].get(name)
        if before is None:
            print(f"{name:40} {'-':>9} {after['median_ms']:9.3f}")
            continue
        change = (after["median_ms"] - before["median_ms"]) / before["median_ms"]
        allowed = max(before["median_ms"] * BUDGET_FRACTION, BUDGET_FLOOR_MS)
        within = after["median_ms"] - before["median_ms"] <= allowed
        over_budget += not within
        statements = f"{before['statements']}->{after['statements']}"
        verdict = "ok" if within else "OVER"
        print(
            f"{name:40} {before['median_ms']:9.3f} {after['median_ms']:9.3f} {change:+8.1%} {statements:>9}  {verdict}"
        )
    return 1 if over_budget else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=int, default=20_000)
    parser.add_argument("--boards", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path, help="write the results to this file")
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("BASE", "HEAD"), help="compare two result files")
    args = parser.parse_args()
    if args.boards < 2:
        parser.error("--boards must be at least 2: projects claim the oldest boards, and the newest stays unclaimed")

    if args.compare:
        base, head = (json.loads(path.read_text()) for path in args.compare)
        return compare(base, head)

    results = run(args.images, args.boards, args.seed)
    for name, result in results["operations"].items():
        print(
            f"{name:40} median {result['median_ms']:8.3f} ms   p95 {result['p95_ms']:8.3f} ms   "
            f"{result['statements']} statements"
        )
    if args.json:
        args.json.write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
