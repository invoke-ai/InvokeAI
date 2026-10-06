import asyncio
from logging import Logger

import torch

from invokeai.app.services.app_settings import AppSettingsService
from invokeai.app.services.auth.token_service import set_jwt_secret
from invokeai.app.services.board_image_records.board_image_records_default import BoardImageRecordStorage
from invokeai.app.services.board_images.board_images_default import BoardImagesService
from invokeai.app.services.board_records.board_records_default import BoardRecordStorage
from invokeai.app.services.board_video_records.board_video_records_default import BoardVideoRecordStorage
from invokeai.app.services.boards.boards_default import BoardService
from invokeai.app.services.bulk_download.bulk_download_default import BulkDownloadService
from invokeai.app.services.client_state_persistence.client_state_persistence_default import ClientStatePersistence
from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.download.download_default import DownloadQueueService
from invokeai.app.services.events.events_fastapievents import FastAPIEventService
from invokeai.app.services.external_generation.external_generation_default import ExternalGenerationService
from invokeai.app.services.external_generation.providers import (
    AlibabaCloudProvider,
    GeminiProvider,
    OpenAIProvider,
    SeedreamProvider,
)
from invokeai.app.services.external_generation.startup import sync_configured_external_starter_models
from invokeai.app.services.fonts.fonts_default import FontService
from invokeai.app.services.gallery.gallery_default import GalleryService
from invokeai.app.services.image_files.image_files_disk import DiskImageFileStorage
from invokeai.app.services.image_index.image_index_default import ImageIndexService, warm_up_attention
from invokeai.app.services.image_index.image_index_records_default import ImageIndexRecords
from invokeai.app.services.image_moves.image_moves_default import ImageMoveService
from invokeai.app.services.image_records.image_records_default import ImageRecordStorage
from invokeai.app.services.images.images_default import ImageService
from invokeai.app.services.intermediates.intermediates_default import IntermediatesService
from invokeai.app.services.intermediates.intermediates_records_default import IntermediatesRecords
from invokeai.app.services.invocation_cache.invocation_cache_memory import MemoryInvocationCache
from invokeai.app.services.invocation_services import InvocationServices
from invokeai.app.services.invocation_stats.invocation_stats_default import InvocationStatsService
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.model_images.model_images_default import ModelImageFileStorageDisk
from invokeai.app.services.model_manager.model_manager_default import ModelManagerService
from invokeai.app.services.model_records.model_records_sql import ModelRecordServiceSQL
from invokeai.app.services.model_relationship_records.model_relationship_records_default import (
    ModelRelationshipRecordStorage,
)
from invokeai.app.services.model_relationships.model_relationships_default import ModelRelationshipsService
from invokeai.app.services.names.names_default import SimpleNameService
from invokeai.app.services.object_serializer.object_serializer_disk import ObjectSerializerDisk
from invokeai.app.services.object_serializer.object_serializer_forward_cache import ObjectSerializerForwardCache
from invokeai.app.services.progress_previews.progress_previews_default import MemoryProgressPreviews
from invokeai.app.services.project_records.project_records_default import ProjectRecordsStorage
from invokeai.app.services.session_processor.session_processor_default import (
    DefaultSessionProcessor,
    DefaultSessionRunner,
)
from invokeai.app.services.session_queue.session_queue_sqlite import SqliteSessionQueue
from invokeai.app.services.shared.sqlite.sqlite_util import init_db
from invokeai.app.services.style_preset_images.style_preset_images_disk import StylePresetImageFileStorageDisk
from invokeai.app.services.style_preset_records.style_preset_records_default import StylePresetRecordsStorage
from invokeai.app.services.system_prompt_records.system_prompt_records_default import SystemPromptRecordsStorage
from invokeai.app.services.urls.urls_default import LocalUrlService
from invokeai.app.services.users.users_default import UserService
from invokeai.app.services.video_files.video_files_disk import DiskVideoFileStorage
from invokeai.app.services.video_records.video_records_default import VideoRecordStorage
from invokeai.app.services.videos.videos_default import VideoService
from invokeai.app.services.wildcard_records.wildcard_records_default import WildcardRecordsStorage
from invokeai.app.services.workflow_records.workflow_records_default import WorkflowRecordsStorage
from invokeai.app.services.workflow_thumbnails.workflow_thumbnails_disk import WorkflowThumbnailFileStorageDisk
from invokeai.backend.architectures import conditioning_safe_globals
from invokeai.backend.architectures import validate as validate_architectures
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ConditioningFieldData
from invokeai.backend.util.logging import InvokeAILogger
from invokeai.version.invokeai_version import __version__


# TODO: is there a better way to achieve this?
def check_internet() -> bool:
    """
    Return true if the internet is reachable.
    It does this by pinging huggingface.co.
    """
    import urllib.request

    host = "http://huggingface.co"
    try:
        urllib.request.urlopen(host, timeout=1)
        return True
    except Exception:
        return False


logger = InvokeAILogger.get_logger()


class ApiDependencies:
    """Contains and initializes all dependencies for the API"""

    invoker: Invoker

    @staticmethod
    def initialize(
        config: InvokeAIAppConfig,
        event_handler_id: int,
        loop: asyncio.AbstractEventLoop,
        logger: Logger = logger,
    ) -> None:
        # The one authoritative architecture gate. Every entry point that builds services comes
        # through here — `run_app`'s lifespan, and the embedders and tests that never go through
        # `run_app` at all — so this is the only place it is called. It runs before anything else
        # because `ObjectSerializerDisk` below derives its `safe_globals` from the registry, and
        # that mutates process-global torch state. The import above is at module scope rather than
        # lazily inside this function for the same reason, one step earlier: it is what fills the
        # registry, and importing this module is all `scripts/generate_openapi_schema.py` does.
        validate_architectures()

        logger.info(f"InvokeAI version {__version__}")
        logger.info(f"Root directory = {str(config.root_path)}")

        output_folder = config.outputs_path
        if output_folder is None:
            raise ValueError("Output folder is not set")

        image_files = DiskImageFileStorage(f"{output_folder}/images")
        video_files = DiskVideoFileStorage(f"{output_folder}/videos")

        model_images_folder = config.models_path
        style_presets_folder = config.style_presets_path
        workflow_thumbnails_folder = config.workflow_thumbnails_path

        db = init_db(config=config, logger=logger, image_files=image_files)

        # Initialize JWT secret from database
        app_settings = AppSettingsService(db.database)
        jwt_secret = app_settings.get_jwt_secret()
        set_jwt_secret(jwt_secret)
        logger.info("JWT secret loaded from database")

        configuration = config
        logger = logger

        board_image_records = BoardImageRecordStorage(db.database)
        board_images = BoardImagesService()
        board_records = BoardRecordStorage(db.database)
        boards = BoardService()
        events = FastAPIEventService(event_handler_id, loop=loop)
        bulk_download = BulkDownloadService()
        image_records = ImageRecordStorage(db.database)
        image_moves = ImageMoveService(db=db, image_files=image_files, config=configuration, logger=logger)
        images = ImageService()
        video_records = VideoRecordStorage(db.database)
        videos = VideoService()
        board_video_records = BoardVideoRecordStorage(db.database)
        gallery = GalleryService(db.database)
        invocation_cache = MemoryInvocationCache(max_cache_size=config.node_cache_size)
        tensors = ObjectSerializerForwardCache(
            ObjectSerializerDisk[torch.Tensor](
                output_folder / "tensors",
                safe_globals=[torch.Tensor],
                ephemeral=True,
            ),
        )
        conditioning = ObjectSerializerForwardCache(
            ObjectSerializerDisk[ConditioningFieldData](
                output_folder / "conditioning",
                # Every architecture's conditioning class, from what each declares under
                # invokeai/backend/architectures/defs/. Missing one here fails nowhere near
                # here: the encoder runs, writes its output, and the denoise node then dies
                # unpickling it — which is why the list is assembled in one place and this call
                # site is asserted against it in tests/backend/architectures/test_conditioning.py.
                safe_globals=conditioning_safe_globals(),
                ephemeral=True,
            ),
        )
        download_queue_service = DownloadQueueService(app_config=configuration, event_bus=events)
        model_record_service = ModelRecordServiceSQL(db.database, logger=logger)
        model_manager = ModelManagerService.build_model_manager(
            app_config=configuration,
            model_record_service=model_record_service,
            download_queue=download_queue_service,
            events=events,
        )
        external_generation = ExternalGenerationService(
            providers={
                AlibabaCloudProvider.provider_id: AlibabaCloudProvider(app_config=configuration, logger=logger),
                GeminiProvider.provider_id: GeminiProvider(app_config=configuration, logger=logger),
                OpenAIProvider.provider_id: OpenAIProvider(app_config=configuration, logger=logger),
                SeedreamProvider.provider_id: SeedreamProvider(app_config=configuration, logger=logger),
            },
            logger=logger,
            record_store=model_record_service,
        )
        model_images_service = ModelImageFileStorageDisk(model_images_folder / "model_images")
        model_relationships = ModelRelationshipsService()
        model_relationship_records = ModelRelationshipRecordStorage(db.database)
        names = SimpleNameService()
        performance_statistics = InvocationStatsService()
        session_processor = DefaultSessionProcessor(session_runner=DefaultSessionRunner())
        session_queue = SqliteSessionQueue(db=db)
        urls = LocalUrlService()
        workflow_records = WorkflowRecordsStorage(db.database)
        style_preset_records = StylePresetRecordsStorage(db.database)
        wildcard_records = WildcardRecordsStorage(db.database)
        style_preset_image_files = StylePresetImageFileStorageDisk(style_presets_folder / "images")
        system_prompt_records = SystemPromptRecordsStorage(db.database)
        workflow_thumbnails = WorkflowThumbnailFileStorageDisk(workflow_thumbnails_folder)
        client_state_persistence = ClientStatePersistence(db.database)
        project_records = ProjectRecordsStorage(db.database)
        users = UserService(db.database)
        image_index_records = ImageIndexRecords(db.database)
        image_index = ImageIndexService()
        intermediates = IntermediatesService(records=IntermediatesRecords(db.database), logger=logger)
        fonts = FontService(
            db=db,
            fonts_dir=configuration.fonts_path,
            storage_dir=configuration.fonts_storage_path,
            logger=logger,
            max_upload_bytes=configuration.max_font_upload_bytes,
            max_library_bytes=configuration.max_font_library_bytes,
        )

        services = InvocationServices(
            board_image_records=board_image_records,
            board_images=board_images,
            board_records=board_records,
            boards=boards,
            bulk_download=bulk_download,
            configuration=configuration,
            events=events,
            image_files=image_files,
            image_moves=image_moves,
            progress_previews=MemoryProgressPreviews(),
            image_records=image_records,
            images=images,
            invocation_cache=invocation_cache,
            logger=logger,
            model_images=model_images_service,
            model_manager=model_manager,
            model_relationships=model_relationships,
            model_relationship_records=model_relationship_records,
            download_queue=download_queue_service,
            external_generation=external_generation,
            names=names,
            performance_statistics=performance_statistics,
            session_processor=session_processor,
            session_queue=session_queue,
            urls=urls,
            workflow_records=workflow_records,
            tensors=tensors,
            conditioning=conditioning,
            style_preset_records=style_preset_records,
            wildcard_records=wildcard_records,
            style_preset_image_files=style_preset_image_files,
            system_prompt_records=system_prompt_records,
            workflow_thumbnails=workflow_thumbnails,
            client_state_persistence=client_state_persistence,
            project_records=project_records,
            users=users,
            videos=videos,
            video_files=video_files,
            video_records=video_records,
            board_video_records=board_video_records,
            gallery=gallery,
            image_index_records=image_index_records,
            image_index=image_index,
            fonts=fonts,
            intermediates=intermediates,
        )

        # Constructing the Invoker starts every service, including the session
        # processor thread (which may immediately resume queue items that were
        # pending at shutdown). Trigger torch attention's lazy, non-thread-safe
        # kernel init while the process is still single-threaded.
        warm_up_attention(config, logger)

        ApiDependencies.invoker = Invoker(services)
        configured_external_providers = {
            provider_id
            for provider_id, status in external_generation.get_provider_statuses().items()
            if status.configured
        }
        sync_configured_external_starter_models(
            configured_provider_ids=configured_external_providers,
            model_manager=model_manager,
            logger=logger,
        )
        db.clean()

    @staticmethod
    def shutdown() -> None:
        if ApiDependencies.invoker:
            ApiDependencies.invoker.stop()
