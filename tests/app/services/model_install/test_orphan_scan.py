"""The startup orphan scan must survive a model file it cannot register.

`_register_orphaned_models` runs inside `ModelInstallService.start()`, which stores and re-raises
anything that escapes it. Identification can now reject a file outright — a checkpoint recognised as
a PiD decoder and found unusable, or anything at all when `allow_unknown_models` is off — where it
previously always produced at worst an `Unknown_Config` record.

Two layers keep that from ending the boot: `ModelSearch._walk_directory` wraps every
`on_model_found` call, and the callback itself catches `InvalidModelConfigException`. This pins the
outcome rather than either mechanism, so removing one without the other is caught here: startup
completes, and the unusable file is not registered.
"""

from pathlib import Path

import torch
from requests.sessions import Session

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.download import DownloadQueueServiceBase
from invokeai.app.services.model_install import ModelInstallService
from invokeai.app.services.model_records import ModelRecordServiceSQL
from invokeai.backend.util.logging import InvokeAILogger
from tests.backend.model_manager.model_manager_fixtures import *  # noqa F403
from tests.fixtures.sqlite_database import create_mock_sqlite_database
from tests.test_nodes import TestEventService

# A single `lq_proj.*` weight: enough for identification to recognise a PiD decoder, far too little
# for one to be built from. `.pth` is the format NVIDIA ships.
_ORPHAN_RELATIVE_PATH = "pid/model_ema_bf16.pth"


def test_startup_scan_skips_a_checkpoint_it_cannot_register(
    mm2_app_config: InvokeAIAppConfig,
    mm2_download_queue: DownloadQueueServiceBase,
    mm2_session: Session,
) -> None:
    orphan = mm2_app_config.models_path / _ORPHAN_RELATIVE_PATH
    orphan.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"net.lq_proj.latent_proj.1.weight": torch.zeros(1)}, orphan)
    mm2_app_config.scan_models_on_startup = True

    logger = InvokeAILogger.get_logger()
    installer = ModelInstallService(
        app_config=mm2_app_config,
        record_store=ModelRecordServiceSQL(create_mock_sqlite_database(mm2_app_config, logger), logger),
        download_queue=mm2_download_queue,
        event_bus=TestEventService(),
        session=mm2_session,
    )

    try:
        installer.start()
        assert installer._startup_error is None
        assert _ORPHAN_RELATIVE_PATH not in {Path(m.path).as_posix() for m in installer.record_store.all_models()}
    finally:
        installer.stop()
