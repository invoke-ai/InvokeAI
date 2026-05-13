import argparse
import sys
from pathlib import Path
from typing import Sequence

from invokeai.app.services.config.config_default import InvokeAIAppConfig, load_and_migrate_config
from invokeai.app.services.image_files.image_files_disk import DiskImageFileStorage
from invokeai.app.services.image_moves.image_moves_default import ImageMoveResult, ImageMoveService
from invokeai.app.services.session_queue.session_queue_sqlite import SqliteSessionQueue
from invokeai.app.services.shared.sqlite.sqlite_util import init_db
from invokeai.backend.util.logging import InvokeAILogger


def build_image_move_service(root: Path | None = None, config_file: Path | None = None) -> ImageMoveService:
    config = InvokeAIAppConfig()
    if root is not None:
        config._root = root
    if config_file is not None:
        config._config_file = config_file

    if config.config_file_path.exists():
        config.update_config(load_and_migrate_config(config.config_file_path), clobber=False)

    if config.outputs_path is None:
        raise RuntimeError("Output folder is not set")

    logger = InvokeAILogger.get_logger()
    image_files = DiskImageFileStorage(config.outputs_path / "images")
    db = init_db(config=config, logger=logger, image_files=image_files)
    service = ImageMoveService(db=db, image_files=image_files, config=config, logger=logger)
    service.set_session_queue(SqliteSessionQueue(db=db))
    return service


def _print_result(operation: str, result: ImageMoveResult) -> None:
    print(
        f"{operation}: planned={result.planned}, committed={result.committed}, errors={result.errors}",
        flush=True,
    )


def _print_status(service: ImageMoveService) -> bool:
    latest_job = service.get_latest_job()
    active_job_id = service.get_active_job_id()
    if latest_job is None:
        print("No image storage maintenance jobs found.", flush=True)
    else:
        print(
            "Latest image storage maintenance job: "
            f"id={latest_job.id}, state={latest_job.state}, error={latest_job.error_message or 'none'}",
            flush=True,
        )
    if active_job_id is not None:
        print(f"Active image storage maintenance job: id={active_job_id}", flush=True)
    return active_job_id is not None or (latest_job is not None and latest_job.state == "error")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="InvokeAI image storage maintenance utility")
    parser.add_argument("operation", choices=["status", "recover", "move"], help="Operation to perform.")
    parser.add_argument("--root", type=Path, default=None, help="InvokeAI root directory.")
    parser.add_argument("--config", dest="config_file", type=Path, default=None, help="Path to invokeai.yaml.")
    args = parser.parse_args(argv)

    try:
        service = build_image_move_service(root=args.root, config_file=args.config_file)
        # TODO: Add an interprocess guard so this script cannot run image moves while Invoke is active.
        if args.operation == "status":
            requires_attention = _print_status(service)
            return 1 if requires_attention else 0
        if args.operation == "recover":
            result = service.startup_recovery()
        else:
            service.assert_no_active_queue_work()
            result = service.move_all_images()
        _print_result(args.operation, result)
        if result.errors > 0 or service.is_maintenance_active():
            print("Image storage maintenance requires operator attention.", file=sys.stderr, flush=True)
            return 1
        return 0
    except KeyboardInterrupt:
        print("Image storage maintenance canceled.", file=sys.stderr, flush=True)
        return 130
    except Exception as e:
        print(f"Image storage maintenance failed: {e}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
