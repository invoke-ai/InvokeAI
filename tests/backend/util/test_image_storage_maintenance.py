from dataclasses import dataclass

from invokeai.app.services.image_moves.image_moves_default import ImageMoveJob, ImageMoveQueueActive, ImageMoveResult
from invokeai.backend.util import image_storage_maintenance


@dataclass
class _FakeImageMoveService:
    result: ImageMoveResult = ImageMoveResult()
    active: bool = False
    latest_job: ImageMoveJob | None = None
    recovered: bool = False
    moved: bool = False
    checked_queue: bool = False

    def startup_recovery(self) -> ImageMoveResult:
        self.recovered = True
        return self.result

    def move_all_images(self) -> ImageMoveResult:
        self.moved = True
        return self.result

    def assert_no_active_queue_work(self) -> None:
        self.checked_queue = True

    def is_maintenance_active(self) -> bool:
        return self.active

    def get_latest_job(self) -> ImageMoveJob | None:
        return self.latest_job

    def get_active_job_id(self) -> int | None:
        return self.latest_job.id if self.active and self.latest_job is not None else None


def test_script_recover_uses_shared_image_move_service(monkeypatch, capsys) -> None:
    service = _FakeImageMoveService(result=ImageMoveResult(committed=2))
    monkeypatch.setattr(image_storage_maintenance, "build_image_move_service", lambda root, config_file: service)

    exit_code = image_storage_maintenance.main(["recover", "--root", "/tmp/invokeai"])

    assert exit_code == 0
    assert service.recovered is True
    assert service.moved is False
    assert "recover: planned=0, committed=2, errors=0" in capsys.readouterr().out


def test_script_move_exits_nonzero_when_job_remains_active(monkeypatch, capsys) -> None:
    service = _FakeImageMoveService(
        result=ImageMoveResult(planned=1, committed=0, errors=1),
        active=True,
        latest_job=ImageMoveJob(id=1, state="planned", error_message="temporary failure"),
    )
    monkeypatch.setattr(image_storage_maintenance, "build_image_move_service", lambda root, config_file: service)

    exit_code = image_storage_maintenance.main(["move"])

    assert exit_code == 1
    assert service.checked_queue is True
    assert service.moved is True
    assert "requires operator attention" in capsys.readouterr().err


def test_script_move_rejects_active_queue_work(monkeypatch, capsys) -> None:
    service = _FakeImageMoveService()

    def raise_active_queue() -> None:
        service.checked_queue = True
        raise ImageMoveQueueActive("queue work is active")

    service.assert_no_active_queue_work = raise_active_queue
    monkeypatch.setattr(image_storage_maintenance, "build_image_move_service", lambda root, config_file: service)

    exit_code = image_storage_maintenance.main(["move"])

    assert exit_code == 1
    assert service.checked_queue is True
    assert service.moved is False
    assert "queue work is active" in capsys.readouterr().err


def test_script_status_reports_active_job(monkeypatch, capsys) -> None:
    service = _FakeImageMoveService(
        active=True,
        latest_job=ImageMoveJob(id=7, state="planned", error_message="temporary failure"),
    )
    monkeypatch.setattr(image_storage_maintenance, "build_image_move_service", lambda root, config_file: service)

    exit_code = image_storage_maintenance.main(["status"])

    assert exit_code == 1
    output = capsys.readouterr().out
    assert "id=7, state=planned, error=temporary failure" in output
    assert "Active image storage maintenance job: id=7" in output


def test_script_status_exits_nonzero_for_error_job(monkeypatch, capsys) -> None:
    service = _FakeImageMoveService(
        active=False,
        latest_job=ImageMoveJob(id=8, state="error", error_message="ambiguous filesystem state"),
    )
    monkeypatch.setattr(image_storage_maintenance, "build_image_move_service", lambda root, config_file: service)

    exit_code = image_storage_maintenance.main(["status"])

    assert exit_code == 1
    assert "id=8, state=error, error=ambiguous filesystem state" in capsys.readouterr().out
