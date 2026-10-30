"""The console-script entry module: the process-wide environment defaults it pins before any heavy import, and the
server's runtime, the startup steps that decide kernels and allocation, which the calibration scripts apply too."""

import importlib
import logging
import os

import pytest

from invokeai.app.services.config.config_default import InvokeAIAppConfig


@pytest.fixture
def reload_run_app(monkeypatch: pytest.MonkeyPatch):
    """Re-execute ``invokeai.app.run_app``'s module body under a patched environment."""
    import invokeai.app.run_app as run_app

    def reload() -> None:
        importlib.reload(run_app)

    yield reload
    importlib.reload(run_app)


@pytest.mark.parametrize("name, default", [("MIOPEN_FIND_MODE", "FAST"), ("TOKENIZERS_PARALLELISM", "false")])
def test_run_app_sets_environment_defaults(name: str, default: str, monkeypatch: pytest.MonkeyPatch, reload_run_app):
    monkeypatch.delenv(name, raising=False)
    reload_run_app()
    assert os.environ[name] == default


@pytest.mark.parametrize("name", ["MIOPEN_FIND_MODE", "TOKENIZERS_PARALLELISM"])
def test_run_app_keeps_an_explicit_environment_value(name: str, monkeypatch: pytest.MonkeyPatch, reload_run_app):
    monkeypatch.setenv(name, "user-choice")
    reload_run_app()
    assert os.environ[name] == "user-choice"


@pytest.fixture
def runtime_steps(monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    """The runtime's steps, recorded instead of run, each with the settings it was given; the first also with the
    ``MIOPEN_FIND_MODE`` it ran under. Each does its work on its own platform only, and an explicit allocator
    configuration refuses to apply once torch is imported."""
    from invokeai.app.util import rocm_aotriton, rocm_integrated_gpu, startup_utils, torch_cuda_allocator

    steps: list[tuple] = []

    def hide_integrated_gpus(logger: logging.Logger) -> None:
        steps.append(("hide integrated GPUs", os.environ.get("MIOPEN_FIND_MODE")))

    def configure_allocator(conf: str, logger: logging.Logger) -> None:
        steps.append(("configure the allocator", conf))

    def allocator_default(logger: logging.Logger) -> None:
        steps.append(("the platform's allocator default",))

    def aotriton(setting: str, generation_devices: object, logger: logging.Logger) -> None:
        steps.append(("AOTriton", setting, generation_devices))

    def monkeypatches(attention_backend: str) -> None:
        steps.append(("monkeypatches", attention_backend))

    monkeypatch.setattr(rocm_integrated_gpu, "hide_integrated_gpus_on_rocm_windows", hide_integrated_gpus)
    monkeypatch.setattr(torch_cuda_allocator, "configure_torch_cuda_allocator", configure_allocator)
    monkeypatch.setattr(torch_cuda_allocator, "apply_rocm_windows_allocator_default", allocator_default)
    monkeypatch.setattr(rocm_aotriton, "apply_rocm_aotriton_setting", aotriton)
    monkeypatch.setattr(startup_utils, "apply_monkeypatches", monkeypatches)
    return steps


def _config(**settings) -> InvokeAIAppConfig:
    """A config whose runtime settings each differ from their defaults."""
    return InvokeAIAppConfig(
        rocm_aotriton_experimental="off", generation_devices=["cuda:1"], attention_backend="sage", **settings
    )


@pytest.mark.parametrize(
    "alloc_conf, allocator_step",
    [
        (None, ("the platform's allocator default",)),
        ("backend:cudaMallocAsync", ("configure the allocator", "backend:cudaMallocAsync")),
    ],
    ids=["platform-default", "configured"],
)
def test_the_server_runtime_applies_its_steps_in_the_servers_order(
    runtime_steps: list[tuple], monkeypatch: pytest.MonkeyPatch, alloc_conf, allocator_step
):
    import invokeai.app.run_app as run_app

    monkeypatch.delenv("MIOPEN_FIND_MODE", raising=False)

    run_app.apply_server_runtime(_config(pytorch_cuda_alloc_conf=alloc_conf), logging.getLogger(__name__))

    assert runtime_steps == [
        ("hide integrated GPUs", "FAST"),
        allocator_step,
        ("AOTriton", "off", ["cuda:1"]),
        ("monkeypatches", "sage"),
    ]


class _Started(Exception):
    """Ends a server start once it is as far as a test needs."""


def test_the_server_starts_with_the_runtime_it_gives_the_calibration_scripts(
    runtime_steps: list[tuple], monkeypatch: pytest.MonkeyPatch
):
    import invokeai.app.run_app as run_app
    from invokeai.app.services.config import config_default
    from invokeai.app.util import startup_utils
    from invokeai.backend.util.devices import TorchDevice
    from invokeai.frontend.cli.arg_parser import InvokeAIArgs

    config = _config(pytorch_cuda_alloc_conf="backend:cudaMallocAsync")

    def started() -> None:
        raise _Started

    monkeypatch.setattr(InvokeAIArgs, "parse_args", staticmethod(lambda: None))  # pytest's arguments are not the app's
    monkeypatch.setattr(config_default, "get_config", lambda: config)
    # The server's own steps between the runtime's two parts, without asking a GPU or the network.
    monkeypatch.setattr(TorchDevice, "get_generation_devices_summary", classmethod(lambda cls, devices: "a GPU"))
    monkeypatch.setattr(startup_utils, "find_open_port", lambda port: port)
    # The startup step after the runtime: the server is as far as this test needs. Should that step move, the app still
    # does not come up.
    monkeypatch.setattr(startup_utils, "register_mime_types", started)
    monkeypatch.setattr(run_app, "get_app", started)

    with pytest.raises(_Started):
        run_app.run_app()
    server = list(runtime_steps)
    runtime_steps.clear()
    run_app.apply_server_runtime(config, logging.getLogger(__name__))

    assert server == runtime_steps
