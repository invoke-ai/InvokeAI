import subprocess
import sys

from invokeai.app.services.shared import graph as graph_module
from invokeai.app.services.shared import graph_scheduler


def test_graph_facade_reexports_scheduler_classes() -> None:
    assert graph_module._ExecutionScheduler is graph_scheduler._ExecutionScheduler
    assert graph_module._GenericGraphSchedulerAdapter is graph_scheduler._GenericGraphSchedulerAdapter


def test_graph_scheduler_import_does_not_import_graph_facade() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import invokeai.app.services.shared.graph_scheduler; "
            "print('invokeai.app.services.shared.graph' in sys.modules)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "False"
