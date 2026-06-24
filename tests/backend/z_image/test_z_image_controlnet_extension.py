from types import SimpleNamespace
from unittest.mock import patch

import torch

from invokeai.backend.z_image.z_image_controlnet_extension import ZImageControlNetExtension


def test_init_logs_control_adapter_diagnostics_without_stdout(capsys):
    control_adapter = SimpleNamespace(
        control_layers=[
            SimpleNamespace(
                after_proj=SimpleNamespace(weight=torch.zeros(1)),
            )
        ]
    )

    with patch("invokeai.backend.z_image.z_image_controlnet_extension.logger") as logger_mock:
        ZImageControlNetExtension(control_adapter=control_adapter, control_cond=torch.zeros(1))

    captured = capsys.readouterr()
    assert captured.out == ""
    assert logger_mock.debug.call_count == 3
    logger_mock.warning.assert_called_once()
