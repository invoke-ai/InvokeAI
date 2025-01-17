import os

import pytest
import torch

IS_GITHUB_ACTION = os.environ.get("GITHUB_ACTION") == "true"
HAS_MPS_DEVICE = torch.backends.mps.is_available()

# Some tests that use MPS are flaky on Github Actions.
# Specifically, they fail with `MPS backend out of memory` even though there is plenty of memory available.
# I haven't taken the time to get to the bottom of this yet. The tests pass locally on MPS.
# There are several reports of similar issues
# (e.g. https://discuss.pytorch.org/t/mps-back-end-out-of-memory-on-github-action/189773).
mark_flaky_mps_github_action_test = pytest.mark.xfail(
    condition=IS_GITHUB_ACTION and HAS_MPS_DEVICE, reason="This test is flaky on GitHub Actions with MPS.", strict=False
)
