# conftest.py is a special pytest file. Fixtures defined in this file will be accessible to all tests in this directory
# without needing to explicitly import them. (https://docs.pytest.org/en/6.2.x/fixture.html)


# We import the model_installer and torch_device fixtures here so that they can be used by all tests. Flake8 does not
# play well with fixtures (F401 and F811), so this is cleaner than importing in all files that use these fixtures.
from invokeai.backend.util.test_utils import torch_device  # noqa: F401
from tests.fixtures.event_service import mock_event_service  # noqa: F401
