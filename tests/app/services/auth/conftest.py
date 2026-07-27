import pytest

from invokeai.app.services.auth.token_service import set_jwt_secret


@pytest.fixture(autouse=True)
def setup_jwt_secret() -> None:
    set_jwt_secret("test-secret-key-for-unit-tests-only-do-not-use-in-production")
