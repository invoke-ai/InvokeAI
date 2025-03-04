import os
from contextlib import contextmanager


@contextmanager
def unset_env_var(env_var: str):
    """Context manager that unsets an environment variable, re-setting it to the original value when the context is exited."""
    prev_value = os.environ.get(env_var, None)
    if prev_value is not None:
        del os.environ[env_var]
    yield
    if prev_value is not None:
        os.environ[env_var] = prev_value


@contextmanager
def set_env_var(env_var: str, value: str):
    """Context manager that sets an environment variable, re-setting it to the original value when the context is exited."""
    prev_value = os.environ.get(env_var, None)
    os.environ[env_var] = value
    yield
    if prev_value is not None:
        os.environ[env_var] = prev_value
    else:
        del os.environ[env_var]
