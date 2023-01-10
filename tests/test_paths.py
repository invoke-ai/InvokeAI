import pytest
from ldm.invoke.config.paths import InvokePaths, DEFAULT_RUNTIME_DIR
from pathlib import Path

@pytest.fixture(autouse=True)
def reset_path_singleton():
    InvokePaths._instances = {}

class TestInvokePathsSingleton:
    """
    Test correct singleton behaviour
    """

    def test_identity(self):
        p1 = InvokePaths()
        p2 = InvokePaths()

        assert p1 is p2

    def test_override_envvar_transfers(self, monkeypatch):
        monkeypatch.delenv("VIRTUAL_ENV", raising = False)

        monkeypatch.setenv("INVOKEAI_ROOT", "/someplace")
        p1 = InvokePaths()
        p2 = InvokePaths()

        assert p1.root.location == Path("/someplace")

        p2.root = "~/some-other-place"

        assert p1.root.location == Path("~/some-other-place")
        assert p1.root.location is p2.root.location

class TestInvokePathsRootDir:

    def test_fallback(self, monkeypatch):
        """
        When root dir isn't set and no env vars are set, it should fall back to default
        """

        monkeypatch.delenv("VIRTUAL_ENV", raising = False)
        monkeypatch.delenv("INVOKEAI_ROOT", raising = False)
        p = InvokePaths()
        assert p.root.location == Path(DEFAULT_RUNTIME_DIR)

    def test_root_envvar(self, monkeypatch):
        """
        When INVOKEAI_ROOT env var is set, it should be used
        """

        monkeypatch.delenv("VIRTUAL_ENV", raising = False)

        for location in [
            "/absolute/path/invokeai",
            "relative/path/invokeai",
            "~/home/path/invokeai",
        ]:
            monkeypatch.setenv("INVOKEAI_ROOT", location)
            p = InvokePaths()
            assert p.root.location == Path(location)


    def test_virtualenv_envvar(self, monkeypatch):
        """
        When VIRTUAL_ENV env var is set, it should be used
        """

        monkeypatch.delenv("INVOKEAI_ROOT", raising = False)
        monkeypatch.setenv("VIRTUAL_ENV", f"{DEFAULT_RUNTIME_DIR}/.venv")
        p = InvokePaths()
        assert p.root.location == Path(DEFAULT_RUNTIME_DIR)

    def test_both_envvars(self, monkeypatch):
        """
        When both envvars are set, the INVOKEAI_ROOT env var should take precedence
        """

        monkeypatch.setenv("INVOKEAI_ROOT", "/someplace")
        monkeypatch.setenv("VIRTUAL_ENV", f"{DEFAULT_RUNTIME_DIR}/.venv")
        p = InvokePaths()
        assert p.root.location == Path("/someplace")

# class TestInvokePathsOutputsDir:

    # def test_changing_root_dir_default_outdir():

    # def test_changing_root_dir_custom_outdir():


    # def test_fallback(self, monkeypatch):
    #     """
    #     When root dir isn't set and no env vars are set, it should fall back to default
    #     """

    #     monkeypatch.delenv("VIRTUAL_ENV", raising = False)
    #     monkeypatch.delenv("INVOKEAI_ROOT", raising = False)
    #     p = InvokePaths()
    #     assert p.root.location == Path(DEFAULT_RUNTIME_DIR)

    # def test_root_envvar(self, monkeypatch):
    #     """
    #     When INVOKEAI_ROOT env var is set, it should be used
    #     """

    #     monkeypatch.delenv("VIRTUAL_ENV", raising = False)

    #     for location in [
    #         "/absolute/path/invokeai",
    #         "relative/path/invokeai",
    #         "~/home/path/invokeai",
    #     ]:
    #         monkeypatch.setenv("INVOKEAI_ROOT", location)
    #         p = InvokePaths()
    #         assert p.root.location == Path(location)


    # def test_virtualenv_envvar(self, monkeypatch):
    #     """
    #     When VIRTUAL_ENV env var is set, it should be used
    #     """

    #     monkeypatch.delenv("INVOKEAI_ROOT", raising = False)
    #     monkeypatch.setenv("VIRTUAL_ENV", f"{DEFAULT_RUNTIME_DIR}/.venv")
    #     p = InvokePaths()
    #     assert p.root.location == Path(DEFAULT_RUNTIME_DIR)

    # def test_both_envvars(self, monkeypatch):
    #     """
    #     When both envvars are set, the INVOKEAI_ROOT env var should take precedence
    #     """

    #     monkeypatch.setenv("INVOKEAI_ROOT", "/someplace")
    #     monkeypatch.setenv("VIRTUAL_ENV", f"{DEFAULT_RUNTIME_DIR}/.venv")
    #     p = InvokePaths()
    #     assert p.root.location == Path("/someplace")