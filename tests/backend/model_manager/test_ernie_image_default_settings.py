from invokeai.backend.model_manager.configs.main import MainModelDefaultSettings
from invokeai.backend.model_manager.taxonomy import BaseModelType


class TestErnieImageDefaultSettings:
    def test_base_defaults(self) -> None:
        s = MainModelDefaultSettings.from_base(BaseModelType.ErnieImage, None, "ERNIE-Image")
        assert s is not None
        assert (s.steps, s.cfg_scale) == (50, 4.0)

    def test_turbo_detected_from_name(self) -> None:
        s = MainModelDefaultSettings.from_base(BaseModelType.ErnieImage, None, "ERNIE-Image-Turbo")
        assert s is not None
        assert (s.steps, s.cfg_scale) == (8, 1.0)

    def test_turbo_detected_from_path_when_renamed(self) -> None:
        # The two checkpoints are architecturally identical, so detection falls back to the name.
        # Renaming the model in the install dialog must not lose the Turbo defaults, hence the
        # install path is consulted too.
        s = MainModelDefaultSettings.from_base(
            BaseModelType.ErnieImage, None, "my ernie", "main/ernie-image/ERNIE-Image-Turbo"
        )
        assert s is not None
        assert (s.steps, s.cfg_scale) == (8, 1.0)

    def test_no_turbo_hint_falls_back_to_base_defaults(self) -> None:
        s = MainModelDefaultSettings.from_base(BaseModelType.ErnieImage, None, "my ernie", "main/ernie-image/whatever")
        assert s is not None
        assert (s.steps, s.cfg_scale) == (50, 4.0)
