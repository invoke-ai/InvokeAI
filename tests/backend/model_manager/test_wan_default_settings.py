"""Tests for Wan 2.2 default settings."""

from invokeai.backend.model_manager.configs.main import MainModelDefaultSettings
from invokeai.backend.model_manager.taxonomy import BaseModelType, WanVariantType


class TestWanDefaultSettings:
    def test_a14b_defaults(self) -> None:
        s = MainModelDefaultSettings.from_base(BaseModelType.Wan, WanVariantType.T2V_A14B)
        assert s is not None
        assert s.steps == 40
        assert s.cfg_scale == 4.0
        assert s.width == 1024
        assert s.height == 1024

    def test_ti2v_5b_defaults(self) -> None:
        s = MainModelDefaultSettings.from_base(BaseModelType.Wan, WanVariantType.TI2V_5B)
        assert s is not None
        assert s.steps == 30
        assert s.cfg_scale == 5.0

    def test_no_variant_falls_back_to_a14b_settings(self) -> None:
        s = MainModelDefaultSettings.from_base(BaseModelType.Wan)
        assert s is not None
        assert s.steps == 40
