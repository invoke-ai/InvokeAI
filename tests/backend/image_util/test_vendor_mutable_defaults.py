"""Tests for the mutable default argument fix in imwatermark/vendor.py."""

from invokeai.backend.image_util.imwatermark.vendor import EmbedMaxDct, WatermarkEncoder


class TestSetByBitsNoSharedState:
    """set_by_bits() used to have bits=[] as a default arg.
    If it were still mutable, successive calls without an explicit arg
    would accumulate state. After the fix (bits=None), each call gets
    a fresh list."""

    def test_set_by_bits_default_is_independent(self):
        enc1 = WatermarkEncoder()
        enc1.set_by_bits()
        assert enc1._watermarks == []
        assert enc1._wmLen == 0

        enc2 = WatermarkEncoder()
        enc2.set_by_bits()
        assert enc2._watermarks == []
        assert enc2._wmLen == 0

    def test_set_by_bits_with_explicit_arg(self):
        enc = WatermarkEncoder()
        enc.set_by_bits([1, 0, 1])
        assert enc._watermarks == [1, 0, 1]
        assert enc._wmLen == 3
        assert enc._wmType == "bits"


class TestEmbedMaxDctNoSharedState:
    """EmbedMaxDct.__init__ used to have watermarks=[] and scales=[0,36,36].
    After the fix (both default to None), each instance gets its own list."""

    def test_default_watermarks_independent(self):
        e1 = EmbedMaxDct()
        e1._watermarks.append(999)

        e2 = EmbedMaxDct()
        assert 999 not in e2._watermarks
        assert e2._watermarks == []

    def test_default_scales_independent(self):
        e1 = EmbedMaxDct()
        e1._scales.append(72)

        e2 = EmbedMaxDct()
        assert e2._scales == [0, 36, 36]

    def test_explicit_args_still_work(self):
        wm = [1, 0, 1, 1]
        sc = [0, 50, 50]
        e = EmbedMaxDct(watermarks=wm, wmLen=4, scales=sc, block=8)
        assert e._watermarks == wm
        assert e._wmLen == 4
        assert e._scales == sc
        assert e._block == 8
