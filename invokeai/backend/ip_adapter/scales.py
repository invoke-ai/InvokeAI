class Scales:
    """The IP-Adapter scales for a patched UNet. This object can be used to dynamically change the scales for a patched
    UNet.
    """

    def __init__(self, scales: list[float]):
        self._scales = scales

    @property
    def scales(self):
        return self._scales

    @scales.setter
    def scales(self, scales: list[float]):
        assert len(scales) == len(self._scales)
        self._scales = scales

    def __len__(self):
        return len(self._scales)
