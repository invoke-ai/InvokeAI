import torch
from PIL import Image

from invokeai.app.invocations.spandrel_image_to_image import SpandrelImageToImageInvocation
from invokeai.backend.util.devices import TorchDevice


def test_upscale_image_does_not_crop_full_image_for_untiled_input(monkeypatch):
    image = Image.new("RGB", (8, 8))

    class IdentityModel:
        scale = 1
        dtype = torch.float32

        @staticmethod
        def run(image_tensor: torch.Tensor) -> torch.Tensor:
            return image_tensor

    def fail_crop(_image: Image.Image, _box: tuple[int, int, int, int]) -> Image.Image:
        raise AssertionError("untiled input must not be copied with a full-image crop")

    monkeypatch.setattr(Image.Image, "crop", fail_crop)
    monkeypatch.setattr(TorchDevice, "choose_torch_device", staticmethod(lambda: torch.device("cpu")))

    result = SpandrelImageToImageInvocation.upscale_image(
        image,
        tile_size=0,
        spandrel_model=IdentityModel(),
        is_canceled=lambda: False,
        step_callback=lambda *_: None,
    )

    assert result.size == image.size
    assert result.mode == image.mode
    assert result.tobytes() == image.tobytes()
