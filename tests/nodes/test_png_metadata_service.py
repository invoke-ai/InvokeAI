import json
import os

from PIL import Image, PngImagePlugin

from invokeai.app.invocations.generate import TextToImageInvocation
from invokeai.app.services.metadata import PngMetadataService

valid_metadata = {
    "session_id": "1",
    "node": {
        "id": "1",
        "type": "txt2img",
        "prompt": "dog",
        "seed": 178785523,
        "steps": 30,
        "width": 512,
        "height": 512,
        "cfg_scale": 7.5,
        "scheduler": "lms",
        "model": "stable-diffusion-1.5",
    },
}

metadata_service = PngMetadataService()


def test_can_load_and_parse_invokeai_metadata(tmp_path):
    raw_metadata = {"session_id": "123", "node": {"id": "456", "type": "test_type"}}

    temp_image = Image.new("RGB", (512, 512))
    temp_image_path = os.path.join(tmp_path, "test.png")

    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("invokeai", json.dumps(raw_metadata))

    temp_image.save(temp_image_path, pnginfo=pnginfo)

    image = Image.open(temp_image_path)

    loaded_metadata = metadata_service.get_metadata(image)

    assert loaded_metadata is not None
    assert raw_metadata == loaded_metadata


def test_can_build_invokeai_metadata():
    session_id = valid_metadata["session_id"]
    node = TextToImageInvocation(**valid_metadata["node"])

    metadata = metadata_service.build_metadata(session_id=session_id, node=node)

    assert valid_metadata == metadata
