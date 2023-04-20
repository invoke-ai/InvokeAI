import json
import os

from copy import deepcopy
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
        "image": {"image_type": "results", "image_name": "1"},
        "cfg_scale": 7.5,
        "scheduler": "k_lms",
        "seamless": False,
        "model": "stable-diffusion-1.5",
        "progress_images": True,
    },
}

metadata_service = PngMetadataService()


def test_is_good_metadata_unchanged():
    parsed_metadata = metadata_service._parse_invokeai_metadata(valid_metadata)

    expected = deepcopy(valid_metadata)

    assert expected == parsed_metadata


def test_can_parse_missing_session_id():
    metadata_missing_session_id = deepcopy(valid_metadata)
    del metadata_missing_session_id["session_id"]

    expected = deepcopy(valid_metadata)
    del expected["session_id"]

    parsed_metadata = metadata_service._parse_invokeai_metadata(
        metadata_missing_session_id
    )
    assert metadata_missing_session_id == parsed_metadata


def test_can_parse_invalid_session_id():
    metadata_invalid_session_id = deepcopy(valid_metadata)
    metadata_invalid_session_id["session_id"] = 123

    expected = deepcopy(valid_metadata)
    del expected["session_id"]

    parsed_metadata = metadata_service._parse_invokeai_metadata(
        metadata_invalid_session_id
    )
    assert expected == parsed_metadata


def test_can_parse_missing_node():
    metadata_missing_node = deepcopy(valid_metadata)
    del metadata_missing_node["node"]

    expected = deepcopy(valid_metadata)
    del expected["node"]

    parsed_metadata = metadata_service._parse_invokeai_metadata(metadata_missing_node)
    assert expected == parsed_metadata


def test_can_parse_invalid_node():
    metadata_invalid_node = deepcopy(valid_metadata)
    metadata_invalid_node["node"] = 123

    expected = deepcopy(valid_metadata)
    del expected["node"]

    parsed_metadata = metadata_service._parse_invokeai_metadata(metadata_invalid_node)
    assert expected == parsed_metadata


def test_can_parse_missing_node_id():
    metadata_missing_node_id = deepcopy(valid_metadata)
    del metadata_missing_node_id["node"]["id"]

    expected = deepcopy(valid_metadata)
    del expected["node"]["id"]

    parsed_metadata = metadata_service._parse_invokeai_metadata(
        metadata_missing_node_id
    )
    assert expected == parsed_metadata


def test_can_parse_invalid_node_id():
    metadata_invalid_node_id = deepcopy(valid_metadata)
    metadata_invalid_node_id["node"]["id"] = 123

    expected = deepcopy(valid_metadata)
    del expected["node"]["id"]

    parsed_metadata = metadata_service._parse_invokeai_metadata(
        metadata_invalid_node_id
    )
    assert expected == parsed_metadata


def test_can_parse_missing_node_type():
    metadata_missing_node_type = deepcopy(valid_metadata)
    del metadata_missing_node_type["node"]["type"]

    expected = deepcopy(valid_metadata)
    del expected["node"]["type"]

    parsed_metadata = metadata_service._parse_invokeai_metadata(
        metadata_missing_node_type
    )
    assert expected == parsed_metadata


def test_can_parse_invalid_node_type():
    metadata_invalid_node_type = deepcopy(valid_metadata)
    metadata_invalid_node_type["node"]["type"] = 123

    expected = deepcopy(valid_metadata)
    del expected["node"]["type"]

    parsed_metadata = metadata_service._parse_invokeai_metadata(
        metadata_invalid_node_type
    )
    assert expected == parsed_metadata


def test_can_parse_no_node_attrs():
    metadata_no_node_attrs = deepcopy(valid_metadata)
    metadata_no_node_attrs["node"] = {}

    expected = deepcopy(valid_metadata)
    del expected["node"]

    parsed_metadata = metadata_service._parse_invokeai_metadata(metadata_no_node_attrs)
    assert expected == parsed_metadata


def test_can_parse_array_attr():
    metadata_array_attr = deepcopy(valid_metadata)
    metadata_array_attr["node"]["seed"] = [1, 2, 3]

    expected = deepcopy(valid_metadata)
    del expected["node"]["seed"]

    parsed_metadata = metadata_service._parse_invokeai_metadata(metadata_array_attr)
    assert expected == parsed_metadata


def test_can_parse_invalid_dict_attr():
    metadata_invalid_dict_attr = deepcopy(valid_metadata)
    metadata_invalid_dict_attr["node"]["seed"] = {"a": 1}

    expected = deepcopy(valid_metadata)
    del expected["node"]["seed"]

    parsed_metadata = metadata_service._parse_invokeai_metadata(
        metadata_invalid_dict_attr
    )
    assert expected == parsed_metadata


def test_can_parse_missing_image_field_image_type():
    metadata_missing_image_field_image_type = deepcopy(valid_metadata)
    del metadata_missing_image_field_image_type["node"]["image"]["image_type"]

    expected = deepcopy(valid_metadata)
    del expected["node"]["image"]

    parsed_metadata = metadata_service._parse_invokeai_metadata(
        metadata_missing_image_field_image_type
    )
    assert expected == parsed_metadata


def test_can_parse_invalid_image_field_image_type():
    metadata_invalid_image_field_image_type = deepcopy(valid_metadata)
    metadata_invalid_image_field_image_type["node"]["image"][
        "image_type"
    ] = "bad image type"

    expected = deepcopy(valid_metadata)
    del expected["node"]["image"]

    parsed_metadata = metadata_service._parse_invokeai_metadata(
        metadata_invalid_image_field_image_type
    )
    assert expected == parsed_metadata


def test_can_parse_invalid_latents_field_latents_name():
    metadata_invalid_latents_field_latents_name = deepcopy(valid_metadata)
    metadata_invalid_latents_field_latents_name["node"]["latents"] = {
        "latents_name": 123
    }

    expected = deepcopy(valid_metadata)

    parsed_metadata = metadata_service._parse_invokeai_metadata(
        metadata_invalid_latents_field_latents_name
    )

    assert expected == parsed_metadata


def test_can_load_and_parse_invokeai_metadata(tmp_path):
    raw_metadata = {"session_id": "123", "node": {"id": "456", "type": "test_type"}}

    temp_image = Image.new("RGB", (512, 512))
    temp_image_path = os.path.join(tmp_path, "test.png")

    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("invokeai", json.dumps(raw_metadata))

    temp_image.save(temp_image_path, pnginfo=pnginfo)

    image = Image.open(temp_image_path)

    loaded_metadata = metadata_service._load_metadata(image)
    parsed_metadata = metadata_service._parse_invokeai_metadata(loaded_metadata)
    loaded_and_parsed_metadata = metadata_service.get_metadata(image)

    assert raw_metadata == loaded_metadata
    assert raw_metadata == parsed_metadata
    assert raw_metadata == loaded_and_parsed_metadata


def test_can_build_invokeai_metadata():
    session_id = "123"
    invocation = TextToImageInvocation(
        id="456",
        prompt="test",
        seed=1,
        steps=10,
        width=512,
        height=512,
        cfg_scale=7.5,
        scheduler="k_lms",
        seamless=False,
        model="test_mode",
        progress_images=True,
    )

    metadata = metadata_service.build_metadata(session_id=session_id, node=invocation)

    expected_metadata_dict = {
        "session_id": "123",
        "node": {
            "id": "456",
            "type": "txt2img",
            "prompt": "test",
            "seed": 1,
            "steps": 10,
            "width": 512,
            "height": 512,
            "cfg_scale": 7.5,
            "scheduler": "k_lms",
            "seamless": False,
            "model": "test_mode",
            "progress_images": True,
        },
    }

    assert expected_metadata_dict == metadata
