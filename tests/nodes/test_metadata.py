import json
import os

from copy import deepcopy
from PIL import Image, PngImagePlugin

from invokeai.app.invocations.generate import TextToImageInvocation
from invokeai.app.modules.metadata import InvokeAIMetadata, MetadataModule

good_metadata_dict = {
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

bad_metadata_dict_missing_session_id = deepcopy(good_metadata_dict)
bad_metadata_dict_missing_session_id["session_id"] = None

bad_metadata_dict_invalid_session_id = deepcopy(good_metadata_dict)
bad_metadata_dict_invalid_session_id["session_id"] = 123

bad_metadata_dict_missing_node = deepcopy(good_metadata_dict)
bad_metadata_dict_missing_node["node"] = None

bad_metadata_dict_invalid_node = deepcopy(good_metadata_dict)
bad_metadata_dict_invalid_node["node"] = 123

bad_metadata_dict_missing_node_id = deepcopy(good_metadata_dict)
del bad_metadata_dict_missing_node_id["node"]["id"]

bad_metadata_dict_invalid_node_id = deepcopy(good_metadata_dict)
bad_metadata_dict_invalid_node_id["node"]["id"] = 123

bad_metadata_dict_missing_node_type = deepcopy(good_metadata_dict)
del bad_metadata_dict_missing_node_type["node"]["type"]

bad_metadata_dict_invalid_node_type = deepcopy(good_metadata_dict)
bad_metadata_dict_invalid_node_type["node"]["type"] = 123

bad_metadata_dict_no_node_attrs = deepcopy(good_metadata_dict)
bad_metadata_dict_no_node_attrs["node"] = {}

bad_metadata_dict_array_attr = deepcopy(good_metadata_dict)
bad_metadata_dict_array_attr["node"]["seed"] = [1, 2, 3]

bad_metadata_dict_invalid_dict_attr = deepcopy(good_metadata_dict)
bad_metadata_dict_invalid_dict_attr["node"]["seed"] = {"a": 1}

bad_metadata_dict_missing_image_field_image_type = deepcopy(good_metadata_dict)
del bad_metadata_dict_missing_image_field_image_type["node"]["image"]["image_type"]

bad_metadata_dict_invalid_image_field_image_type = deepcopy(good_metadata_dict)
bad_metadata_dict_invalid_image_field_image_type["node"]["image"][
    "image_type"
] = "bad image type"

bad_metadata_dict_invalid_latents_field_latents_name = deepcopy(good_metadata_dict)
bad_metadata_dict_invalid_latents_field_latents_name["node"]["latents"] = {
    "latents_name": 123
}


def test_is_good_metadata_unchanged():
    parsed_metadata = MetadataModule._parse_invokeai_metadata(good_metadata_dict)
    assert good_metadata_dict == parsed_metadata.dict()


def test_bad_metadata_dict_missing_session_id():
    parsed_metadata = MetadataModule._parse_invokeai_metadata(
        bad_metadata_dict_missing_session_id
    )
    assert bad_metadata_dict_missing_session_id == parsed_metadata.dict()


def test_bad_metadata_dict_invalid_session_id():
    parsed_metadata = MetadataModule._parse_invokeai_metadata(
        bad_metadata_dict_invalid_session_id
    )
    assert bad_metadata_dict_missing_session_id == parsed_metadata.dict()


def test_bad_metadata_dict_missing_node():
    parsed_metadata = MetadataModule._parse_invokeai_metadata(
        bad_metadata_dict_missing_node
    )
    assert bad_metadata_dict_missing_node == parsed_metadata.dict()


def test_bad_metadata_dict_invalid_node():
    parsed_metadata = MetadataModule._parse_invokeai_metadata(
        bad_metadata_dict_invalid_node
    )
    assert bad_metadata_dict_missing_node == parsed_metadata.dict()


def test_bad_metadata_dict_missing_node_id():
    parsed_metadata = MetadataModule._parse_invokeai_metadata(
        bad_metadata_dict_missing_node_id
    )
    assert bad_metadata_dict_missing_node_id == parsed_metadata.dict()


def test_bad_metadata_dict_invalid_node_id():
    parsed_metadata = MetadataModule._parse_invokeai_metadata(
        bad_metadata_dict_invalid_node_id
    )
    assert bad_metadata_dict_missing_node_id == parsed_metadata.dict()


def test_bad_metadata_dict_missing_node_type():
    parsed_metadata = MetadataModule._parse_invokeai_metadata(
        bad_metadata_dict_missing_node_type
    )
    assert bad_metadata_dict_missing_node_type == parsed_metadata.dict()


def test_bad_metadata_dict_invalid_node_type():
    parsed_metadata = MetadataModule._parse_invokeai_metadata(
        bad_metadata_dict_invalid_node_type
    )
    assert bad_metadata_dict_missing_node_type == parsed_metadata.dict()


def test_bad_metadata_dict_no_node_attrs():
    parsed_metadata = MetadataModule._parse_invokeai_metadata(
        bad_metadata_dict_no_node_attrs
    )
    assert bad_metadata_dict_missing_node == parsed_metadata.dict()


def test_bad_metadata_dict_array_attr():
    expected = deepcopy(good_metadata_dict)
    del expected["node"]["seed"]

    parsed_metadata = MetadataModule._parse_invokeai_metadata(
        bad_metadata_dict_array_attr
    )
    assert expected == parsed_metadata.dict()


def test_bad_metadata_dict_invalid_dict_attr():
    expected = deepcopy(good_metadata_dict)
    del expected["node"]["seed"]

    parsed_metadata = MetadataModule._parse_invokeai_metadata(
        bad_metadata_dict_invalid_dict_attr
    )
    assert expected == parsed_metadata.dict()


def test_bad_metadata_dict_missing_image_field_image_type():
    expected = deepcopy(good_metadata_dict)
    del expected["node"]["image"]

    parsed_metadata = MetadataModule._parse_invokeai_metadata(
        bad_metadata_dict_missing_image_field_image_type
    )
    assert expected == parsed_metadata.dict()


def test_bad_metadata_dict_invalid_image_field_image_type():
    expected = deepcopy(good_metadata_dict)
    del expected["node"]["image"]

    parsed_metadata = MetadataModule._parse_invokeai_metadata(
        bad_metadata_dict_invalid_image_field_image_type
    )
    assert expected == parsed_metadata.dict()


def test_bad_metadata_dict_invalid_latents_field_latents_name():
    parsed_metadata = MetadataModule._parse_invokeai_metadata(
        bad_metadata_dict_invalid_latents_field_latents_name
    )
    assert good_metadata_dict == parsed_metadata.dict()


def test_can_load_and_parse_invokeai_metadata(tmp_path):
    raw_metadata = {"session_id": "123", "node": {"id": "456", "type": "test_type"}}

    temp_image = Image.new("RGB", (512, 512))
    temp_image_path = os.path.join(tmp_path, "test.png")

    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("invokeai", json.dumps(raw_metadata))

    temp_image.save(temp_image_path, pnginfo=pnginfo)

    image = Image.open(temp_image_path)

    loaded_metadata = MetadataModule._load_metadata(image)
    parsed_metadata = MetadataModule._parse_invokeai_metadata(loaded_metadata)
    loaded_and_parsed_metadata = MetadataModule.get_metadata(image)

    assert raw_metadata == loaded_metadata
    assert raw_metadata == parsed_metadata.dict()
    assert raw_metadata == loaded_and_parsed_metadata.dict()


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

    metadata = MetadataModule.build_metadata(
        session_id=session_id, invocation=invocation
    )

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

    assert type(metadata) is InvokeAIMetadata
    assert expected_metadata_dict == metadata.dict()
