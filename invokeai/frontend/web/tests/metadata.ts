export default {};

// python metadata parsing tests to rebuild

// # def test_is_good_metadata_unchanged():
// #     parsed_metadata = metadata_service._parse_invokeai_metadata(valid_metadata)

// #     expected = deepcopy(valid_metadata)

// #     assert expected == parsed_metadata

// # def test_can_parse_missing_session_id():
// #     metadata_missing_session_id = deepcopy(valid_metadata)
// #     del metadata_missing_session_id["session_id"]

// #     expected = deepcopy(valid_metadata)
// #     del expected["session_id"]

// #     parsed_metadata = metadata_service._parse_invokeai_metadata(
// #         metadata_missing_session_id
// #     )
// #     assert metadata_missing_session_id == parsed_metadata

// # def test_can_parse_invalid_session_id():
// #     metadata_invalid_session_id = deepcopy(valid_metadata)
// #     metadata_invalid_session_id["session_id"] = 123

// #     expected = deepcopy(valid_metadata)
// #     del expected["session_id"]

// #     parsed_metadata = metadata_service._parse_invokeai_metadata(
// #         metadata_invalid_session_id
// #     )
// #     assert expected == parsed_metadata

// # def test_can_parse_missing_node():
// #     metadata_missing_node = deepcopy(valid_metadata)
// #     del metadata_missing_node["node"]

// #     expected = deepcopy(valid_metadata)
// #     del expected["node"]

// #     parsed_metadata = metadata_service._parse_invokeai_metadata(metadata_missing_node)
// #     assert expected == parsed_metadata

// # def test_can_parse_invalid_node():
// #     metadata_invalid_node = deepcopy(valid_metadata)
// #     metadata_invalid_node["node"] = 123

// #     expected = deepcopy(valid_metadata)
// #     del expected["node"]

// #     parsed_metadata = metadata_service._parse_invokeai_metadata(metadata_invalid_node)
// #     assert expected == parsed_metadata

// # def test_can_parse_missing_node_id():
// #     metadata_missing_node_id = deepcopy(valid_metadata)
// #     del metadata_missing_node_id["node"]["id"]

// #     expected = deepcopy(valid_metadata)
// #     del expected["node"]["id"]

// #     parsed_metadata = metadata_service._parse_invokeai_metadata(
// #         metadata_missing_node_id
// #     )
// #     assert expected == parsed_metadata

// # def test_can_parse_invalid_node_id():
// #     metadata_invalid_node_id = deepcopy(valid_metadata)
// #     metadata_invalid_node_id["node"]["id"] = 123

// #     expected = deepcopy(valid_metadata)
// #     del expected["node"]["id"]

// #     parsed_metadata = metadata_service._parse_invokeai_metadata(
// #         metadata_invalid_node_id
// #     )
// #     assert expected == parsed_metadata

// # def test_can_parse_missing_node_type():
// #     metadata_missing_node_type = deepcopy(valid_metadata)
// #     del metadata_missing_node_type["node"]["type"]

// #     expected = deepcopy(valid_metadata)
// #     del expected["node"]["type"]

// #     parsed_metadata = metadata_service._parse_invokeai_metadata(
// #         metadata_missing_node_type
// #     )
// #     assert expected == parsed_metadata

// # def test_can_parse_invalid_node_type():
// #     metadata_invalid_node_type = deepcopy(valid_metadata)
// #     metadata_invalid_node_type["node"]["type"] = 123

// #     expected = deepcopy(valid_metadata)
// #     del expected["node"]["type"]

// #     parsed_metadata = metadata_service._parse_invokeai_metadata(
// #         metadata_invalid_node_type
// #     )
// #     assert expected == parsed_metadata

// # def test_can_parse_no_node_attrs():
// #     metadata_no_node_attrs = deepcopy(valid_metadata)
// #     metadata_no_node_attrs["node"] = {}

// #     expected = deepcopy(valid_metadata)
// #     del expected["node"]

// #     parsed_metadata = metadata_service._parse_invokeai_metadata(metadata_no_node_attrs)
// #     assert expected == parsed_metadata

// # def test_can_parse_array_attr():
// #     metadata_array_attr = deepcopy(valid_metadata)
// #     metadata_array_attr["node"]["seed"] = [1, 2, 3]

// #     expected = deepcopy(valid_metadata)
// #     del expected["node"]["seed"]

// #     parsed_metadata = metadata_service._parse_invokeai_metadata(metadata_array_attr)
// #     assert expected == parsed_metadata

// # def test_can_parse_invalid_dict_attr():
// #     metadata_invalid_dict_attr = deepcopy(valid_metadata)
// #     metadata_invalid_dict_attr["node"]["seed"] = {"a": 1}

// #     expected = deepcopy(valid_metadata)
// #     del expected["node"]["seed"]

// #     parsed_metadata = metadata_service._parse_invokeai_metadata(
// #         metadata_invalid_dict_attr
// #     )
// #     assert expected == parsed_metadata

// # def test_can_parse_missing_image_field_image_type():
// #     metadata_missing_image_field_image_type = deepcopy(valid_metadata)
// #     del metadata_missing_image_field_image_type["node"]["image"]["image_type"]

// #     expected = deepcopy(valid_metadata)
// #     del expected["node"]["image"]

// #     parsed_metadata = metadata_service._parse_invokeai_metadata(
// #         metadata_missing_image_field_image_type
// #     )
// #     assert expected == parsed_metadata

// # def test_can_parse_invalid_image_field_image_type():
// #     metadata_invalid_image_field_image_type = deepcopy(valid_metadata)
// #     metadata_invalid_image_field_image_type["node"]["image"][
// #         "image_type"
// #     ] = "bad image type"

// #     expected = deepcopy(valid_metadata)
// #     del expected["node"]["image"]

// #     parsed_metadata = metadata_service._parse_invokeai_metadata(
// #         metadata_invalid_image_field_image_type
// #     )
// #     assert expected == parsed_metadata

// # def test_can_parse_invalid_latents_field_latents_name():
// #     metadata_invalid_latents_field_latents_name = deepcopy(valid_metadata)
// #     metadata_invalid_latents_field_latents_name["node"]["latents"] = {
// #         "latents_name": 123
// #     }

// #     expected = deepcopy(valid_metadata)

// #     parsed_metadata = metadata_service._parse_invokeai_metadata(
// #         metadata_invalid_latents_field_latents_name
// #     )

// #     assert expected == parsed_metadata
