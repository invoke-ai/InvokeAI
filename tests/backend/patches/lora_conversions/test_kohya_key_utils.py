import pytest

from invokeai.backend.patches.lora_conversions.kohya_key_utils import (
    INDEX_PLACEHOLDER,
    ParsingTree,
    generate_kohya_parsing_tree_from_keys,
    insert_periods_into_kohya_key,
)


def test_insert_periods_into_kohya_key():
    """Test that insert_periods_into_kohya_key() correctly inserts periods into a Kohya key."""
    key = "module_a_module_b_0_attn_to_k"
    parsing_tree: ParsingTree = {
        "module_a": {
            "module_b": {
                INDEX_PLACEHOLDER: {
                    "attn": {
                        "to_k": {},
                    },
                },
            },
        },
    }
    result = insert_periods_into_kohya_key(key, parsing_tree)
    assert result == "module_a.module_b.0.attn.to_k"


def test_insert_periods_into_kohya_key_invalid_key():
    """Test that insert_periods_into_kohya_key() raises ValueError for a key that is invalid."""
    key = "invalid_key_format"
    parsing_tree: ParsingTree = {
        "module_a": {
            "module_b": {
                INDEX_PLACEHOLDER: {
                    "attn": {
                        "to_k": {},
                    },
                },
            },
        },
    }
    with pytest.raises(ValueError):
        insert_periods_into_kohya_key(key, parsing_tree)


def test_insert_periods_into_kohya_key_too_long():
    """Test that insert_periods_into_kohya_key() raises ValueError for a key that has a valid prefix, but is too long."""
    key = "module_a.module_b.0.attn.to_k.invalid_suffix"
    parsing_tree: ParsingTree = {
        "module_a": {
            "module_b": {
                INDEX_PLACEHOLDER: {
                    "attn": {
                        "to_k": {},
                    },
                },
            },
        },
    }
    with pytest.raises(ValueError):
        insert_periods_into_kohya_key(key, parsing_tree)


def test_generate_kohya_parsing_tree_from_keys():
    """Test that generate_kohya_parsing_tree_from_keys() correctly generates a parsing tree."""
    keys = [
        "module_a.module_b.0.attn.to_k",
        "module_a.module_b.1.attn.to_k",
        "module_a.module_c.proj",
    ]

    expected_tree: ParsingTree = {
        "module_a": {
            "module_b": {
                INDEX_PLACEHOLDER: {
                    "attn": {
                        "to_k": {},
                    },
                }
            },
            "module_c": {
                "proj": {},
            },
        }
    }

    tree = generate_kohya_parsing_tree_from_keys(keys)
    assert tree == expected_tree


def test_generate_kohya_parsing_tree_from_empty_keys():
    """Test that generate_kohya_parsing_tree_from_keys() handles empty input."""
    keys: list[str] = []
    tree = generate_kohya_parsing_tree_from_keys(keys)
    assert tree == {}
