from unittest.mock import Mock

import pytest

from invokeai.app.invocations.collections import (
    MAX_CARTESIAN_PRODUCT_SIZE,
    CollectionCartesianInvocation,
    CollectionConcatInvocation,
    CollectionZipInvocation,
)


def test_collection_concat_preserves_left_then_right_order() -> None:
    invocation = CollectionConcatInvocation(id="concat", first=[1, 2], second=[3, 4])

    output = invocation.invoke(Mock())

    assert output.collection == [1, 2, 3, 4]


def test_collection_concat_handles_empty_inputs() -> None:
    invocation = CollectionConcatInvocation(id="concat", first=[], second=["value"])

    output = invocation.invoke(Mock())

    assert output.collection == ["value"]


def test_collection_concat_does_not_mutate_input_collections() -> None:
    first = ["left"]
    second = ["right"]
    invocation = CollectionConcatInvocation(id="concat", first=first, second=second)

    output = invocation.invoke(Mock())
    output.collection.append("changed")

    assert first == ["left"]
    assert second == ["right"]


def test_collection_zip_preserves_positional_order_as_pairs() -> None:
    invocation = CollectionZipInvocation(id="zip", first=[1, 2], second=["a", "b"])

    output = invocation.invoke(Mock())

    assert output.collection == [[1, "a"], [2, "b"]]


def test_collection_zip_rejects_unequal_input_lengths() -> None:
    invocation = CollectionZipInvocation(id="zip", first=[1], second=["a", "b"])

    with pytest.raises(ValueError, match="same length"):
        invocation.invoke(Mock())


def test_collection_zip_handles_empty_inputs() -> None:
    invocation = CollectionZipInvocation(id="zip", first=[], second=[])

    output = invocation.invoke(Mock())

    assert output.collection == []


def test_collection_zip_does_not_mutate_input_collections() -> None:
    first = ["left"]
    second = ["right"]
    invocation = CollectionZipInvocation(id="zip", first=first, second=second)

    output = invocation.invoke(Mock())
    output.collection.append(["changed", "value"])

    assert first == ["left"]
    assert second == ["right"]


def test_collection_cartesian_preserves_left_major_right_minor_order() -> None:
    invocation = CollectionCartesianInvocation(id="cartesian", first=[1, 2], second=["a", "b"])

    output = invocation.invoke(Mock())

    assert output.collection == [[1, "a"], [1, "b"], [2, "a"], [2, "b"]]


def test_collection_cartesian_accepts_unequal_input_lengths() -> None:
    invocation = CollectionCartesianInvocation(id="cartesian", first=[1], second=["a", "b", "c"])

    output = invocation.invoke(Mock())

    assert output.collection == [[1, "a"], [1, "b"], [1, "c"]]


def test_collection_cartesian_handles_empty_inputs() -> None:
    invocation = CollectionCartesianInvocation(id="cartesian", first=[], second=["value"])

    output = invocation.invoke(Mock())

    assert output.collection == []


def test_collection_cartesian_does_not_mutate_input_collections() -> None:
    first = ["left"]
    second = ["right"]
    invocation = CollectionCartesianInvocation(id="cartesian", first=first, second=second)

    output = invocation.invoke(Mock())
    output.collection.append(["changed", "value"])

    assert first == ["left"]
    assert second == ["right"]


def test_collection_cartesian_rejects_products_above_the_bound() -> None:
    invocation = CollectionCartesianInvocation(
        id="cartesian", first=list(range(MAX_CARTESIAN_PRODUCT_SIZE + 1)), second=["value"]
    )

    with pytest.raises(ValueError, match=str(MAX_CARTESIAN_PRODUCT_SIZE)):
        invocation.invoke(Mock())


def test_collection_cartesian_accepts_a_product_at_the_bound() -> None:
    invocation = CollectionCartesianInvocation(
        id="cartesian", first=list(range(MAX_CARTESIAN_PRODUCT_SIZE)), second=["value"]
    )

    output = invocation.invoke(Mock())

    assert len(output.collection) == MAX_CARTESIAN_PRODUCT_SIZE
