from typing import Any, Literal, Optional, Union

import pytest
from pydantic import BaseModel


class TestModel(BaseModel):
    foo: Literal["bar"] = "bar"


@pytest.mark.parametrize(
    "input_type, expected",
    [
        (str, False),
        (list[str], False),
        (list[dict[str, Any]], False),
        (list[None], False),
        (list[dict[str, None]], False),
        (Any, False),
        (True, False),
        (False, False),
        (Union[str, False], False),
        (Union[str, True], False),
        (None, False),
        (str | None, True),
        (Union[str, None], True),
        (Optional[str], True),
        (str | int | None, True),
        (None | str | int, True),
        (Union[None, str], True),
        (Optional[str], True),
        (Optional[int], True),
        (Optional[str], True),
        (TestModel | None, True),
        (Union[TestModel, None], True),
        (Optional[TestModel], True),
    ],
)
def test_is_optional(input_type: Any, expected: bool) -> None:
    """
    Test the is_optional function.
    """
    from invokeai.app.invocations.baseinvocation import is_optional

    result = is_optional(input_type)
    assert result == expected, f"Expected {expected} but got {result} for input type {input_type}"
