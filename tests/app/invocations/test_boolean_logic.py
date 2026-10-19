from typing import Literal
from unittest.mock import MagicMock

import pytest

from invokeai.app.invocations.logic import BooleanLogicInvocation
from invokeai.app.invocations.primitives import BooleanOutput


@pytest.mark.parametrize("operation", ["AND", "OR", "XOR", "NAND", "NOR", "XNOR", "NOT"])
@pytest.mark.parametrize("a", [False, True])
@pytest.mark.parametrize("b", [False, True])
def test_boolean_logic(operation: Literal["AND", "OR", "XOR", "NAND", "NOR", "XNOR", "NOT"], a: bool, b: bool) -> None:
    expected = {
        "AND": a and b,
        "OR": a or b,
        "XOR": a != b,
        "NAND": not (a and b),
        "NOR": not (a or b),
        "XNOR": a == b,
        "NOT": not a,
    }[operation]

    output = BooleanLogicInvocation(id="boolean_logic", operation=operation, a=a, b=b).invoke(MagicMock())

    assert isinstance(output, BooleanOutput)
    assert output.value is expected
