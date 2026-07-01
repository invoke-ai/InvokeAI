from fastapi import FastAPI
from pydantic import create_model

from invokeai.app.invocations.baseinvocation import InvocationRegistry
from invokeai.app.util.custom_openapi import get_openapi_func


class _FakeOutput:
    pass


class _InvocationB:
    __name__ = "InvocationB"

    @classmethod
    def model_json_schema(cls, mode: str, ref_template: str) -> dict:
        return {"type": "object", "properties": {}}

    @classmethod
    def get_output_annotation(cls) -> type:
        return _FakeOutput

    @classmethod
    def get_type(cls) -> str:
        return "b_type"


class _InvocationA:
    __name__ = "InvocationA"

    @classmethod
    def model_json_schema(cls, mode: str, ref_template: str) -> dict:
        return {"type": "object", "properties": {}}

    @classmethod
    def get_output_annotation(cls) -> type:
        return _FakeOutput

    @classmethod
    def get_type(cls) -> str:
        return "a_type"


def test_invocation_output_map_required_is_sorted(monkeypatch: object) -> None:
    """The 'required' list in InvocationOutputMap must be sorted so that the
    generated openapi.json is deterministic regardless of set-iteration order."""

    # A FastAPI app needs at least one route to produce a schema with 'components'.
    DummyResponse = create_model("DummyResponse", ok=(bool, ...))
    app = FastAPI(title="test")
    app.get("/healthz", response_model=DummyResponse)(lambda: DummyResponse(ok=True))

    monkeypatch.setattr(InvocationRegistry, "get_output_classes", classmethod(lambda cls: []))  # type: ignore[arg-type]
    monkeypatch.setattr(  # type: ignore[arg-type]
        InvocationRegistry, "get_invocation_classes", classmethod(lambda cls: [_InvocationB, _InvocationA])
    )

    schema = get_openapi_func(app)()
    required = schema["components"]["schemas"]["InvocationOutputMap"]["required"]

    assert required == ["a_type", "b_type"], f"Expected sorted required list, got: {required}"
