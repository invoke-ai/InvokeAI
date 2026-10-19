from pydantic import BaseModel

from invokeai.app.invocations.fields import OutputField, OutputScope


class ScopedOutputModel(BaseModel):
    iteration_value: str = OutputField(output_scope=OutputScope.Iteration)
    final_value: str = OutputField(output_scope=OutputScope.Final)
    ordinary_value: str = OutputField()


def test_output_field_scope_is_included_in_json_schema() -> None:
    schema = ScopedOutputModel.model_json_schema()

    assert schema["properties"]["iteration_value"]["output_scope"] == "iteration"
    assert schema["properties"]["final_value"]["output_scope"] == "final"
    assert "output_scope" not in schema["properties"]["ordinary_value"]
