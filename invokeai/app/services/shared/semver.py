from typing import Annotated, Any, Callable

from pydantic import GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema
from semver import Version


class _VersionPydanticAnnotation:
    """
    Pydantic annotation for semver.Version.

    Requires a field_serializer to serialize to a string.

    Usage:
        class MyModel(BaseModel):
            version: SemVer = Field(..., description="The version of the model.")

            @field_serializer("version")
            def serialize_version(self, version: SemVer, _info):
                return str(version)

        MyModel(version=semver.Version.parse("1.2.3"))
        MyModel.model_validate({"version":"1.2.3"})
    """

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: Callable[[Any], core_schema.CoreSchema],
    ) -> core_schema.CoreSchema:
        def validate_from_str(value: str) -> Version:
            return Version.parse(value)

        from_str_schema = core_schema.chain_schema(
            [
                core_schema.str_schema(),
                core_schema.no_info_plain_validator_function(validate_from_str),
            ]
        )

        return core_schema.json_or_python_schema(
            json_schema=from_str_schema,
            python_schema=core_schema.union_schema(
                [
                    core_schema.is_instance_schema(Version),
                    from_str_schema,
                ]
            ),
            serialization=core_schema.to_string_ser_schema(),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, _core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        return handler(core_schema.str_schema())


SemVer = Annotated[Version, _VersionPydanticAnnotation]
