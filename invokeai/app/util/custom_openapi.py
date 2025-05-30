from typing import Any, Callable, Optional

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from pydantic.json_schema import models_json_schema

from invokeai.app.invocations.baseinvocation import (
    InvocationRegistry,
    UIConfigBase,
)
from invokeai.app.invocations.fields import InputFieldJSONSchemaExtra, OutputFieldJSONSchemaExtra
from invokeai.app.invocations.model import ModelIdentifierField
from invokeai.app.services.events.events_common import EventBase
from invokeai.app.services.session_processor.session_processor_common import ProgressImage
from invokeai.backend.util.logging import InvokeAILogger

logger = InvokeAILogger.get_logger()


def move_defs_to_top_level(openapi_schema: dict[str, Any], component_schema: dict[str, Any]) -> None:
    """Moves a component schema's $defs to the top level of the openapi schema. Useful when generating a schema
    for a single model that needs to be added back to the top level of the schema. Mutates openapi_schema and
    component_schema."""

    defs = component_schema.pop("$defs", {})
    for schema_key, json_schema in defs.items():
        if schema_key in openapi_schema["components"]["schemas"]:
            continue
        openapi_schema["components"]["schemas"][schema_key] = json_schema


def get_openapi_func(
    app: FastAPI, post_transform: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None
) -> Callable[[], dict[str, Any]]:
    """Gets the OpenAPI schema generator function.

    Args:
        app (FastAPI): The FastAPI app to generate the schema for.
        post_transform (Optional[Callable[[dict[str, Any]], dict[str, Any]]], optional): A function to apply to the
            generated schema before returning it. Defaults to None.

    Returns:
        Callable[[], dict[str, Any]]: The OpenAPI schema generator function. When first called, the generated schema is
            cached in `app.openapi_schema`. On subsequent calls, the cached schema is returned. This caching behaviour
            matches FastAPI's default schema generation caching.
    """

    def openapi() -> dict[str, Any]:
        if app.openapi_schema:
            return app.openapi_schema

        openapi_schema = get_openapi(
            title=app.title,
            description="An API for invoking AI image operations",
            version="1.0.0",
            routes=app.routes,
            separate_input_output_schemas=False,  # https://fastapi.tiangolo.com/how-to/separate-openapi-schemas/
        )

        # We'll create a map of invocation type to output schema to make some types simpler on the client.
        invocation_output_map_properties: dict[str, Any] = {}
        invocation_output_map_required: list[str] = []

        # We need to manually add all outputs to the schema - pydantic doesn't add them because they aren't used directly.
        for output in InvocationRegistry.get_output_classes():
            json_schema = output.model_json_schema(mode="serialization", ref_template="#/components/schemas/{model}")
            # Remove output_metadata that is only used on back-end from the schema
            if "output_meta" in json_schema["properties"]:
                json_schema["properties"].pop("output_meta")

            move_defs_to_top_level(openapi_schema, json_schema)
            openapi_schema["components"]["schemas"][output.__name__] = json_schema

        # Technically, invocations are added to the schema by pydantic, but we still need to manually set their output
        # property, so we'll just do it all manually.
        for invocation in InvocationRegistry.get_invocation_classes():
            json_schema = invocation.model_json_schema(
                mode="serialization", ref_template="#/components/schemas/{model}"
            )
            move_defs_to_top_level(openapi_schema, json_schema)
            output_title = invocation.get_output_annotation().__name__
            outputs_ref = {"$ref": f"#/components/schemas/{output_title}"}
            json_schema["output"] = outputs_ref
            openapi_schema["components"]["schemas"][invocation.__name__] = json_schema

            # Add this invocation and its output to the output map
            invocation_type = invocation.get_type()
            invocation_output_map_properties[invocation_type] = json_schema["output"]
            invocation_output_map_required.append(invocation_type)

        # Add the output map to the schema
        openapi_schema["components"]["schemas"]["InvocationOutputMap"] = {
            "type": "object",
            "properties": dict(sorted(invocation_output_map_properties.items())),
            "required": invocation_output_map_required,
        }

        # Some models don't end up in the schemas as standalone definitions because they aren't used directly in the API.
        # We need to add them manually here. WARNING: Pydantic can choke if you call `model.model_json_schema()` to get
        # a schema. This has something to do with schema refs - not totally clear. For whatever reason, using
        # `models_json_schema` seems to work fine.
        additional_models = [
            *EventBase.get_events(),
            UIConfigBase,
            InputFieldJSONSchemaExtra,
            OutputFieldJSONSchemaExtra,
            ModelIdentifierField,
            ProgressImage,
        ]

        additional_schemas = models_json_schema(
            [(m, "serialization") for m in additional_models],
            ref_template="#/components/schemas/{model}",
        )
        # additional_schemas[1] is a dict of $defs that we need to add to the top level of the schema
        move_defs_to_top_level(openapi_schema, additional_schemas[1])

        if post_transform is not None:
            openapi_schema = post_transform(openapi_schema)

        openapi_schema["components"]["schemas"] = dict(sorted(openapi_schema["components"]["schemas"].items()))

        app.openapi_schema = openapi_schema
        return app.openapi_schema

    return openapi
