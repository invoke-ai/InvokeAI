from copy import deepcopy
from typing import Any, Callable, TypeAlias

from pydantic import BaseModel, create_model
from pydantic.fields import FieldInfo

from invokeai.app.services.session_queue.session_queue_common import FieldIdentifier
from invokeai.app.services.shared.graph import Graph

DictOfFieldsMetadata: TypeAlias = dict[str, tuple[type[Any], FieldInfo]]


class ComposedFieldMetadata(BaseModel):
    node_id: str
    field_name: str
    field_type_class_name: str


def dedupe_field_name(field_metadata: DictOfFieldsMetadata, field_name: str) -> str:
    """Given a field name, return a name that is not already in the field metadata.
    If the field name is not in the field metadata, return the field name.
    If the field name is in the field metadata, generate a new name by appending an underscore and integer to the field name, starting with 2.
    """

    if field_name not in field_metadata:
        return field_name

    i = 2
    while True:
        new_field_name = f"{field_name}_{i}"
        if new_field_name not in field_metadata:
            return new_field_name
        i += 1


def compose_model_from_fields(
    g: Graph,
    field_identifiers: list[FieldIdentifier],
    composed_model_class_name: str = "ComposedModel",
    model_field_overrides: dict[type[Any], tuple[type[Any], FieldInfo]] | None = None,
    model_field_filter: Callable[[type[Any]], bool] | None = None,
) -> type[BaseModel]:
    """Given a graph and a list of field identifiers, create a new pydantic model composed of the fields of the nodes in the graph.

    The resultant model can be used to validate a JSON payload that contains the fields of the nodes in the graph, or generate an
    OpenAPI schema for the model.

    Args:
        g: The graph containing the nodes whose fields will be composed into the new model.
        field_identifiers: A list of FieldIdentifier instances, each representing a field on a node in the graph.
        model_name: The name of the composed model.
        kind: The kind of model to create. Must be "input" or "output". Defaults to "input".
        model_field_overrides: A dictionary mapping type annotations to tuples of (new_type_annotation, new_field_info).
            This can be used to override the type annotation and field info of a field in the composed model. For example,
            if `ModelIdentifierField` should be replaced by a string, the dictionary would look like this:
            ```python
            {ModelIdentifierField: (str, Field(description="The model id."))}
            ```
        model_field_filter: A function that takes a type annotation and returns True if the field should be included in the composed model.
            If None, all fields will be included. For example, to omit `BoardField` fields, the filter would look like this:
            ```python
            def model_field_filter(field_type: type[Any]) -> bool:
                return field_type not in {BoardField}
            ```
            Optional fields - or any other complex field types like unions - must be explicitly included in the filter. For example,
            to omit `BoardField` _and_ `Optional[BoardField]`:
            ```python
            def model_field_filter(field_type: type[Any]) -> bool:
                return field_type not in {BoardField, Optional[BoardField]}
            ```
            Note that the filter is applied to the type annotation of the field, not the field itself.

    Example usage:
    ```python
    # Create some nodes.
    add_node = AddInvocation()
    sub_node = SubtractInvocation()
    color_node = ColorInvocation()

    # Create a graph with the nodes.
    g = Graph(
        nodes={
            add_node.id: add_node,
            sub_node.id: sub_node,
            color_node.id: color_node,
        }
    )

    # Select the fields to compose.
    fields_to_compose = [
        FieldIdentifier(node_id=add_node.id, field_name="a"),
        FieldIdentifier(node_id=sub_node.id, field_name="a"),  # this will be deduped to "a_2"
        FieldIdentifier(node_id=add_node.id, field_name="b"),
        FieldIdentifier(node_id=color_node.id, field_name="color"),
    ]

    # Compose the model from the fields.
    composed_model = compose_model_from_fields(g, fields_to_compose, model_name="ComposedModel")

    # Generate the OpenAPI schema for the model.
    json_schema = composed_model.model_json_schema(mode="validation")
    ```
    """

    # Temp storage for the composed fields. Pydantic needs a type annotation and instance of FieldInfo to create a model.
    field_metadata: DictOfFieldsMetadata = {}
    model_field_overrides = model_field_overrides or {}

    for field_identifier in field_identifiers:
        node_id = field_identifier.node_id
        field_name = field_identifier.field_name

        # Pull the node instance from the graph so we can introspect it.
        node_instance = g.nodes[node_id]

        if field_identifier.kind == "input":
            # Get the class of the node. This will be a BaseInvocation subclass, e.g. AddInvocation, DenoiseLatentsInvocation, etc.
            pydantic_model = type(node_instance)
        else:
            # Otherwise the the type of the node's output class. This will be a BaseInvocationOutput subclass, e.g. IntegerOutput, ImageOutput, etc.
            pydantic_model = type(node_instance).get_output_annotation()

        # Get the FieldInfo instance for the field. For example:
        # a: int = Field(..., description="The first number to add.")
        #          ^^^^^ The return value of this Field call is the FieldInfo instance (Field is a function).
        og_field_info = pydantic_model.model_fields[field_name]

        # Get the type annotation of the field. For example:
        # a: int = Field(..., description="The first number to add.")
        #    ^^^ this is the type annotation
        og_field_type = og_field_info.annotation

        # Apparently pydantic allows fields without type annotations. We don't support that.
        assert og_field_type is not None, (
            f"{field_identifier.kind.capitalize()} field {field_name} on node {node_id} has no type annotation."
        )

        # Now that we have the type annotation, we can apply the filter to see if we should include the field in the composed model.
        if model_field_filter and not model_field_filter(og_field_type):
            continue

        # Ok, we want this type of field. Retrieve any overrides for the field type. This is a dictionary mapping
        # type annotations to tuples of (override_type_annotation, override_field_info).
        (override_field_type, override_field_info) = model_field_overrides.get(og_field_type, (None, None))

        # The override tuple's first element is the new type annotation, if it exists.
        composed_field_type = override_field_type if override_field_type is not None else og_field_type

        # Create a deep copy of the FieldInfo instance (or override it if it exists) so we can modify it without
        # affecting the original. This is important because we are going to modify the FieldInfo instance and
        # don't want to affect the original model's schema.
        composed_field_info = deepcopy(override_field_info if override_field_info is not None else og_field_info)

        # Invocation fields have some extra metadata, used by the UI to render the field in the frontend. This data is
        # included in the OpenAPI schema for each field. For example, we add a "ui_order" field, which the UI uses to
        # sort fields when rendering them.
        #
        # The composed model's OpenAPI schema should not have this information. It should only have a standard OpenAPI
        # schema for the field. We need to strip out the UI-specific metadata from the FieldInfo instance before adding
        # it to the composed model.
        #
        # We will replace this metadata with some custom metadata:
        # - node_id: The id of the node that this field belongs to.
        # - field_name: The name of the field on the node.
        # - original_data_type: The original data type of the field.

        composed_field_metadata = ComposedFieldMetadata(
            node_id=node_id,
            field_name=field_name,
            field_type_class_name=og_field_type.__name__,
        )

        composed_field_info.json_schema_extra = {
            "composed_field_metadata": composed_field_metadata.model_dump(),
        }

        # Override the name, title and description if overrides are provided. Dedupe the field name if necessary.
        final_field_name = dedupe_field_name(field_metadata, field_name)

        # Store the field metadata.
        field_metadata.update({final_field_name: (composed_field_type, composed_field_info)})

    # Splat in the composed fields to create the new model. There are some type errors here because create_model's kwargs are not typed,
    # but it wants a tuple of (type, FieldInfo) for each field.
    return create_model(composed_model_class_name, **field_metadata)  # pyright: ignore[reportUnknownVariableType, reportCallIssue, reportArgumentType]
