# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

from __future__ import annotations

from abc import ABC, abstractmethod
from inspect import signature
from typing import TYPE_CHECKING, Dict, List, Literal, TypedDict, get_args, get_type_hints

from pydantic import BaseConfig, BaseModel, Field

if TYPE_CHECKING:
    from ..services.invocation_services import InvocationServices


class InvocationContext:
    services: InvocationServices
    graph_execution_state_id: str

    def __init__(self, services: InvocationServices, graph_execution_state_id: str):
        self.services = services
        self.graph_execution_state_id = graph_execution_state_id


class BaseInvocationOutput(BaseModel):
    """Base class for all invocation outputs"""

    # All outputs must include a type name like this:
    # type: Literal['your_output_name']

    @classmethod
    def get_all_subclasses_tuple(cls):
        subclasses = []
        toprocess = [cls]
        while len(toprocess) > 0:
            next = toprocess.pop(0)
            next_subclasses = next.__subclasses__()
            subclasses.extend(next_subclasses)
            toprocess.extend(next_subclasses)
        return tuple(subclasses)


class BaseInvocation(ABC, BaseModel):
    """A node to process inputs and produce outputs.
    May use dependency injection in __init__ to receive providers.
    """

    # All invocations must include a type name like this:
    # type: Literal['your_output_name']

    @classmethod
    def get_all_subclasses(cls):
        subclasses = []
        toprocess = [cls]
        while len(toprocess) > 0:
            next = toprocess.pop(0)
            next_subclasses = next.__subclasses__()
            subclasses.extend(next_subclasses)
            toprocess.extend(next_subclasses)
        return subclasses

    @classmethod
    def get_invocations(cls):
        return tuple(BaseInvocation.get_all_subclasses())

    @classmethod
    def get_invocations_map(cls):
        # Get the type strings out of the literals and into a dictionary
        return dict(
            map(
                lambda t: (get_args(get_type_hints(t)["type"])[0], t),
                BaseInvocation.get_all_subclasses(),
            )
        )

    @classmethod
    def get_output_type(cls):
        return signature(cls.invoke).return_annotation

    @abstractmethod
    def invoke(self, context: InvocationContext) -> BaseInvocationOutput:
        """Invoke with provided context and return outputs."""
        pass

    # fmt: off
    id: str = Field(description="The id of this node. Must be unique among all nodes.")
    is_intermediate: bool = Field(default=False, description="Whether or not this node is an intermediate node.")
    # fmt: on


# TODO: figure out a better way to provide these hints
# TODO: when we can upgrade to python 3.11, we can use the`NotRequired` type instead of `total=False`
class UIConfig(TypedDict, total=False):
    type_hints: Dict[
        str,
        Literal[
            "integer",
            "float",
            "boolean",
            "string",
            "enum",
            "image",
            "latents",
            "model",
            "control",
            "image_collection",
            "vae_model",
            "lora_model",
        ],
    ]
    tags: List[str]
    title: str


class CustomisedSchemaExtra(TypedDict):
    ui: UIConfig


class InvocationConfig(BaseConfig):
    """Customizes pydantic's BaseModel.Config class for use by Invocations.

    Provide `schema_extra` a `ui` dict to add hints for generated UIs.

    `tags`
    - A list of strings, used to categorise invocations.

    `type_hints`
    - A dict of field types which override the types in the invocation definition.
    - Each key should be the name of one of the invocation's fields.
    - Each value should be one of the valid types:
      - `integer`, `float`, `boolean`, `string`, `enum`, `image`, `latents`, `model`

    ```python
    class Config(InvocationConfig):
      schema_extra = {
          "ui": {
              "tags": ["stable-diffusion", "image"],
              "type_hints": {
                  "initial_image": "image",
              },
          },
      }
    ```
    """

    schema_extra: CustomisedSchemaExtra
