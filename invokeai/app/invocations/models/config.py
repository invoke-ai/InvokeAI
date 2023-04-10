from typing import Dict, List, Literal, TypedDict
from pydantic import BaseModel


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
        ],
    ]
    tags: List[str]


class CustomisedSchemaExtra(TypedDict):
    ui: UIConfig


class InvocationConfig(BaseModel.Config):
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
