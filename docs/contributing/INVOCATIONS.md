# Invocations

Invocations represent a single operation, its inputs, and its outputs. These
operations and their outputs can be chained together to generate and modify
images.

## Creating a new invocation

To create a new invocation, either find the appropriate module file in
`/ldm/invoke/app/invocations` to add your invocation to, or create a new one in
that folder. All invocations in that folder will be discovered and made
available to the CLI and API automatically. Invocations make use of
[typing](https://docs.python.org/3/library/typing.html) and
[pydantic](https://pydantic-docs.helpmanual.io/) for validation and integration
into the CLI and API.

An invocation looks like this:

```py
class UpscaleInvocation(BaseInvocation):
    """Upscales an image."""

    # fmt: off
    type: Literal["upscale"] = "upscale"

    # Inputs
    image: Union[ImageField, None] = Field(description="The input image", default=None)
    strength: float                = Field(default=0.75, gt=0, le=1, description="The strength")
    level: Literal[2, 4]           = Field(default=2, description="The upscale level")
    # fmt: on

    # Schema customisation
    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "tags": ["upscaling", "image"],
            },
        }

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.services.images.get_pil_image(
            self.image.image_origin, self.image.image_name
        )
        results = context.services.restoration.upscale_and_reconstruct(
            image_list=[[image, 0]],
            upscale=(self.level, self.strength),
            strength=0.0,  # GFPGAN strength
            save_original=False,
            image_callback=None,
        )

        # Results are image and seed, unwrap for now
        # TODO: can this return multiple results?
        image_dto = context.services.images.create(
            image=results[0][0],
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
        )

        return ImageOutput(
            image=ImageField(
                image_name=image_dto.image_name,
                image_origin=image_dto.image_origin,
            ),
            width=image_dto.width,
            height=image_dto.height,
        )

```

Each portion is important to implement correctly.

### Class definition and type

```py
class UpscaleInvocation(BaseInvocation):
    """Upscales an image."""
    type: Literal['upscale'] = 'upscale'
```

All invocations must derive from `BaseInvocation`. They should have a docstring
that declares what they do in a single, short line. They should also have a
`type` with a type hint that's `Literal["command_name"]`, where `command_name`
is what the user will type on the CLI or use in the API to create this
invocation. The `command_name` must be unique. The `type` must be assigned to
the value of the literal in the type hint.

### Inputs

```py
    # Inputs
    image: Union[ImageField,None] = Field(description="The input image")
    strength: float               = Field(default=0.75, gt=0, le=1, description="The strength")
    level: Literal[2,4]           = Field(default=2, description="The upscale level")
```

Inputs consist of three parts: a name, a type hint, and a `Field` with default,
description, and validation information. For example:

| Part      | Value                                                         | Description                                                                                                               |
| --------- | ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| Name      | `strength`                                                    | This field is referred to as `strength`                                                                                   |
| Type Hint | `float`                                                       | This field must be of type `float`                                                                                        |
| Field     | `Field(default=0.75, gt=0, le=1, description="The strength")` | The default value is `0.75`, the value must be in the range (0,1], and help text will show "The strength" for this field. |

Notice that `image` has type `Union[ImageField,None]`. The `Union` allows this
field to be parsed with `None` as a value, which enables linking to previous
invocations. All fields should either provide a default value or allow `None` as
a value, so that they can be overwritten with a linked output from another
invocation.

The special type `ImageField` is also used here. All images are passed as
`ImageField`, which protects them from pydantic validation errors (since images
only ever come from links).

Finally, note that for all linking, the `type` of the linked fields must match.
If the `name` also matches, then the field can be **automatically linked** to a
previous invocation by name and matching.

### Config

```py
    # Schema customisation
    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "tags": ["upscaling", "image"],
            },
        }
```

This is an optional configuration for the invocation. It inherits from
pydantic's model `Config` class, and it used primarily to customize the
autogenerated OpenAPI schema.

The UI relies on the OpenAPI schema in two ways:

- An API client & Typescript types are generated from it. This happens at build
  time.
- The node editor parses the schema into a template used by the UI to create the
  node editor UI. This parsing happens at runtime.

In this example, a `ui` key has been added to the `schema_extra` dict to provide
some tags for the UI, to facilitate filtering nodes.

See the Schema Generation section below for more information.

### Invoke Function

```py
    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.services.images.get_pil_image(
            self.image.image_origin, self.image.image_name
        )
        results = context.services.restoration.upscale_and_reconstruct(
            image_list=[[image, 0]],
            upscale=(self.level, self.strength),
            strength=0.0,  # GFPGAN strength
            save_original=False,
            image_callback=None,
        )

        # Results are image and seed, unwrap for now
        # TODO: can this return multiple results?
        image_dto = context.services.images.create(
            image=results[0][0],
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
        )

        return ImageOutput(
            image=ImageField(
                image_name=image_dto.image_name,
                image_origin=image_dto.image_origin,
            ),
            width=image_dto.width,
            height=image_dto.height,
        )
```

The `invoke` function is the last portion of an invocation. It is provided an
`InvocationContext` which contains services to perform work as well as a
`session_id` for use as needed. It should return a class with output values that
derives from `BaseInvocationOutput`.

Before being called, the invocation will have all of its fields set from
defaults, inputs, and finally links (overriding in that order).

Assume that this invocation may be running simultaneously with other
invocations, may be running on another machine, or in other interesting
scenarios. If you need functionality, please provide it as a service in the
`InvocationServices` class, and make sure it can be overridden.

### Outputs

```py
class ImageOutput(BaseInvocationOutput):
    """Base class for invocations that output an image"""

    # fmt: off
    type: Literal["image_output"] = "image_output"
    image:      ImageField = Field(default=None, description="The output image")
    width:             int = Field(description="The width of the image in pixels")
    height:            int = Field(description="The height of the image in pixels")
    # fmt: on

    class Config:
        schema_extra = {"required": ["type", "image", "width", "height"]}
```

Output classes look like an invocation class without the invoke method. Prefer
to use an existing output class if available, and prefer to name inputs the same
as outputs when possible, to promote automatic invocation linking.

## Schema Generation

Invocation, output and related classes are used to generate an OpenAPI schema.

### Required Properties

The schema generation treat all properties with default values as optional. This
makes sense internally, but when when using these classes via the generated
schema, we end up with e.g. the `ImageOutput` class having its `image` property
marked as optional.

We know that this property will always be present, so the additional logic
needed to always check if the property exists adds a lot of extraneous cruft.

To fix this, we can leverage `pydantic`'s
[schema customisation](https://docs.pydantic.dev/usage/schema/#schema-customization)
to mark properties that we know will always be present as required.

Here's that `ImageOutput` class, without the needed schema customisation:

```python
class ImageOutput(BaseInvocationOutput):
    """Base class for invocations that output an image"""

    # fmt: off
    type: Literal["image_output"] = "image_output"
    image:      ImageField = Field(default=None, description="The output image")
    width:             int = Field(description="The width of the image in pixels")
    height:            int = Field(description="The height of the image in pixels")
    # fmt: on
```

The OpenAPI schema that results from this `ImageOutput` will have the `type`,
`image`, `width` and `height` properties marked as optional, even though we know
they will always have a value.

```python
class ImageOutput(BaseInvocationOutput):
    """Base class for invocations that output an image"""

    # fmt: off
    type: Literal["image_output"] = "image_output"
    image:      ImageField = Field(default=None, description="The output image")
    width:             int = Field(description="The width of the image in pixels")
    height:            int = Field(description="The height of the image in pixels")
    # fmt: on

    # Add schema customization
    class Config:
        schema_extra = {"required": ["type", "image", "width", "height"]}
```

With the customization in place, the schema will now show these properties as
required, obviating the need for extensive null checks in client code.

See this `pydantic` issue for discussion on this solution:
<https://github.com/pydantic/pydantic/discussions/4577>
