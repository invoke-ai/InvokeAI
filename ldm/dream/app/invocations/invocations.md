# Invocations

Invocations represent a single operation, its inputs, and its outputs. These operations and their outputs can be chained together to generate and modify images.

## Creating a new invocation

To create a new invocation, either find the appropriate module file in `/ldm/dream/app/invocations` to add your invocation to, or create a new one in that folder. All invocations in that folder will be discovered and made available to the CLI and API automatically. Invocations make use of [typing](https://docs.python.org/3/library/typing.html) and [pydantic](https://pydantic-docs.helpmanual.io/) for validation and integration into the CLI and API.

An invocation looks like this:

```py
class UpscaleInvocation(BaseInvocation):
    """Generates an image using text2img."""
    type: Literal['upscale'] = 'upscale'

    # Inputs
    image: Union[ImageField,None] = Field(description="The input image")
    strength: float               = Field(default=0.75, gt=0, le=1, description="The strength")
    level: UpscaleLevel           = Field(default=2, description="The upscale level")

    def invoke(self, services: InvocationServices, session_id: str) -> ImageOutput: 
        result_image = services.generate.upscale(
            image    = self.image.get(),
            level    = self.level
            strength = self.strength
        )

        return ImageOutput.construct(
            image = ImageField.from_image(result_image)
        )
```

Each portion is important to implement correctly.

### Class definition and type
```py
class UpscaleInvocation(BaseInvocation):
    """Generates an image using text2img."""
    type: Literal["upscale"]
```
All invocations must derive from `BaseInvocation`. They should have a docstring that declares what they do in a single, short line. They should also have a `type` with a type hint that's `Literal["command_name"]`, where `command_name` is what the user will type on the CLI or use in the API to create this invocation. The `command_name` must be unique.

### Inputs
```py
    # Inputs
    image: Union[ImageField,None] = Field(description="The input image")
    strength: float               = Field(default=0.75, gt=0, le=1, description="The strength")
    level: UpscaleLevel           = Field(default=2, description="The upscale level")
```
Inputs consist of three parts: a name, a type hint, and a `Field` with default, description, and validation information. For example:
| Part | Value | Description |
| ---- | ----- | ----------- |
| Name | `strength` | This field is referred to as `strength` |
| Type Hint | `float` | This field must be of type `float` |
| Field | `Field(default=0.75, gt=0, le=1, description="The strength")` | The default value is `0.75`, the value must be in the range (0,1], and help text will show "The strength" for this field. |

Notice that `image` has type `Union[ImageField,None]`. The `Union` allows this field to be parsed with `None` as a value, which enables linking to previous invocations. All fields should either provide a default value or allow `None` as a value, so that they can be overwritten with a linked output from another invocation.

The special type `ImageField` is also used here. All images are passed as `ImageField`, which protects them from pydantic validation errors (since images only ever come from links).

Finally, note that for all linking, the `type` of the linked fields must match. If the `name` also matches, then the field can be **automatically linked** to a previous invocation by name and matching.

### Invoke Function
```py
    def invoke(self, services: InvocationServices, session_id: str) -> ImageOutput: 
        result_image = context.services.generate.upscale(
            image    = self.image.get(),
            level    = self.level
            strength = self.strength
        )

        return ImageOutput.construct(
            image = ImageField.from_image(result_image)
        )
```
The `invoke` function is the last portion of an invocation. It is provided an `InvocationServices` which contains services to perform work. It should return an Outputs class that derives from `BaseInvocationOutput`.

Before being called, the invocation will have all of its fields set from defaults, inputs, and finally links (overriding in that order).

Assume that this invocation may be running simultaneously with other invocations, may be running on another machine, or in other interesting scenarios. If you need functionality, please provide it as a service in the `InvocationServices` class, and make sure it can be overridden.

Finally, you may notice that the Outputs are being created with `construct()`. There are [many ways of creating a pydantic model](https://pydantic-docs.helpmanual.io/usage/models/). Construct will skip validation, though you are welcome to use another if you'd prefer.
