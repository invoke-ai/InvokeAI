# Invocations

Invocations represent a single operation, its inputs, and its outputs. These operations and their outputs can be chained together to generate and modify images.

## Creating a new invocation

To create a new invocation, either find the appropriate module file in `/ldm/invoke/app/invocations` to add your invocation to, or create a new one in that folder. All invocations in that folder will be discovered and made available to the CLI and API automatically. Invocations make use of [typing](https://docs.python.org/3/library/typing.html) and [pydantic](https://pydantic-docs.helpmanual.io/) for validation and integration into the CLI and API.

An invocation looks like this:

```py
class UpscaleInvocation(BaseInvocation):
    """Upscales an image."""
    type: Literal['upscale'] = 'upscale'

    # Inputs
    image: Union[ImageField,None] = Field(description="The input image")
    strength: float               = Field(default=0.75, gt=0, le=1, description="The strength")
    level: Literal[2,4]           = Field(default=2, description = "The upscale level")

    def invoke(self, services: InvocationServices, session_id: str) -> ImageOutput:
        image = services.images.get(self.image.image_type, self.image.image_name)
        results = services.generate.upscale_and_reconstruct(
            image_list     = [[image, 0]],
            upscale        = (self.level, self.strength),
            strength       = 0.0, # GFPGAN strength
            save_original  = False,
            image_callback = None,
        )

        # Results are image and seed, unwrap for now
        # TODO: can this return multiple results?
        image_type = ImageType.RESULT
        image_name = f'{session_id}_{self.id}_{str(int(datetime.now(timezone.utc).timestamp()))}.png'
        services.images.save(image_type, image_name, results[0][0])
        return ImageOutput(
            image = ImageField(image_type = image_type, image_name = image_name)
        )
```

Each portion is important to implement correctly.

### Class definition and type
```py
class UpscaleInvocation(BaseInvocation):
    """Upscales an image."""
    type: Literal['upscale'] = 'upscale'
```
All invocations must derive from `BaseInvocation`. They should have a docstring that declares what they do in a single, short line. They should also have a `type` with a type hint that's `Literal["command_name"]`, where `command_name` is what the user will type on the CLI or use in the API to create this invocation. The `command_name` must be unique. The `type` must be assigned to the value of the literal in the type hint.

### Inputs
```py
    # Inputs
    image: Union[ImageField,None] = Field(description="The input image")
    strength: float               = Field(default=0.75, gt=0, le=1, description="The strength")
    level: Literal[2,4]           = Field(default=2, description="The upscale level")
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
        image = services.images.get(self.image.image_type, self.image.image_name)
        results = services.generate.upscale_and_reconstruct(
            image_list     = [[image, 0]],
            upscale        = (self.level, self.strength),
            strength       = 0.0, # GFPGAN strength
            save_original  = False,
            image_callback = None,
        )

        # Results are image and seed, unwrap for now
        image_type = ImageType.RESULT
        image_name = f'{session_id}_{self.id}_{str(int(datetime.now(timezone.utc).timestamp()))}.png'
        services.images.save(image_type, image_name, results[0][0])
        return ImageOutput(
            image = ImageField(image_type = image_type, image_name = image_name)
        )
```
The `invoke` function is the last portion of an invocation. It is provided an `InvocationServices` which contains services to perform work as well as a `session_id` for use as needed. It should return a class with output values that derives from `BaseInvocationOutput`.

Before being called, the invocation will have all of its fields set from defaults, inputs, and finally links (overriding in that order).

Assume that this invocation may be running simultaneously with other invocations, may be running on another machine, or in other interesting scenarios. If you need functionality, please provide it as a service in the `InvocationServices` class, and make sure it can be overridden.

### Outputs
```py
class ImageOutput(BaseInvocationOutput):
    """Base class for invocations that output an image"""
    type: Literal['image'] = 'image'

    image: ImageField = Field(default=None, description="The output image")
```
Output classes look like an invocation class without the invoke method. Prefer to use an existing output class if available, and prefer to name inputs the same as outputs when possible, to promote automatic invocation linking.
