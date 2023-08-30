# Invocations

Features in InvokeAI are added in the form of modular node-like systems called
**Invocations**.

An Invocation is simply a single operation that takes in some inputs and gives
out some outputs. We can then chain multiple Invocations together to create more
complex functionality.

## Invocations Directory

InvokeAI Invocations can be found in the `invokeai/app/invocations` directory.

You can add your new functionality to one of the existing Invocations in this
directory or create a new file in this directory as per your needs.

**Note:** _All Invocations must be inside this directory for InvokeAI to
recognize them as valid Invocations._

## Creating A New Invocation

In order to understand the process of creating a new Invocation, let us actually
create one.

In our example, let us create an Invocation that will take in an image, resize
it and output the resized image.

The first set of things we need to do when creating a new Invocation are -

- Create a new class that derives from a predefined parent class called
  `BaseInvocation`.
- The name of every Invocation must end with the word `Invocation` in order for
  it to be recognized as an Invocation.
- Every Invocation must have a `docstring` that describes what this Invocation
  does.
- Every Invocation must have a unique `type` field defined which becomes its
  indentifier.
- Invocations are strictly typed. We make use of the native
  [typing](https://docs.python.org/3/library/typing.html) library and the
  installed [pydantic](https://pydantic-docs.helpmanual.io/) library for
  validation.

So let us do that.

```python
from typing import Literal
from .baseinvocation import BaseInvocation

class ResizeInvocation(BaseInvocation):
    '''Resizes an image'''
    type: Literal['resize'] = 'resize'
```

That's great.

Now we have setup the base of our new Invocation. Let us think about what inputs
our Invocation takes.

- We need an `image` that we are going to resize.
- We will need new `width` and `height` values to which we need to resize the
  image to.

### **Inputs**

Every Invocation input is a pydantic `Field` and like everything else should be
strictly typed and defined.

So let us create these inputs for our Invocation. First up, the `image` input we
need. Generally, we can use standard variable types in Python but InvokeAI
already has a custom `ImageField` type that handles all the stuff that is needed
for image inputs.

But what is this `ImageField` ..? It is a special class type specifically
written to handle how images are dealt with in InvokeAI. We will cover how to
create your own custom field types later in this guide. For now, let's go ahead
and use it.

```python
from typing import Literal, Union
from pydantic import Field

from .baseinvocation import BaseInvocation
from ..models.image import ImageField

class ResizeInvocation(BaseInvocation):
    '''Resizes an image'''
    type: Literal['resize'] = 'resize'

    # Inputs
    image: Union[ImageField, None] = Field(description="The input image", default=None)
```

Let us break down our input code.

```python
image: Union[ImageField, None] = Field(description="The input image", default=None)
```

| Part      | Value                                                | Description                                                                                        |
| --------- | ---------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| Name      | `image`                                              | The variable that will hold our image                                                              |
| Type Hint | `Union[ImageField, None]`                            | The types for our field. Indicates that the image can either be an `ImageField` type or `None`     |
| Field     | `Field(description="The input image", default=None)` | The image variable is a field which needs a description and a default value that we set to `None`. |

Great. Now let us create our other inputs for `width` and `height`

```python
from typing import Literal, Union
from pydantic import Field

from .baseinvocation import BaseInvocation
from ..models.image import ImageField

class ResizeInvocation(BaseInvocation):
    '''Resizes an image'''
    type: Literal['resize'] = 'resize'

    # Inputs
    image: Union[ImageField, None] = Field(description="The input image", default=None)
    width: int = Field(default=512, ge=64, le=2048, description="Width of the new image")
    height: int = Field(default=512, ge=64, le=2048, description="Height of the new image")
```

As you might have noticed, we added two new parameters to the field type for
`width` and `height` called `gt` and `le`. These basically stand for _greater
than or equal to_ and _less than or equal to_. There are various other param
types for field that you can find on the **pydantic** documentation.

**Note:** _Any time it is possible to define constraints for our field, we
should do it so the frontend has more information on how to parse this field._

Perfect. We now have our inputs. Let us do something with these.

### **Invoke Function**

The `invoke` function is where all the magic happens. This function provides you
the `context` parameter that is of the type `InvocationContext` which will give
you access to the current context of the generation and all the other services
that are provided by it by InvokeAI.

Let us create this function first.

```python
from typing import Literal, Union
from pydantic import Field

from .baseinvocation import BaseInvocation, InvocationContext
from ..models.image import ImageField

class ResizeInvocation(BaseInvocation):
    '''Resizes an image'''
    type: Literal['resize'] = 'resize'

    # Inputs
    image: Union[ImageField, None] = Field(description="The input image", default=None)
    width: int = Field(default=512, ge=64, le=2048, description="Width of the new image")
    height: int = Field(default=512, ge=64, le=2048, description="Height of the new image")

    def invoke(self, context: InvocationContext):
        pass
```

### **Outputs**

The output of our Invocation will be whatever is returned by this `invoke`
function. Like with our inputs, we need to strongly type and define our outputs
too.

What is our output going to be? Another image. Normally you'd have to create a
type for this but InvokeAI already offers you an `ImageOutput` type that handles
all the necessary info related to image outputs. So let us use that.

We will cover how to create your own output types later in this guide.

```python
from typing import Literal, Union
from pydantic import Field

from .baseinvocation import BaseInvocation, InvocationContext
from ..models.image import ImageField
from .image import ImageOutput

class ResizeInvocation(BaseInvocation):
    '''Resizes an image'''
    type: Literal['resize'] = 'resize'

    # Inputs
    image: Union[ImageField, None] = Field(description="The input image", default=None)
    width: int = Field(default=512, ge=64, le=2048, description="Width of the new image")
    height: int = Field(default=512, ge=64, le=2048, description="Height of the new image")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        pass
```

Perfect. Now that we have our Invocation setup, let us do what we want to do.

- We will first load the image. Generally we do this using the `PIL` library but
  we can use one of the services provided by InvokeAI to load the image.
- We will resize the image using `PIL` to our input data.
- We will output this image in the format we set above.

So let's do that.

```python
from typing import Literal, Union
from pydantic import Field

from .baseinvocation import BaseInvocation, InvocationContext
from ..models.image import ImageField, ResourceOrigin, ImageCategory
from .image import ImageOutput

class ResizeInvocation(BaseInvocation):
    '''Resizes an image'''
    type: Literal['resize'] = 'resize'

    # Inputs
    image: Union[ImageField, None] = Field(description="The input image", default=None)
    width: int = Field(default=512, ge=64, le=2048, description="Width of the new image")
    height: int = Field(default=512, ge=64, le=2048, description="Height of the new image")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        # Load the image using InvokeAI's predefined Image Service.
        image = context.services.images.get_pil_image(self.image.image_origin, self.image.image_name)

        # Resizing the image
        # Because we used the above service, we already have a PIL image. So we can simply resize.
        resized_image = image.resize((self.width, self.height))

        # Preparing the image for output using InvokeAI's predefined Image Service.
        output_image = context.services.images.create(
            image=resized_image,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
        )

        # Returning the Image
        return ImageOutput(
            image=ImageField(
                image_name=output_image.image_name,
                image_origin=output_image.image_origin,
            ),
            width=output_image.width,
            height=output_image.height,
        )
```

**Note:** Do not be overwhelmed by the `ImageOutput` process. InvokeAI has a
certain way that the images need to be dispatched in order to be stored and read
correctly. In 99% of the cases when dealing with an image output, you can simply
copy-paste the template above.

That's it. You made your own **Resize Invocation**.

## Result

Once you make your Invocation correctly, the rest of the process is fully
automated for you.

When you launch InvokeAI, you can go to `http://localhost:9090/docs` and see
your new Invocation show up there with all the relevant info.

![resize invocation](../assets/contributing/resize_invocation.png)

When you launch the frontend UI, you can go to the Node Editor tab and find your
new Invocation ready to be used.

![resize node editor](../assets/contributing/resize_node_editor.png)

## Contributing Nodes
Once you've created a Node, the next step is to share it with the community! The best way to do this is to submit a Pull Request to add the Node to the [Community Nodes](nodes/communityNodes) list. If you're not sure how to do that, take a look a at our [contributing nodes overview](contributingNodes). 

## Advanced

### Custom Input Fields

Now that you know how to create your own Invocations, let us dive into slightly
more advanced topics.

While creating your own Invocations, you might run into a scenario where the
existing input types in InvokeAI do not meet your requirements. In such cases,
you can create your own input types.

Let us create one as an example. Let us say we want to create a color input
field that represents a color code. But before we start on that here are some
general good practices to keep in mind.

**Good Practices**

- There is no naming convention for input fields but we highly recommend that
  you name it something appropriate like `ColorField`.
- It is not mandatory but it is heavily recommended to add a relevant
  `docstring` to describe your input field.
- Keep your field in the same file as the Invocation that it is made for or in
  another file where it is relevant.

All input types a class that derive from the `BaseModel` type from `pydantic`.
So let's create one.

```python
from pydantic import BaseModel

class ColorField(BaseModel):
    '''A field that holds the rgba values of a color'''
    pass
```

Perfect. Now let us create our custom inputs for our field. This is exactly
similar how you created input fields for your Invocation. All the same rules
apply. Let us create four fields representing the _red(r)_, _blue(b)_,
_green(g)_ and _alpha(a)_ channel of the color.

```python
class ColorField(BaseModel):
    '''A field that holds the rgba values of a color'''
    r: int = Field(ge=0, le=255, description="The red channel")
    g: int = Field(ge=0, le=255, description="The green channel")
    b: int = Field(ge=0, le=255, description="The blue channel")
    a: int = Field(ge=0, le=255, description="The alpha channel")
```

That's it. We now have a new input field type that we can use in our Invocations
like this.

```python
color: ColorField = Field(default=ColorField(r=0, g=0, b=0, a=0), description='Background color of an image')
```

**Extra Config**

All input fields also take an additional `Config` class that you can use to do
various advanced things like setting required parameters and etc.

Let us do that for our _ColorField_ and enforce all the values because we did
not define any defaults for our fields.

```python
class ColorField(BaseModel):
    '''A field that holds the rgba values of a color'''
    r: int = Field(ge=0, le=255, description="The red channel")
    g: int = Field(ge=0, le=255, description="The green channel")
    b: int = Field(ge=0, le=255, description="The blue channel")
    a: int = Field(ge=0, le=255, description="The alpha channel")

    class Config:
        schema_extra = {"required": ["r", "g", "b", "a"]}
```

Now it becomes mandatory for the user to supply all the values required by our
input field.

We will discuss the `Config` class in extra detail later in this guide and how
you can use it to make your Invocations more robust.

### Custom Output Types

Like with custom inputs, sometimes you might find yourself needing custom
outputs that InvokeAI does not provide. We can easily set one up.

Now that you are familiar with Invocations and Inputs, let us use that knowledge
to put together a custom output type for an Invocation that returns _width_,
_height_ and _background_color_ that we need to create a blank image.

- A custom output type is a class that derives from the parent class of
  `BaseInvocationOutput`.
- It is not mandatory but we recommend using names ending with `Output` for
  output types. So we'll call our class `BlankImageOutput`
- It is not mandatory but we highly recommend adding a `docstring` to describe
  what your output type is for.
- Like Invocations, each output type should have a `type` variable that is
  **unique**

Now that we know the basic rules for creating a new output type, let us go ahead
and make it.

```python
from typing import Literal
from pydantic import Field

from .baseinvocation import BaseInvocationOutput

class BlankImageOutput(BaseInvocationOutput):
    '''Base output type for creating a blank image'''
    type: Literal['blank_image_output'] = 'blank_image_output'

    # Inputs
    width: int = Field(description='Width of blank image')
    height: int = Field(description='Height of blank image')
    bg_color: ColorField = Field(description='Background color of blank image')

    class Config:
        schema_extra = {"required": ["type", "width", "height", "bg_color"]}
```

All set. We now have an output type that requires what we need to create a
blank_image. And if you noticed it, we even used the `Config` class to ensure
the fields are required.

### Custom Configuration

As you might have noticed when making inputs and outputs, we used a class called
`Config` from _pydantic_ to further customize them. Because our inputs and
outputs essentially inherit from _pydantic_'s `BaseModel` class, all
[configuration options](https://docs.pydantic.dev/latest/usage/schema/#schema-customization)
that are valid for _pydantic_ classes are also valid for our inputs and outputs.
You can do the same for your Invocations too but InvokeAI makes our life a
little bit easier on that end.

InvokeAI provides a custom configuration class called `InvocationConfig`
particularly for configuring Invocations. This is exactly the same as the raw
`Config` class from _pydantic_ with some extra stuff on top to help faciliate
parsing of the scheme in the frontend UI.

At the current moment, tihs `InvocationConfig` class is further improved with
the following features related the `ui`.

| Config Option | Field Type                                                                                                    | Example                                                                                                               |
| ------------- | ------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| type_hints    | `Dict[str, Literal["integer", "float", "boolean", "string", "enum", "image", "latents", "model", "control"]]` | `type_hint: "model"` provides type hints related to the model like displaying a list of available models              |
| tags          | `List[str]`                                                                                                   | `tags: ['resize', 'image']` will classify your invocation under the tags of resize and image.                         |
| title         | `str`                                                                                                         | `title: 'Resize Image` will rename your to this custom title rather than infer from the name of the Invocation class. |

So let us update your `ResizeInvocation` with some extra configuration and see
how that works.

```python
from typing import Literal, Union
from pydantic import Field

from .baseinvocation import BaseInvocation, InvocationContext, InvocationConfig
from ..models.image import ImageField, ResourceOrigin, ImageCategory
from .image import ImageOutput

class ResizeInvocation(BaseInvocation):
    '''Resizes an image'''
    type: Literal['resize'] = 'resize'

    # Inputs
    image: Union[ImageField, None] = Field(description="The input image", default=None)
    width: int = Field(default=512, ge=64, le=2048, description="Width of the new image")
    height: int = Field(default=512, ge=64, le=2048, description="Height of the new image")

    class Config(InvocationConfig):
        schema_extra: {
            ui: {
                tags: ['resize', 'image'],
                title: ['My Custom Resize']
            }
        }

    def invoke(self, context: InvocationContext) -> ImageOutput:
        # Load the image using InvokeAI's predefined Image Service.
        image = context.services.images.get_pil_image(self.image.image_origin, self.image.image_name)

        # Resizing the image
        # Because we used the above service, we already have a PIL image. So we can simply resize.
        resized_image = image.resize((self.width, self.height))

        # Preparing the image for output using InvokeAI's predefined Image Service.
        output_image = context.services.images.create(
            image=resized_image,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
        )

        # Returning the Image
        return ImageOutput(
            image=ImageField(
                image_name=output_image.image_name,
                image_origin=output_image.image_origin,
            ),
            width=output_image.width,
            height=output_image.height,
        )
```

We now customized our code to let the frontend know that our Invocation falls
under `resize` and `image` categories. So when the user searches for these
particular words, our Invocation will show up too.

We also set a custom title for our Invocation. So instead of being called
`Resize`, it will be called `My Custom Resize`.

As simple as that.

As time goes by, InvokeAI will further improve and add more customizability for
Invocation configuration. We will have more documentation regarding this at a
later time.

# **[TODO]**

### Custom Components For Frontend

Every backend input type should have a corresponding frontend component so the
UI knows what to render when you use a particular field type.

If you are using existing field types, we already have components for those. So
you don't have to worry about creating anything new. But this might not always
be the case. Sometimes you might want to create new field types and have the
frontend UI deal with it in a different way.

This is where we venture into the world of React and Javascript and create our
own new components for our Invocations. Do not fear the world of JS. It's
actually pretty straightforward.

Let us create a new component for our custom color field we created above. When
we use a color field, let us say we want the UI to display a color picker for
the user to pick from rather than entering values. That is what we will build
now.

---

<!-- # OLD -- TO BE DELETED OR MOVED LATER

---

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
<https://github.com/pydantic/pydantic/discussions/4577> -->

