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
- Every Invocation must have a `docstring` that describes what this Invocation
  does.
- While not strictly required, we suggest every invocation class name ends in
  "Invocation", eg "CropImageInvocation".
- Every Invocation must use the `@invocation` decorator to provide its unique
  invocation type. You may also provide its title, tags and category using the
  decorator.
- Invocations are strictly typed. We make use of the native
  [typing](https://docs.python.org/3/library/typing.html) library and the
  installed [pydantic](https://pydantic-docs.helpmanual.io/) library for
  validation.

So let us do that.

```python
from .baseinvocation import BaseInvocation, invocation

@invocation('resize')
class ResizeInvocation(BaseInvocation):
    '''Resizes an image'''
```

That's great.

Now we have setup the base of our new Invocation. Let us think about what inputs
our Invocation takes.

- We need an `image` that we are going to resize.
- We will need new `width` and `height` values to which we need to resize the
  image to.

### **Inputs**

Every Invocation input must be defined using the `InputField` function. This is
a wrapper around the pydantic `Field` function, which handles a few extra things
and provides type hints. Like everything else, this should be strictly typed and
defined.

So let us create these inputs for our Invocation. First up, the `image` input we
need. Generally, we can use standard variable types in Python but InvokeAI
already has a custom `ImageField` type that handles all the stuff that is needed
for image inputs.

But what is this `ImageField` ..? It is a special class type specifically
written to handle how images are dealt with in InvokeAI. We will cover how to
create your own custom field types later in this guide. For now, let's go ahead
and use it.

```python
from .baseinvocation import BaseInvocation, InputField, invocation
from .primitives import ImageField

@invocation('resize')
class ResizeInvocation(BaseInvocation):

    # Inputs
    image: ImageField = InputField(description="The input image")
```

Let us break down our input code.

```python
image: ImageField = InputField(description="The input image")
```

| Part      | Value                                       | Description                                                                     |
| --------- | ------------------------------------------- | ------------------------------------------------------------------------------- |
| Name      | `image`                                     | The variable that will hold our image                                           |
| Type Hint | `ImageField`                                | The types for our field. Indicates that the image must be an `ImageField` type. |
| Field     | `InputField(description="The input image")` | The image variable is an `InputField` which needs a description.                |

Great. Now let us create our other inputs for `width` and `height`

```python
from .baseinvocation import BaseInvocation, InputField, invocation
from .primitives import ImageField

@invocation('resize')
class ResizeInvocation(BaseInvocation):
    '''Resizes an image'''

    # Inputs
    image: ImageField = InputField(description="The input image")
    width: int = InputField(default=512, ge=64, le=2048, description="Width of the new image")
    height: int = InputField(default=512, ge=64, le=2048, description="Height of the new image")
```

As you might have noticed, we added two new arguments to the `InputField`
definition for `width` and `height`, called `gt` and `le`. They stand for
_greater than or equal to_ and _less than or equal to_.

These impose contraints on those fields, and will raise an exception if the
values do not meet the constraints. Field constraints are provided by
**pydantic**, so anything you see in the **pydantic docs** will work.

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
from .baseinvocation import BaseInvocation, InputField, invocation
from .primitives import ImageField

@invocation('resize')
class ResizeInvocation(BaseInvocation):
    '''Resizes an image'''

    # Inputs
    image: ImageField = InputField(description="The input image")
    width: int = InputField(default=512, ge=64, le=2048, description="Width of the new image")
    height: int = InputField(default=512, ge=64, le=2048, description="Height of the new image")

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
from .baseinvocation import BaseInvocation, InputField, invocation
from .primitives import ImageField
from .image import ImageOutput

@invocation('resize')
class ResizeInvocation(BaseInvocation):
    '''Resizes an image'''

    # Inputs
    image: ImageField = InputField(description="The input image")
    width: int = InputField(default=512, ge=64, le=2048, description="Width of the new image")
    height: int = InputField(default=512, ge=64, le=2048, description="Height of the new image")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        pass
```

Perfect. Now that we have our Invocation setup, let us do what we want to do.

- We will first load the image using one of the services provided by InvokeAI to
  load the image.
- We will resize the image using `PIL` to our input data.
- We will output this image in the format we set above.

So let's do that.

```python
from .baseinvocation import BaseInvocation, InputField, invocation
from .primitives import ImageField
from .image import ImageOutput

@invocation("resize")
class ResizeInvocation(BaseInvocation):
    """Resizes an image"""

    image: ImageField = InputField(description="The input image")
    width: int = InputField(default=512, ge=64, le=2048, description="Width of the new image")
    height: int = InputField(default=512, ge=64, le=2048, description="Height of the new image")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        # Load the image using InvokeAI's predefined Image Service. Returns the PIL image.
        image = context.services.images.get_pil_image(self.image.image_name)

        # Resizing the image
        resized_image = image.resize((self.width, self.height))

        # Save the image using InvokeAI's predefined Image Service. Returns the prepared PIL image.
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
            ),
            width=output_image.width,
            height=output_image.height,
        )
```

**Note:** Do not be overwhelmed by the `ImageOutput` process. InvokeAI has a
certain way that the images need to be dispatched in order to be stored and read
correctly. In 99% of the cases when dealing with an image output, you can simply
copy-paste the template above.

### Customization

We can use the `@invocation` decorator to provide some additional info to the
UI, like a custom title, tags and category.

We also encourage providing a version. This must be a
[semver](https://semver.org/) version string ("$MAJOR.$MINOR.$PATCH"). The UI
will let users know if their workflow is using a mismatched version of the node.

```python
@invocation("resize", title="My Resizer", tags=["resize", "image"], category="My Invocations", version="1.0.0")
class ResizeInvocation(BaseInvocation):
    """Resizes an image"""

    image: ImageField = InputField(description="The input image")
    ...
```

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

Once you've created a Node, the next step is to share it with the community! The
best way to do this is to submit a Pull Request to add the Node to the
[Community Nodes](nodes/communityNodes) list. If you're not sure how to do that,
take a look a at our [contributing nodes overview](contributingNodes).

## Advanced

### Custom Output Types

Like with custom inputs, sometimes you might find yourself needing custom
outputs that InvokeAI does not provide. We can easily set one up.

Now that you are familiar with Invocations and Inputs, let us use that knowledge
to create an output that has an `image` field, a `color` field and a `string`
field.

- An invocation output is a class that derives from the parent class of
  `BaseInvocationOutput`.
- All invocation outputs must use the `@invocation_output` decorator to provide
  their unique output type.
- Output fields must use the provided `OutputField` function. This is very
  similar to the `InputField` function described earlier - it's a wrapper around
  `pydantic`'s `Field()`.
- It is not mandatory but we recommend using names ending with `Output` for
  output types.
- It is not mandatory but we highly recommend adding a `docstring` to describe
  what your output type is for.

Now that we know the basic rules for creating a new output type, let us go ahead
and make it.

```python
from .baseinvocation import BaseInvocationOutput, OutputField, invocation_output
from .primitives import ImageField, ColorField

@invocation_output('image_color_string_output')
class ImageColorStringOutput(BaseInvocationOutput):
    '''Base class for nodes that output a single image'''

    image: ImageField = OutputField(description="The image")
    color: ColorField = OutputField(description="The color")
    text: str = OutputField(description="The string")
```

That's all there is to it.

<!-- TODO: DANGER - we probably do not want people to create their own field types, because this requires a lot of work on the frontend to accomodate.

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
-->
