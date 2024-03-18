# Invocation API

Each invocation's `invoke` method is provided a single arg - the Invocation
Context.

This object provides access to various methods, used to interact with the
application. Loading and saving images, logging messages, etc.

!!! warning ""

    This API may shift slightly until the release of v4.0.0 as we work through a few final updates to the Model Manager.

```py
class MyInvocation(BaseInvocation):
  ...
  def invoke(self, context: InvocationContext) -> ImageOutput:
      image_pil = context.images.get_pil(image_name)
      # Do something to the image
      image_dto = context.images.save(image_pil)
      # Log a message
      context.logger.info(f"Did something cool, image saved!")
      ...
```

The full API is documented below.

## Invocation Mixins

Two important mixins are provided to facilitate working with metadata and gallery boards.

### `WithMetadata`

Inherit from this class (in addition to `BaseInvocation`) to add a `metadata` input to your node. When you do this, you can access the metadata dict from `self.metadata` in the `invoke()` function.

The dict will be populated via the node's input, and you can add any metadata you'd like to it. When you call `context.images.save()`, if the metadata dict has any data, it be automatically embedded in the image.

### `WithBoard`

Inherit from this class (in addition to `BaseInvocation`) to add a `board` input to your node. This renders as a drop-down to select a board. The user's selection will be accessible from `self.board` in the `invoke()` function.

When you call `context.images.save()`, if a board was selected, the image will added to that board as it is saved.

<!-- prettier-ignore-start -->
::: invokeai.app.services.shared.invocation_context.InvocationContext
    options:
        members: false

::: invokeai.app.services.shared.invocation_context.ImagesInterface

::: invokeai.app.services.shared.invocation_context.TensorsInterface

::: invokeai.app.services.shared.invocation_context.ConditioningInterface

::: invokeai.app.services.shared.invocation_context.ModelsInterface

::: invokeai.app.services.shared.invocation_context.LoggerInterface

::: invokeai.app.services.shared.invocation_context.ConfigInterface

::: invokeai.app.services.shared.invocation_context.UtilInterface

::: invokeai.app.services.shared.invocation_context.BoardsInterface
<!-- prettier-ignore-end -->
