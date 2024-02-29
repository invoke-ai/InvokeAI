# Invoke v4.0.0 Nodes API Migration guide

Invoke v4.0.0 is versioned as such due to breaking changes to the API utilized
by nodes, both core and custom.

## Motivation

Prior to v4.0.0, the `invokeai` python package has not be set up to be utilized
as a library. That is to say, it didn't have any explicitly public API, and node
authors had to work with the unstable internal application API.

v4.0.0 introduces a stable public API for nodes.

## Changes

There are two node-author-facing changes:

1. Import Paths
1. Invocation Context API

### Import Paths

All public objects are now exported from `invokeai.invocation_api`:

```py
# Old
from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    InputField,
    InvocationContext,
    invocation,
)
from invokeai.app.invocations.primitives import ImageField

# New
from invokeai.invocation_api import (
    BaseInvocation,
    ImageField,
    InputField,
    InvocationContext,
    invocation,
)
```

It's possible that we've missed some classes you need in your node. Please let
us know if that's the case.

### Invocation Context API

Most nodes utilize the Invocation Context, an object that is passed to the
`invoke` that provides access to data and services a node may need.

Until now, that object and the services it exposed were internal. Exposing them
to nodes means that changes to our internal implementation could break nodes.
The methods on the services are also often fairly complicated and allowed nodes
to footgun.

In v4.0.0, this object has been refactored to be much simpler.

See [INVOCATION_API](./INVOCATION_API.md) for full details of the API.

!!! warning ""

    This API may shift slightly until the release of v4.0.0 as we work through a few final updates to the Model Manager.

#### Improved Service Methods

The biggest offender was the image save method:

```py
# Old
image_dto = context.services.images.create(
    image=image,
    image_origin=ResourceOrigin.INTERNAL,
    image_category=ImageCategory.GENERAL,
    node_id=self.id,
    session_id=context.graph_execution_state_id,
    is_intermediate=self.is_intermediate,
    metadata=self.metadata,
    workflow=context.workflow,
)

# New
image_dto = context.images.save(image=image)
```

Other methods are simplified, or enhanced with additional functionality:

```py
# Old
image = context.services.images.get_pil_image(image_name)

# New
image = context.images.get_pil(image_name)
image_cmyk = context.images.get_pil(image_name, "CMYK")
```

We also had some typing issues around tensors:

```py
# Old
# `latents` typed as `torch.Tensor`, but could be `ConditioningFieldData`
latents = context.services.latents.get(self.latents.latents_name)
# `data` typed as `torch.Tenssor,` but could be `ConditioningFieldData`
context.services.latents.save(latents_name, data)

# New - separate methods for tensors and conditioning data w/ correct typing
# Also, the service generates the names
tensor_name = context.tensors.save(tensor)
tensor = context.tensors.load(tensor_name)
# For conditioning
cond_name = context.conditioning.save(cond_data)
cond_data = context.conditioning.load(cond_name)
```

#### Output Construction

Core Outputs have builder functions right on them - no need to manually
construct these objects, or use an extra utility:

```py
# Old
image_output = ImageOutput(
    image=ImageField(image_name=image_dto.image_name),
    width=image_dto.width,
    height=image_dto.height,
)
latents_output = build_latents_output(latents_name=name, latents=latents, seed=None)
noise_output = NoiseOutput(
    noise=LatentsField(latents_name=latents_name, seed=seed),
    width=latents.size()[3] * 8,
    height=latents.size()[2] * 8,
)
cond_output = ConditioningOutput(
    conditioning=ConditioningField(
        conditioning_name=conditioning_name,
    ),
)

# New
image_output = ImageOutput.build(image_dto)
latents_output = LatentsOutput.build(latents_name=name, latents=noise, seed=self.seed)
noise_output = NoiseOutput.build(latents_name=name, latents=noise, seed=self.seed)
cond_output = ConditioningOutput.build(conditioning_name)
```

You can still create the objects using constructors if you want, but we suggest
using the builder methods.
