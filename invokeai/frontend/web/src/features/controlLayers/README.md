# Canvas

The canvas is a fairly complex feature. It uses "native" KonvaJS (i.e. not the Konva react bindings) to render a drawing canvas.

It supports layers, drawing, erasing, undo/redo, exporting, backend filters (i.e. filters that require sending image data to teh backend to process) and frontend filters.

## Broad Strokes of Design

The canvas is internally is a hierarchy of classes (modules). All canvas modules inherit from invokeai/frontend/web/src/features/controlLayers/konva/CanvasModuleBase.ts

### Modules

The top-level module is the CanvasManager: invokeai/frontend/web/src/features/controlLayers/konva/CanvasManager.ts

All canvas modules have:

- A unique id (per instance)
- A ref to its parent module and the canvas manager (the top-leve Manager refs itself)
- A repr() method that returns a plain JS object representing the module instance
- A destroy() method to clean up resources
- A log() method that auto-injects context for the module instanc)

Modules can do anything, they are simply plain-JS classes to encapsulate some functionality. Some are singletons. Some examples:

- A singleton module that handles tool-specific interactions: invokeai/frontend/web/src/features/controlLayers/konva/CanvasTool/CanvasToolModule.ts
- Singleton models for each tool e.g. the CanvasBrushToolModule: invokeai/frontend/web/src/features/controlLayers/konva/CanvasTool/CanvasBrushToolModule.ts
- A singleton module to render the background of the canvas: invokeai/frontend/web/src/features/controlLayers/konva/CanvasBackgroundModule.ts
- A strictly logical module that manages various caches of image data: invokeai/frontend/web/src/features/controlLayers/konva/CanvasCacheModule.ts
- A non-singleton module that handles rendering a brush stroke: invokeai/frontend/web/src/features/controlLayers/konva/CanvasObject/CanvasObjectBrushLine.ts

### Layers (Entities) and Adapters modules

Canvas has a number of layer types:

- Raster layers: Traditional raster/pixel layers, much like layers in Photoshop
- Control layers: Internally a raster layer, but designated to hold control data (e.g. depth maps, segmentation masks, etc.) and have special rendering rules
- Regional guidance layers: A mask-like layer (i.e. it has arbitrary shapes but they have no color or texture, it's just a mask region) plus conditioning data like prompts or ref images. The conditioning is applied only to the masked regions
- Inpaint mask layers: Another mask-like layer that indicate regions to inpaint/regenerate

Instances of layers are called "entities" in the codebase. Each entity has a type (one of the above), a number of properties (e.g. visibility, opacity, etc.), objects (e.g. brush strokes, shapes, images) and possibly other data.

Each layer type has a corresponding "adapter" module that handles rendering the layer and its objects, applying filters, etc. The adapter modules are non-singleton modules that are instantiated once per layer entity.

Using the raster layer type as an example, it has a number of sub-modules:

- A top-level module that coordinates everything: invokeai/frontend/web/src/features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterRasterLayer.ts
- An object (e.g. brush strokes, shapes, images) renderer that draws the layer via Konva: invokeai/frontend/web/src/features/controlLayers/konva/CanvasEntity/CanvasEntityObjectRenderer.ts
- A "buffer" object renderer, which renders in-progress objects (e.g. a brush stroke that is being drawn but not yet committed, important for performance): invokeai/frontend/web/src/features/controlLayers/konva/CanvasEntity/CanvasEntityBufferObjectRenderer.ts
- A module that handles previewing and applying backend filters: invokeai/frontend/web/src/features/controlLayers/konva/CanvasEntity/CanvasEntityFilterer.ts
- A module that handles selecting objects from the pixel data of a layer (aka segmentation tasks): invokeai/frontend/web/src/features/controlLayers/konva/CanvasSegmentAnythingModule.ts
- A module that handles transforming the layer (scale, translate, rotate): invokeai/frontend/web/src/features/controlLayers/konva/CanvasEntity/CanvasEntityTransformer.ts

## State mgmt

This gets a bit hairy. We have a mix of redux, Konva and nanostores.

At a high level, we use observable/listener patterns to react to state changes and propagate them to where they need to go.

### Redux

Redux is the source of truth for _persistent_ canvas state - layers, their order, etc.

The redux API includes:

- getState(): Get the entire redux state
- subscribe(listener): Subscribe to state changes, listener is called on _every_ state change, no granularity is provided
- dispatch(action): Dispatch an action to change state

Redux is not suitable for _transient_ state that changes frequently, e.g. the current brush stroke as the user is drawing it. Syncing every change to redux would be too slow and incur a significant performance penalty that would drop FPS too much.

Canvas modules that have persistent state (e.g. layers, their properties, etc.) use redux to store that state and will subscribe to redux to listen for changes and update themselves as needed.

### Konva

Konva's API is imperative (i.e. you call methods on the Konva nodes to change them) but it renders automatically.

There is no simple way to "subscribe" to changes in Konva nodes. You can listen to certain events (e.g. dragmove, transform, etc.) but there is no generic "node changed" event.

So we almost exclusively push data to Konva, we never "read" from it.

### Nanostores

We use https://github.com/nanostores/nanostores as a lightweight observable state management solution. Nanostores has a plain-JS listener API for subscribing to changes, similar to redux's subscribe(). And it has react bindings so we can use it in react components.

Modules often use nanostores to store their internal state, especially when that state needs to be observed by other modules or react components.

For example, the CanvasToolModule uses a nanostore to hold the current tool (brush, eraser, etc.) and its options (brush size, color, etc.). React components can subscribe to that store to update their UI when the tool or its options change.

So this provides a simple two-way binding between canvas modules and react components.

### State -> Canvas

Data may flow from redux state to Canvas. For example, on canvas init we render all layers and their objects from redux state in Konva:

- Create the layer's entity adapter and all sub-modules
- Iterate over the layer's objects and create a module instance for each object (e.g. brush stroke, shape, image)
- Each object module creates the necessary Konva nodes to represent itself and adds them to the layer

The entity adapter subscribes to redux to listen for state changes and pass on the updated state to its sub-modules so they can do whatever they need to do w/ the updated state.

Besides the initial render, we might have to update the Konva representation of a layer when:

- The layer's properties are changed (e.g. visibility, opacity, etc.)
- The layer's order is changed (e.g. move up/down)
- User does an undo/redo operation that affects the layer
- The layer is deleted

### Canvas -> State

When the user interacts w/ the canvas (e.g. draws a brush stroke, erases, moves an object, etc.), we create/update/delete objects in Konva. When the user finishes the interaction (e.g. finishes drawing a brush stroke), we serialize the object to a plain JS object and dispatch a redux action to add the object in redux state.

Using drawing a line on a raster layer as an example, the flow is:

- User initiates a brush stroke and draws
- We create a brush line object module instance in the layer's buffer renderer
- The brush line object is given a unique ID
- The brush line mod creates a Konva.Line node to represent the stroke
- The brush line mod tracks the stroke as the user draws, updating the Konva.Line node as needed, all in the buffer renderer
- When the user finishes the stroke, the brush line module transfers control of itself from the layer's buffer renderer to its main renderer
- As the line is marked complete, the line data is serialized to a plain JS object (i.e. array of points and color) and we dispatch a redux action to add the line object to the layer entity in redux state

Besides drawing tasks, we have similar flows for:

- Transforming a layer (scale, translate, rotate)
- Filtering a layer
- Selecting objects from a layer (segmentation tasks)

## Erasing is hard

HTML Canvas has a limited set of compositing modes. These apply globally to the whole canvas element. There is no "local" compositing mode that applies only to a specific shape or object. There is no concept of layers.

So to implement erasing (and opacity!), we have to get creative. Konva handles much of this for us. Each layer is represented internally by a Konva.Layer, which in turn is drawn to its own HTML Canvas element.

Erasing is accomplished by using a globalCompositeOperation of "destination-out" on the brush stroke that is doing the erasing. The brush stroke "cuts a hole" in the layer it is drawn on.

There is a complication. The UX for erasing a layer should be:

- User has a layer, let's say it has an image on it
- The layer's size is exactly the size of the image
- User erases the right-hand half of the image
- The layer's size shrinks to fit the remaining content, i.e. the left half of the image
- If the user transforms the layer (scale, translate, rotate), the transformations apply only to the remaining content

But the "destination-out" compositing mode only makes the erased pixels transparent. It does not actually remove them from the layer. The layer's bounding box includes the eraser strokes - even though they are transparent. The eraser strokes can actually _enlarge_ the layer's bounding box if the user erases outside the original bounds of the layer.

So, we need a way to calculate the _visual_ bounds of the layer, i.e. the bounding box of all non-transparent pixels. We do this by rendering the layer to an offscreen canvas and reading back the pixel data to calculate the bounds. This process is costly, and we offload some of the work to a web worker to avoid blocking the main thread. Nevertheless, just getting that pixel data is expensive, scaling to the size of the layer.

The usage of the buffer renderer module helps a lot here, as we only need to recalc the bounds when the user finishes a drawing action, not while they are drawing it.

You'll see the relevant code for this in the transformer module. It encapsulates the bounds calculation logic and exposes an observable that holds the last-known visual bounds of the layer.

The worker entrypoint is here invokeai/frontend/web/src/features/controlLayers/konva/CanvasWorkerModule.ts

## Rasterizing layers

Layers consist of a mix of vector and pixel data. For example, a brush stroke is a vector (i.e. array of points) and an image is pixel data.

Ideally we could go straight from user input to pixel data, but this is not feasible for performance reasons. We'd need to write the images to an offscreen canvas, read back the pixel data, send it to the backend, get back the processed pixel data, write it to an offscreen canvas, then read back the pixel data again to update the layer. This would be too slow and block the main thread too much.

So we use a hybrid approach. We keep the vector data in memory and render it to pixel data only when needed, e.g. when the user applies a backend filter or does a transformation on the canvas.

This is unfortunately complicated but we couldn't figure out a more performance way to handle this.

## Compositing layers to prepare for generation

The canvas is a means to an end: provide strong user control and agency for image generation.

When generating an image, the raster layers must be composited toegher into a single image that is sent to the backend. All inpaint masks are similarly composited together into a single mask image. Regional guidance and control layers are not composited together, they are sent as individual images.

This is handled in invokeai/frontend/web/src/features/controlLayers/konva/CanvasCompositorModule.ts

For each compositing task, the compositor creates a unique hash of the layer's state (e.g. objects, properties, etc.) and uses that to cache the resulting composited image's name (which ref a unique ref to the image file stored on disk). This avoids re-compositing layers that haven't changed since the last generation.

## The generation bounding box

Image generation models can only generate images up to certain sizes without causing VRAM OOMs. So we need to give the user a way to specify the size of the generation area. This is done via the "generation bounding box" tool, which is a rectangle that the user can resize and move around the canvas.

Here's the module for it invokeai/frontend/web/src/features/controlLayers/konva/CanvasTool/CanvasBboxToolModule.ts

Models all have width/height constraints - they must be multiples of a certain number (typically 8, 16 or 32). This is related to the internal "latents" representatino of images in diffusion models. So the generation bbox must be constrained to these multiples.

## Staging generations

The typical use pattern for generating images on canvas is to generate a number of variations and pick one or more to keep. This is supported via the "staging area", which is a horizontal strip of image thumbnails below the canvas. These staged images are rendered via React, not Konva.

Once canvas generation starts, much of the canvas is locked down until the user finalizes the staging area, either by accepting a single image, adding one or more images as new layers, or discarding all staged images.

The currently-selected staged image is previewed on the canvas and rendered via invokeai/frontend/web/src/features/controlLayers/konva/CanvasStagingAreaModule.ts

When the user accepts a staged image, it is added as a new raster layer (there are other options for adding as control, saving directly to gallery, etc).

This subsystem tracks generated images by watching the queue of generation tasks. The relevant code for queue tracking is in invokeai/frontend/web/src/features/controlLayers/components/StagingArea/state.ts

## Future enhancements

### Perf: Reduce the number of canvas elements

Each layer has a Konva.Layer which has its own canvas element. Once you get too many of these, the browser starts to struggle.

One idea to improve this would be to have a 3-layer system:

- The active layer is its own Konva.Layer
- All layers behind it are flattened into a single Konva.Layer
- All layers in front of it are flattened into a single Konva.Layer

When the user switches the active layer, we re-flatten the layers as needed. This would reduce the number of canvas elements to 3 regardless of how many layers there are. This would greatly improve performance, especially on lower-end devices.

### Perf: Konva in a web worker

All of the heavy konva rendering could be offloaded to a web worker. This would free up the main thread for user interactions and UI updates. The main thread would send user input and state changes to the worker, and the worker would send back rendered images to display.

There used to be a hacky example of this on the Konva docs but I can't find it as of this writing. It requires proxying mouse and keyboard events to the worker, but wasn't too complicated. This could be a _huge_ perf win.

### Abstract state bindings

Currently the state bindings (redux, nanostores) are all over the place. There is a singleton module that handles much of the redux binding, but it's still a bit messy: invokeai/frontend/web/src/features/controlLayers/konva/CanvasStateApiModule.ts

Many modules still directly subscribe to redux with their own selectors.

Ideally we could have a more abstracted state binding system that could handle multiple backends (e.g. redux, nanostores, etc.) in a more uniform way. This would make it easier to manage state and reduce boilerplate code.

### Do not lock down canvas as much during staging

Currently, once the user starts generating images, much of the canvas is locked down until the user finalizes the staging area. This can be frustrating if the user wants to make small adjustments to layers or settings while previewing staged images, but it prevents footguns.

For example, if the user changes the generation bbox size while staging, then queues up more generations, the output images may not match the bbox size, leading to confusion.

It's more locked-down than it needs to be. Theoretically, most of the canvas could be interactive while staging. Just needs some careful through to not be too confusing.
