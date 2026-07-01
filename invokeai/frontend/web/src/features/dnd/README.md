# Drag and drop

Dnd functionality is implemented with https://github.com/atlassian/pragmatic-drag-and-drop, the successor to https://github.com/atlassian/react-beautiful-dnd

It uses the native HTML5 drag and drop API and is very performant, though a bit more involved to set up. The library doesn't expose a react API, but rather a set of utility functions to hook into the drag and drop events.

## Implementation

The core of our implementation is in invokeai/frontend/web/src/features/dnd/dnd.ts

We support dragging and dropping of single or multiple images within the app. We have "dnd source" and "dnd target" abstractions.

A dnd source is is anything that provides the draggable payload/data. Currently, that's either an image DTO or list of image names along with their origin board.

A dnd target is anything that can accept a drop of that payload. Targets have their own data. For example, a target might be a board with a board ID, or a canvas layer with a layer ID.

The library has a concept of draggable elements (dnd sources), droppable elements (dnd targets), and dnd monitors. The monitors are invisible elements that track drag events and provide information about the current drag operation.

The library is a bit to wrap your head around but once you understand the concepts, it's very nice to work with and super flexible.

## Type safety

Native drag events do not have any built-in type safety. We inject a unique symbol into the sources and targets and check that via typeguard functions. This gives us confidence that the payload is what we expect it to be and not some other data that might have been dropped from outside the app or some other source.

## Defining sources and targets

These are strictly typed in the dnd.ts file. Follow the examples there to define new sources and targets.

Targets are more complicated - they get an isValid callback (which is called with the currently-dragged source to determine if it can accept the drop) and a handler callback (which is called when the drop is made).

Both isValid and handler get the source data, target data, and the redux getState/dispatch functions. They can do whatever they need to do to determine if the drop is valid and to handle the drop.

Typically the isValid function just uses the source type guard function, and the handler function dispatches one or more redux actions to update the state.

## Other uses of Dnd

We use the same library for other dnd things:

- When dragging over some tabbed interface, hovering the tab for a moment will switch to it. See invokeai/frontend/web/src/common/hooks/useCallbackOnDragEnter.ts for a hook that implements this functionality.
- Reordering of canvas layer lists. See invokeai/frontend/web/src/features/controlLayers/components/CanvasEntityList/CanvasEntityGroupList.tsx and invokeai/frontend/web/src/features/controlLayers/components/CanvasEntityList/useCanvasEntityListDnd.ts
- Adding node fields to a workflow form builder and restructuring the form. This gets kinda complicated, as the form builder supports arbitrary nesting of containers with stacking of elements. See invokeai/frontend/web/src/features/nodes/components/sidePanel/builder/dnd-hooks.ts
