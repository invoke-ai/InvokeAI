# Image context menu

The context menu is loosely based on https://github.com/lukasbach/chakra-ui-contextmenu.

That library creates a component for _every_ instance of a thing that needed a context menu, which caused perf issues. This implementation uses a singleton pattern instead, with a single component that listens for context menu events and opens the menu as needed.

Images register themselves with the context menu by mapping their DOM element to their image DTO. When a context menu event is fired, we look up the target element in the map (or its parents) to find the image DTO to show the context menu for.

## Image actions

- Recalling common individual metadata fields or all metadata
- Opening the image in the image viewer or new tab
- Copying the image to clipboard
- Downloading the image
- Selecting the image for comparison
- Deleting the image
- Moving the image to a different board
- "Sending" the image to other parts of the app such as canvas
