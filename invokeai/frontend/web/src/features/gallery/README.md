# Gallery Overview

The gallery renders a scrollable grid of images. The image sizes adapt to the viewport size, and the user can scroll to any part of their gallery. It supports keyboard navigation, multi-select and context menus. Images can be dragged from the gallery to use them in other parts of the app (they are not removed from the gallery).

There is some basic ordering and searching support.

## Boards

Boards act as folders for images.

- Users can create any number of boards.
- Each image can be assigned to at most one board.
- There is a default "no board" board, labeled "Uncategorized".
- User-created boards can be deleted. The no-board board cannot be deleted.
- When deleting a board, users can choose to either delete all images in the board, or move them to the no-board board.
- User-created boards can be renamed. The no-board board cannot be renamed.
- Boards cannot be nested.
- Boards can be archived, which hides them from the board list.
- There is no way to show all images at once. The gallery view always shows images for a specific board.
- Boards can be selected to show their images in the panel below the boards list; the gallery grid.
- Boards can be set as the "auto-add" board. New images will be added to this board as they are generated.

## Image viewer

Clicking an image in the gallery opens it in the image viewer, which presents a larger view of the image, along with a variety of image actions.

The image viewer is rendered in one of the main/center panel tabs.

### Image actions

A handful of common actions are available as buttons in the image viewer header, matching the context menu actions.

See invokeai/frontend/web/src/features/gallery/components/ContextMenu/README.md

### Progress viewer

During generation, we might get "progress images" showing a low-res version of the image at each step in the denoising process. If these are available, the user can open a progress viewer overlay to see the image at each step.

Socket subscriptions and related logic for handling progress images are in the image viewer context. See invokeai/frontend/web/src/features/gallery/components/ImageViewer/context.tsx

### Metadata viewer

The user can enable a metadata overlay to view the image metadata. This is rendered as a semi-transparent overlay on top of the image.

"Metadata" refers to key-value pairs of various settings. For example, the prompt, number of steps and model used to generate the image. This metadata is embedded into the image file itself, but also stored in the database for searching and filtering.

Images also have the execution graph embedded in them. This isn't stored in the database, as it can be large and complex. Instead, we extract it from the image when needed.

Metadata can be recalled, and the graph can be loaded into the workflow editor.

### Image comparison

Users can hold Alt when click an image in the gallery to select it as the "comparison" image. The comparison image is shown alongside the current image in the image viewer with a couple modes (slider, side-by-side, hover-to-swap).

## Data fetching

The gallery uses a windowed list to only render the images that are currently visible in the viewport.

It starts by loading a list of all image names for the selected board or view settings. react-virtuoso reports on the currently-visible range of images (plus some "overscan"). We then fetch the full image DTOs only for those images, which are cached by RTK Query. As the user scrolls, the visible range changes and we fetch more image DTOs as needed.

This affords a nice UX, where the user can scroll to any part of their gallery. The scrollbar size never changes.

We've tried some other approachs in the past, but they all had significant UX or implementation issues:

### Infinite scroll

Load an initial chunk of images, then load more as the user scrolls to the bottom.

The scrollbar continually shrinks as more images are loaded.

This yields a poor UX, as the user cannot easily scroll to a specific part of their gallery. It's also pretty complicated to implement within RTK Query, though since we switched, RTK Query now supports infinite queries. It might be easier to do this today.

### Traditional pagination

Show a fixed number of images per page, with pagination controls.

This is a poor UX, as the user cannot easily scroll to a specific part of their gallery. Gallerys are often very large, and the page size changes depending on the viewport size. The gallery is also constantly inserting new images at the top of the list, which means we are constanty invalidating the current page's query cache and the page numbers are not stable.
