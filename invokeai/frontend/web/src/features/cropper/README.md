# Image cropper

This is a simple image cropping canvas app built with KonvaJS ("native" Konva, _not_ the react bindings).

The editor implementation is here: invokeai/frontend/web/src/features/cropper/lib/editor.ts

It is rendered in a modal.

Currently, the crop functionality is only exposed for reference images. These are the kind of images that most often need cropping (i.e. for FLUX Kontext, which is sensitive to the size/aspect ratio of its ref images). All ref image state is enriched to include a ref to the original image, the cropped image, and the crop attributes.

The functionality could be extended to all images in the future, but there are some questions around whether we consider gallery images immutable. If so, we can't crop them in place. Do we instead add a new cropped image to the gallery? Or do we add a field to the image metadata that points to a cropped version of the image?
