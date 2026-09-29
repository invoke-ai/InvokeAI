# Delete image modal

When users delete images, we show a confirmation dialog to prevent accidental deletions. Users can opt out of this, but we still check if dleeting an image would screw up their workspace and prompt if so.

For example, if an image is currently set as a field in the workflow editor, we warn the user that deleting it will remove it from the node. We warn them even if they have opted out of the confirmation dialog.

These "image usage" checks are done using redux selectors/util functions. See invokeai/frontend/web/src/features/deleteImageModal/store/state.ts
