import { galleryImages, legacyGeneratedImageToGalleryItem, toGalleryItemKey } from '@features/gallery';
import { useWorkbenchCommands } from '@workbench/WorkbenchContext';
import { useCallback, useMemo } from 'react';

export interface MapSelectionActions {
  /** Select one image (hydrates the item, then gallery.selectItem). */
  selectImage: (imageName: string) => void;
  /** Multi-select a cluster with the clicked image as primary. */
  selectCluster: (primaryImageName: string, imageNames: string[]) => void;
}

/**
 * Module-scoped, deliberately: the thing being guarded — the gallery selection —
 * is global, so the counter has to be too. A `useRef` is per-mount, which leaves
 * a hole whenever the widget unmounts with a hydrate in flight (switching the
 * right-panel tab away and back): the abandoned closure compares against its own
 * dead ref, still passes, and overwrites the newer mount's selection.
 */
let selectionSequence = 0;

/**
 * Turns map clicks into gallery selections. The map only knows names; the
 * selection contract wants a full gallery item, so names are hydrated through
 * the bulk by-names resolver — always fresh, since a cached DTO's star/board
 * state can drift. ONE monotonic sequence spans both selection kinds, so
 * rapid clicks always resolve to the latest click regardless of which mode
 * each went through; a slow fetch can never overwrite a newer selection.
 * Preview follows the gallery selection on its own.
 */
export const useMapSelection = (): MapSelectionActions => {
  const commands = useWorkbenchCommands();

  const selectImage = useCallback(
    (imageName: string) => {
      const sequence = ++selectionSequence;

      galleryImages
        .resolveMany([imageName])
        .then((images) => {
          const image = images.at(0);

          if (!image || sequence !== selectionSequence) {
            return;
          }

          // Select the image's board first. The map spans every accessible
          // board, but `selectGalleryItem` stamps `selectedImageQuery` from
          // whatever list the gallery is CURRENTLY showing — it takes the
          // selection on faith as "an item from the visible page". Selecting a
          // point from another board without this left the stored query
          // describing a list the image was never in, so Preview's next/prev
          // found no cursor and went dead until the user re-selected from the
          // grid. Mirrors the command palette's reveal-in-gallery.
          //
          // This lands the gallery on page 0 of that board (selectGalleryBoard
          // resets the page), so next/prev works whenever the image is on the
          // first page. Placing it exactly would need the board's ordering,
          // which only the gallery query knows — Preview does it that way, from
          // pages it has already loaded. The query is at least coherent now:
          // it describes a real list, and any grid selection re-anchors it.
          commands.gallery.selectBoard(image.boardId);
          commands.gallery.selectItem(legacyGeneratedImageToGalleryItem(image));
        })
        .catch(() => {
          // A click on a just-deleted image, or a blip mid-backend-restart,
          // simply leaves the selection unchanged.
        });
    },
    [commands]
  );

  const selectCluster = useCallback(
    (primaryImageName: string, imageNames: string[]) => {
      const sequence = ++selectionSequence;

      galleryImages
        .resolveMany([primaryImageName])
        .then((images) => {
          const image = images.at(0);

          if (!image || sequence !== selectionSequence) {
            return;
          }

          // Map points are always images, so every key is kind-tagged 'image'.
          const itemKeys = imageNames.map((name) => toGalleryItemKey({ kind: 'image', name }));
          commands.gallery.setItemMultiSelection(itemKeys, legacyGeneratedImageToGalleryItem(image));
        })
        .catch(() => {
          // Selection is simply left unchanged on hydrate failure.
        });
    },
    [commands]
  );

  return useMemo(() => ({ selectCluster, selectImage }), [selectCluster, selectImage]);
};
