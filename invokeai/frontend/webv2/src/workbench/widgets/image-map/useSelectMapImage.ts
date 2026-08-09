import { galleryImages, legacyGeneratedImageToGalleryItem } from '@features/gallery';
import { useWorkbenchCommands } from '@workbench/WorkbenchContext';
import { useCallback } from 'react';

/**
 * Module-scoped, deliberately: the thing being guarded — the gallery selection —
 * is global, so the counter has to be too. A `useRef` is per-mount, which leaves
 * a hole whenever the widget unmounts with a hydrate in flight (switching the
 * right-panel tab away and back): the abandoned closure compares against its own
 * dead ref, still passes, and overwrites the newer mount's selection.
 */
let selectionSequence = 0;

/**
 * Turns a map point's image name into the current gallery selection. The map
 * only knows names; the selection contract wants a full gallery item, so names
 * are hydrated through the bulk by-names resolver — always fresh, since a
 * cached DTO's star/board state can drift. A monotonic sequence guards rapid
 * clicks: only the most recent click may dispatch, so a slow fetch can never
 * overwrite a newer selection. Preview follows the gallery selection on its
 * own.
 */
export const useSelectMapImage = (): ((imageName: string) => void) => {
  const commands = useWorkbenchCommands();

  return useCallback(
    (imageName: string) => {
      const sequence = ++selectionSequence;

      galleryImages
        .resolveMany([imageName])
        .then((images) => {
          const image = images.at(0);

          if (!image || sequence !== selectionSequence) {
            return;
          }

          commands.gallery.selectItem(legacyGeneratedImageToGalleryItem(image));
        })
        .catch(() => {
          // A click on a just-deleted image, or a blip mid-backend-restart,
          // simply leaves the selection unchanged.
        });
    },
    [commands]
  );
};
