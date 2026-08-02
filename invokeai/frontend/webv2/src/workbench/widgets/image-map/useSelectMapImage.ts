import { galleryImages, legacyGeneratedImageToGalleryItem } from '@features/gallery';
import { useWorkbenchCommands } from '@workbench/WorkbenchContext';
import { useCallback, useRef } from 'react';

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
  const sequenceRef = useRef(0);

  return useCallback(
    (imageName: string) => {
      const sequence = ++sequenceRef.current;

      galleryImages
        .resolveMany([imageName])
        .then((images) => {
          const image = images.at(0);

          if (!image || sequence !== sequenceRef.current) {
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
