import type { GalleryImage } from '@features/gallery';

import { useEffect, useRef, useState } from 'react';

import type { ImageActions } from './useImageActions';

import { EMPTY_IMAGE_RECALL_CAPABILITIES, type ImageRecallCapabilities } from './imageRecall';

/** Identify a capability request by the image's stable primitives, so a refreshed listing object does not refetch. */
export const getImageRecallCapabilitiesKey = (image: GalleryImage | null): string | null =>
  image ? `${image.imageName}:${image.width}:${image.height}` : null;

/**
 * What an image's menus can recall and reuse from it, read when the menu mounts: hosts mount it on open, so a
 * selection change alone requests nothing. A null image (a bulk target) requests nothing. The answer also depends on
 * the Generate panel's current model, which `actions.getImageRecallCapabilities` closes over, so a new loader asks
 * again.
 */
export const useImageRecallCapabilities = (
  actions: Pick<ImageActions, 'getImageRecallCapabilities'>,
  image: GalleryImage | null
): { capabilities: ImageRecallCapabilities; isLoading: boolean } => {
  const [capabilities, setCapabilities] = useState<ImageRecallCapabilities>(EMPTY_IMAGE_RECALL_CAPABILITIES);
  const [isLoading, setIsLoading] = useState(false);
  const imageRef = useRef<GalleryImage | null>(null);
  const requestKey = getImageRecallCapabilitiesKey(image);
  const getImageRecallCapabilities = actions.getImageRecallCapabilities;

  // eslint-disable-next-line react/refs
  imageRef.current = image;

  // Moved unchanged from the context menu: the request is keyed by image primitives and the loader's identity, and a
  // superseded answer is dropped. Not a `useQuery` yet: the loader's identity follows the Generate panel's model, so a
  // query key would need that identity (or the model inputs it closes over) to avoid serving stale capabilities.
  useEffect(() => {
    if (!requestKey) {
      return;
    }

    let isCancelled = false;
    const recallImage = imageRef.current;

    if (!recallImage) {
      return;
    }

    setIsLoading((current) => (current ? current : true));
    getImageRecallCapabilities(recallImage)
      .then((next) => {
        if (!isCancelled) {
          setCapabilities(next);
        }
      })
      .catch(() => {
        if (!isCancelled) {
          setCapabilities(EMPTY_IMAGE_RECALL_CAPABILITIES);
        }
      })
      .finally(() => {
        if (!isCancelled) {
          setIsLoading(false);
        }
      });

    return () => {
      isCancelled = true;
    };
  }, [getImageRecallCapabilities, requestKey]);

  return { capabilities, isLoading };
};
