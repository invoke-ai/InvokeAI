import {
  getGalleryImageRecoveryRevision,
  getRecoveredGalleryImageUrl,
  reportGalleryImageFailure,
  subscribeGalleryImageRecovery,
} from '@features/gallery/data/imageRecovery';
import { useCallback, useMemo, useState, useSyncExternalStore } from 'react';

interface ImageRecoveryState {
  /** The recovery revision current when the image failed; null once a later one reloaded it. */
  failedAt: number | null;
  /** The recovery revision in the image's URL. */
  revision: number;
  url: string;
}

/**
 * The `src` for an image element showing the gallery image at `url`, and its `onError`. A failed image reloads once
 * when the shared probe sees the server serve images again (an image move answers 409 until it ends). The state
 * belongs to `url` alone: an element handed another image, as index-keyed slots are, starts from its plain URL.
 */
export const useGalleryImageRecovery = (url: string): { onError: () => void; src: string } => {
  const recoveryRevision = useSyncExternalStore(
    subscribeGalleryImageRecovery,
    getGalleryImageRecoveryRevision,
    getGalleryImageRecoveryRevision
  );
  const [state, setState] = useState<ImageRecoveryState | null>(null);
  const own = state?.url === url ? state : null;
  let revision = own?.revision ?? 0;

  if (own && own.failedAt !== null && recoveryRevision > own.failedAt) {
    // Pin the reload to the revision that recovered it, so later recoveries of other images leave this one alone.
    revision = recoveryRevision;
    setState({ failedAt: null, revision, url });
  }

  const src = useMemo(() => getRecoveredGalleryImageUrl(url, revision), [revision, url]);
  const onError = useCallback(() => {
    setState({ failedAt: getGalleryImageRecoveryRevision(), revision, url });
    reportGalleryImageFailure(url, revision);
  }, [revision, url]);

  return { onError, src };
};
