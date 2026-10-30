import { galleryImages, type GalleryImage, type GalleryImageMetadata } from '@features/gallery';
import { galleryKeys } from '@features/gallery/queries';
import { assertAccountScopeCurrent, captureAccountScope } from '@platform/state/accountLifecycle';
import { ApiError } from '@platform/transport/http';
import { useQuery } from '@tanstack/react-query';

import type { ImageActions } from './useImageActions';

import { EMPTY_IMAGE_RECALL_CAPABILITIES, type ImageRecallCapabilities } from './imageRecall';

interface ImageRecallSources {
  hasWorkflow: boolean;
  /** Null when the image records none, or no longer exists: recall then offers what the image itself carries. */
  metadata: GalleryImageMetadata | null;
  /** A read failed for a reason other than a missing image; the answer is a stand-in, read again on next open. */
  isProvisional: boolean;
}

interface ReadResult<T> {
  isProvisional: boolean;
  value: T;
}

/**
 * Degrade a failed read to `fallback` so one failure does not hide what the other read answered. The backend answers
 * 404 for an image it no longer has, a final answer; any other failure is provisional. An abort rethrows.
 */
const readOr = async <T>(read: Promise<T>, fallback: T, signal: AbortSignal): Promise<ReadResult<T>> => {
  try {
    return { isProvisional: false, value: await read };
  } catch (error: unknown) {
    signal.throwIfAborted();
    return { isProvisional: !(error instanceof ApiError && error.status === 404), value: fallback };
  }
};

/**
 * What an image's menus can recall and reuse from it, read when the menu mounts: hosts mount it on open, so a
 * selection change alone requests nothing. A null image (a bulk target) requests nothing. The query holds only the
 * image's server facts, keyed by account and image name; capabilities derive from them on every render, so a Generate
 * model change re-derives without another request.
 */
export const useImageRecallCapabilities = (
  actions: Pick<ImageActions, 'deriveImageRecallCapabilities'>,
  image: GalleryImage | null
): { capabilities: ImageRecallCapabilities; isLoading: boolean } => {
  const owner = captureAccountScope();
  const sourcesQuery = useQuery({
    enabled: image !== null,
    queryKey: galleryKeys.imageMenuRecallSources(owner, image?.imageName ?? null),
    queryFn: async ({ signal }): Promise<ImageRecallSources> => {
      if (!image) {
        return { hasWorkflow: false, isProvisional: false, metadata: null };
      }

      const requestSignal = AbortSignal.any([signal, owner.signal]);
      const [metadata, hasWorkflow] = await Promise.all([
        readOr(galleryImages.metadata(image.imageName, requestSignal), null, requestSignal),
        // Grid listings do not say whether an image embeds a workflow; images from the record endpoints already do.
        // A failed record read offers no workflow, provisionally, the same way a failed metadata read offers none.
        image.hasWorkflow === undefined
          ? readOr(
              galleryImages.resolve(image.imageName, requestSignal).then((record) => record.hasWorkflow === true),
              false,
              requestSignal
            )
          : { isProvisional: false, value: image.hasWorkflow },
      ]);

      assertAccountScopeCurrent(owner);
      return {
        hasWorkflow: hasWorkflow.value,
        isProvisional: metadata.isProvisional || hasWorkflow.isProvisional,
        metadata: metadata.value,
      };
    },
    // An image's metadata and embedded workflow never change, so a final answer is never read again; a provisional one
    // is read again the next time a menu opens.
    staleTime: (query) => (query.state.data?.isProvisional === false ? Infinity : 0),
  });

  if (!image) {
    return { capabilities: EMPTY_IMAGE_RECALL_CAPABILITIES, isLoading: false };
  }

  const sources = sourcesQuery.data;

  return {
    capabilities: sources
      ? actions.deriveImageRecallCapabilities({ ...image, hasWorkflow: sources.hasWorkflow }, sources.metadata)
      : EMPTY_IMAGE_RECALL_CAPABILITIES,
    isLoading: sourcesQuery.isPending,
  };
};
