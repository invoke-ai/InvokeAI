import { galleryImages, type GalleryImage } from '@features/gallery';
import { galleryImageMetadataOptions, galleryKeys } from '@features/gallery/queries';
import { assertAccountScopeCurrent, captureAccountScope } from '@platform/state/accountLifecycle';
import { ApiError } from '@platform/transport/http';
import { useQuery } from '@tanstack/react-query';

import type { ImageActions } from './useImageActions';

import { EMPTY_IMAGE_RECALL_CAPABILITIES, type ImageRecallCapabilities } from './imageRecall';

/**
 * What an image's menus can recall and reuse from it, read when the menu mounts: hosts mount it on open, so a
 * selection change alone requests nothing. A null image (a bulk target) requests nothing. The metadata entry is the
 * gallery's shared one, so Preview Details and recall commands read it once between them. Capabilities derive from
 * the server facts on every render, so a Generate model change re-derives without another request.
 *
 * A failed read offers what the other read and the image itself answer (no metadata, or no workflow); the errored
 * entry holds no answer, so the next open reads it again.
 */
export const useImageRecallCapabilities = (
  actions: Pick<ImageActions, 'deriveImageRecallCapabilities'>,
  image: GalleryImage | null
): { capabilities: ImageRecallCapabilities; isLoading: boolean } => {
  const owner = captureAccountScope();
  const imageName = image?.imageName ?? '';
  const metadataQuery = useQuery({ ...galleryImageMetadataOptions(owner, imageName), enabled: image !== null });
  // Grid listings do not say whether an image embeds a workflow; images from the record endpoints already do.
  const readsRecord = image !== null && image.hasWorkflow === undefined;
  const hasWorkflowQuery = useQuery({
    enabled: readsRecord,
    queryFn: async ({ signal }) => {
      let hasWorkflow: boolean;

      try {
        const record = await galleryImages.resolve(imageName, AbortSignal.any([signal, owner.signal]));
        hasWorkflow = record.hasWorkflow === true;
      } catch (error: unknown) {
        // The server no longer has the image: a final answer, not a failure.
        if (!(error instanceof ApiError && error.status === 404)) {
          throw error;
        }
        hasWorkflow = false;
      }

      assertAccountScopeCurrent(owner);
      return hasWorkflow;
    },
    queryKey: galleryKeys.imageHasWorkflow(owner, imageName),
    // One policy whoever starts the read, as for metadata; whether an image embeds a workflow never changes.
    retry: false,
    staleTime: Infinity,
  });

  if (!image) {
    return { capabilities: EMPTY_IMAGE_RECALL_CAPABILITIES, isLoading: false };
  }

  const isLoading = metadataQuery.isPending || (readsRecord && hasWorkflowQuery.isPending);

  return {
    capabilities: isLoading
      ? EMPTY_IMAGE_RECALL_CAPABILITIES
      : actions.deriveImageRecallCapabilities(
          { ...image, hasWorkflow: readsRecord ? hasWorkflowQuery.data === true : image.hasWorkflow },
          metadataQuery.data ?? null
        ),
    isLoading,
  };
};
