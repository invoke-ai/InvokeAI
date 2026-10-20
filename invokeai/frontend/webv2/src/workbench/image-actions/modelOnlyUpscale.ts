import type { ImageWithDims } from '@features/generation/contracts';
import type { SpandrelModelConfig } from '@features/upscale';
import type { AccountScope } from '@platform/state/accountLifecycle';
import type { WorkbenchCommands } from '@workbench/workbenchStore';
import type { TFunction } from 'i18next';

import { galleryImages } from '@features/gallery';
import { ensureGalleryBoardDestinations } from '@features/gallery/queries';
import { queryClient } from '@platform/query/client';
import { isAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { ApiError, getApiErrorMessage } from '@platform/transport/http';

/**
 * The metadata a Model only upscale copies into its result. Null when the image has none (an upload) or the server no
 * longer has it; any other failure throws, so the run is not queued without the metadata it should carry.
 */
export const readUpscaleSourceMetadata = async (
  imageName: string,
  signal?: AbortSignal
): Promise<Record<string, unknown> | null> => {
  try {
    return await galleryImages.metadata(imageName, signal);
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) {
      return null;
    }
    throw error;
  }
};

/**
 * The image menu's one-click upscale: read the image's metadata, then queue the upscaler alone on it. Results go to the
 * gallery's auto-add board, so its listing is read alongside when none is cached, as an Invoke to the gallery does.
 */
export const queueModelOnlyUpscale = async ({
  commands,
  image,
  owner,
  projectId,
  t,
  upscaleModel,
}: {
  commands: Pick<WorkbenchCommands, 'generation' | 'notifications'>;
  image: ImageWithDims;
  owner: AccountScope;
  projectId: string;
  t: TFunction;
  upscaleModel: SpandrelModelConfig;
}): Promise<void> => {
  let sourceMetadata: Record<string, unknown> | null;

  try {
    [sourceMetadata] = await Promise.all([
      readUpscaleSourceMetadata(image.image_name, owner.signal),
      ensureGalleryBoardDestinations(queryClient),
    ]);
  } catch (error) {
    if (isAccountScopeCurrent(owner)) {
      commands.notifications.add({
        kind: 'error',
        message: getApiErrorMessage(error, String(error)),
        title: t('widgets.gallery.itemActions.upscale.metadataFailed', { name: image.image_name }),
      });
    }
    return;
  }

  if (!isAccountScopeCurrent(owner)) {
    return;
  }

  commands.generation.submitModelOnlyUpscale({
    backendSupportsCancellation: true,
    image,
    projectId,
    sourceMetadata,
    upscaleModel,
  });
};
