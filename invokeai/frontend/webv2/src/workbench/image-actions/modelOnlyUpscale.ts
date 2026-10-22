import type { ImageWithDims } from '@features/generation/contracts';
import type { SpandrelModelConfig } from '@features/upscale';
import type { AccountScope } from '@platform/state/accountLifecycle';
import type { WorkbenchCommands } from '@workbench/workbenchStore';
import type { TFunction } from 'i18next';

import { ensureGalleryBoardDestinations } from '@features/gallery/queries';
import { readUpscaleSourceMetadata } from '@features/upscale/sourceMetadata';
import { queryClient } from '@platform/query/client';
import { isAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { getApiErrorMessage } from '@platform/transport/http';

/**
 * The image menu's one-click upscale: read the image's metadata, then queue the upscaler alone on it at `scale` (`'native'`,
 * or another of `getModelOnlyUpscaleFactors`). Results go to the gallery's auto-add board, so its listing is read alongside when none
 * is cached, as an Invoke to the gallery does.
 */
export const queueModelOnlyUpscale = async ({
  commands,
  image,
  owner,
  projectId,
  scale,
  t,
  upscaleModel,
}: {
  commands: Pick<WorkbenchCommands, 'generation' | 'notifications'>;
  image: ImageWithDims;
  owner: AccountScope;
  projectId: string;
  scale: number | 'native';
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
    scale,
    sourceMetadata,
    upscaleModel,
  });
};
