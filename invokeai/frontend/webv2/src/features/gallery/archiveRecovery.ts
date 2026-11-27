import type { AccountScope } from '@platform/state/accountLifecycle';

import { createLogger } from '@platform/logging/logger';
import { isAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { sleep } from '@platform/transport/http';

import { type GalleryArchive, resumeGalleryArchives } from './data/archives';

export type { GalleryArchive };

// A reload or quick sign-in can find the old poller's lease (~10s on the server) not yet lapsed.
const RETRY_DELAY_MS = 15_000;

const logger = createLogger({ area: 'bulk-download', namespace: 'gallery' });

/**
 * Fetches `owner`'s archives that no client is waiting for, now and once more after the old lease lapses. Stops when
 * `owner` ends. Kept apart from the gallery barrel: App starts it before any workbench code loads.
 */
export const recoverGalleryArchives = async (
  owner: AccountScope,
  onArchive: (archive: GalleryArchive) => void
): Promise<void> => {
  const recover = async (): Promise<void> => {
    try {
      await resumeGalleryArchives(owner, onArchive);
    } catch (error) {
      if (isAccountScopeCurrent(owner)) {
        logger.debug({ error, message: 'Resuming bulk downloads failed', name: 'gallery.bulk-download-resume-failed' });
      }
    }
  };

  await recover();
  try {
    await sleep(RETRY_DELAY_MS, owner.signal);
  } catch {
    return;
  }
  await recover();
};
