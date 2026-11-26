import type { components } from '@api/schema';

import { createLogger } from '@platform/logging/logger';
import { type AccountScope, assertAccountScopeCurrent, captureAccountScope } from '@platform/state/accountLifecycle';
import { apiFetchJson, apiFetchRaw, sleep } from '@platform/transport/http';

export interface GalleryArchive {
  blob: Blob;
  fileName: string;
}

const BULK_DOWNLOAD_POLL_INTERVAL_MS = 2000;
const BULK_DOWNLOAD_TIMEOUT_MS = 5 * 60 * 1000;

const downloadLogger = createLogger({ area: 'bulk-download', namespace: 'gallery' });

/** Polls until the server serves the archive; null when the server no longer knows it (failed, fetched, restarted). */
const pollGalleryArchive = async (
  fileName: string,
  owner: AccountScope,
  requestSignal: AbortSignal
): Promise<Blob | null> => {
  const startedAt = Date.now();

  while (Date.now() - startedAt < BULK_DOWNLOAD_TIMEOUT_MS) {
    const response = await apiFetchRaw(`/api/v1/images/download/${encodeURIComponent(fileName)}`, {
      signal: requestSignal,
    });

    assertAccountScopeCurrent(owner);
    requestSignal.throwIfAborted();
    if (response.ok) {
      const blob = await response.blob();

      assertAccountScopeCurrent(owner);
      requestSignal.throwIfAborted();
      return blob;
    }

    if (response.status === 410) {
      return null;
    }

    if (response.status !== 404) {
      throw new Error(`${response.status} ${response.statusText}`);
    }

    await sleep(BULK_DOWNLOAD_POLL_INTERVAL_MS, requestSignal);
    assertAccountScopeCurrent(owner);
    requestSignal.throwIfAborted();
  }

  throw new Error('Timed out preparing the download archive.');
};

export const downloadGalleryArchive = async ({
  boardId,
  imageNames,
  signal,
}: {
  boardId?: string;
  imageNames?: string[];
  signal?: AbortSignal;
}): Promise<GalleryArchive> => {
  const owner = captureAccountScope();
  const requestSignal = signal ? AbortSignal.any([signal, owner.signal]) : owner.signal;
  const { bulk_download_item_name: fileName } = await apiFetchJson<{ bulk_download_item_name?: string | null }>(
    '/api/v1/images/download',
    {
      body: JSON.stringify({ board_id: boardId, image_names: imageNames }),
      method: 'POST',
      signal: requestSignal,
    }
  );

  assertAccountScopeCurrent(owner);
  requestSignal.throwIfAborted();
  if (!fileName) {
    throw new Error('The bulk download failed to start.');
  }

  const blob = await pollGalleryArchive(fileName, owner, requestSignal);

  if (!blob) {
    throw new Error('The download is no longer available.');
  }

  return { blob, fileName };
};

/**
 * Claims and fetches this account's archives whose poller is gone, such as one lost to a session expiry or reload.
 * Claiming renews each archive's lease, so another tab resuming at the same moment does not fetch it too. Each archive
 * goes to `onArchive` as soon as it arrives: the server deleted it on serving, so it must not wait on its siblings.
 */
export const resumeGalleryArchives = async (
  owner: AccountScope,
  onArchive: (archive: GalleryArchive) => void
): Promise<void> => {
  const { bulk_download_item_names: fileNames } = await apiFetchJson<components['schemas']['OrphanedBulkDownloads']>(
    '/api/v1/images/download/claim',
    { method: 'POST', signal: owner.signal }
  );

  assertAccountScopeCurrent(owner);
  // Concurrent polls keep every claimed archive's lease renewed.
  await Promise.all(
    fileNames.map(async (fileName) => {
      try {
        const blob = await pollGalleryArchive(fileName, owner, owner.signal);
        if (blob) {
          onArchive({ blob, fileName });
        }
      } catch (error) {
        assertAccountScopeCurrent(owner);
        downloadLogger.warn({
          context: { fileName },
          error,
          message: 'A resumed bulk download could not be fetched',
          name: 'gallery.bulk-download-resume-failed',
        });
      }
    })
  );
};
