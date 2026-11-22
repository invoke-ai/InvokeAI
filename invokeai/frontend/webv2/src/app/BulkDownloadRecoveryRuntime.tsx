import { galleryTransfers } from '@features/gallery';
import { downloadBlob } from '@platform/browser/downloadBlob';
import { createLogger } from '@platform/logging/logger';
import { useMountEffect } from '@platform/react/useMountEffect';
import { captureAccountScope, isAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { sleep } from '@platform/transport/http';
import { useNotify } from '@workbench/useNotify';
import i18next from 'i18next';

// A reload or quick sign-in can find the old poller's lease (~10s on the server) not yet lapsed.
const RETRY_DELAY_MS = 15_000;

const logger = createLogger({ area: 'bulk-download', namespace: 'app' });

/**
 * Saves archives prepared for this account while no tab was waiting for them. Runs once per account lifetime; the
 * account scope cancels it, so no unmount cleanup: a StrictMode rerun is harmless, as each archive goes to one claim.
 */
export const BulkDownloadRecoveryRuntime = () => {
  const notify = useNotify();

  useMountEffect(() => {
    const owner = captureAccountScope();
    const recover = async (): Promise<void> => {
      try {
        await galleryTransfers.resumeArchives(owner.signal, ({ blob, fileName }) => {
          if (isAccountScopeCurrent(owner)) {
            void downloadBlob(blob, fileName);
            notify.success(i18next.t('widgets.gallery.recoveredDownload'), fileName);
          }
        });
      } catch (error) {
        if (isAccountScopeCurrent(owner)) {
          logger.debug({ error, message: 'Resuming bulk downloads failed', name: 'app.bulk-download-resume-failed' });
        }
      }
    };

    void recover()
      .then(() => sleep(RETRY_DELAY_MS, owner.signal))
      .then(recover)
      .catch(() => undefined);
  });

  return null;
};
