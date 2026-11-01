import type { ModelInstallJob } from '@features/models/core/types';

import { formatBytes } from '@platform/i18n/languages';
import { ConfirmDialog } from '@platform/ui/ConfirmDialog';
import { useCallback, useState, type ReactNode } from 'react';
import { useTranslation } from 'react-i18next';

import type { InstallByteProgress } from './queueModel';

import { getInstallJobDisplayName } from './queueModel';

/**
 * Cancelling throws away what a download has fetched, so once it has real progress the cancel asks once, naming how
 * much is lost. A download that has fetched nothing (queued, just started) cancels at once: there is nothing to lose.
 */
export const useConfirmedInstallCancel = (
  job: ModelInstallJob,
  /** The row's resolved progress; the caller already follows the live bytes. */
  progress: InstallByteProgress,
  cancel: () => void
): { dialog: ReactNode; requestCancel: () => void } => {
  const { t } = useTranslation();
  const [isOpen, setIsOpen] = useState(false);

  const requestCancel = useCallback(() => {
    if (progress.bytes > 0) {
      setIsOpen(true);
      return;
    }

    cancel();
  }, [cancel, progress.bytes]);
  const close = useCallback(() => setIsOpen(false), []);

  const downloaded = formatBytes(progress.bytes);
  const dialog = (
    <ConfirmDialog
      body={
        progress.totalBytes > 0
          ? t('models.cancelDownloadBody', {
              downloaded,
              name: getInstallJobDisplayName(job),
              total: formatBytes(progress.totalBytes),
            })
          : t('models.cancelDownloadBodyUnknownTotal', { downloaded, name: getInstallJobDisplayName(job) })
      }
      cancelLabel={t('models.keepDownloading')}
      confirmLabel={t('models.cancelDownload')}
      isOpen={isOpen}
      title={t('models.cancelDownloadQuestion')}
      onClose={close}
      onConfirm={cancel}
    />
  );

  return { dialog, requestCancel };
};
