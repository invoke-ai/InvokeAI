import { refreshInstalls } from '@features/models/data/installsStore';
import { ManagerQueueTable } from '@platform/ui/ManagerQueue';
import { DownloadIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

/* eslint-disable react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { InstallQueueRow } from './queueModel';

import { InstallJobRow } from './InstallJobRow';

export const InstallQueueTable = ({
  error,
  rows,
  status,
}: {
  error: string | null;
  rows: InstallQueueRow[];
  status: 'idle' | 'loading' | 'loaded' | 'error';
}) => {
  const { t } = useTranslation();

  return (
    <ManagerQueueTable
      emptyDescription={t('models.noInstallsDescription')}
      emptyIcon={DownloadIcon}
      emptyTitle={t('models.noInstallsYet')}
      isEmpty={rows.length === 0}
      isLoading={status === 'loading' || status === 'idle'}
      label={t('models.installJobs')}
      loadError={
        status === 'error'
          ? { description: error, onRetry: () => void refreshInstalls(), title: t('models.couldNotLoadInstallQueue') }
          : null
      }
    >
      {rows.map((row) => (
        <InstallJobRow key={row.job.id} row={row} />
      ))}
    </ManagerQueueTable>
  );
};
