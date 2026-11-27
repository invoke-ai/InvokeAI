import { useScopedAction } from '@platform/react/useScopedAction';
import { assertAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { apiFetch, getApiErrorMessage } from '@platform/transport/http';
import { ConfirmDialog } from '@platform/ui/ConfirmDialog';
import { useNotify } from '@workbench/useNotify';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const DatabaseMaintenanceDialog = ({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) => {
  const { t } = useTranslation();
  const notify = useNotify();
  const { run } = useScopedAction();

  const runVacuum = useCallback(async () => {
    await run(
      async (owner) => {
        await apiFetch('/api/v1/app/database/vacuum', { method: 'POST', signal: owner.signal });
        assertAccountScopeCurrent(owner);
        notify.success(t('settings.databaseMaintenance.completed'));
      },
      (_message, error) =>
        notify.error(
          t('settings.databaseMaintenance.failed'),
          getApiErrorMessage(error, t('settings.databaseMaintenance.requestFailed'))
        )
    );
  }, [notify, run, t]);

  return (
    <ConfirmDialog
      body={t('settings.databaseMaintenance.confirmBody')}
      confirmLabel={t('settings.databaseMaintenance.compactDatabase')}
      isDestructive
      isOpen={isOpen}
      title={t('settings.databaseMaintenance.confirmTitle')}
      onClose={onClose}
      onConfirm={runVacuum}
    />
  );
};
