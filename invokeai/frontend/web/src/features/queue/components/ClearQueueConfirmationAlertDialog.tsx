import { ConfirmationAlertDialog, Text } from '@invoke-ai/ui-library';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { buildUseBoolean } from 'common/hooks/useBoolean';
import { useClearQueue } from 'features/queue/hooks/useClearQueue';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const [useClearQueueConfirmationAlertDialog] = buildUseBoolean(false);

export const useClearQueueDialog = () => {
  const dialog = useClearQueueConfirmationAlertDialog();
  const { clearQueue, isLoading, isDisabled, queueStatus } = useClearQueue();

  return {
    clearQueue,
    isOpen: dialog.isTrue,
    openDialog: dialog.setTrue,
    closeDialog: dialog.setFalse,
    isLoading,
    queueStatus,
    isDisabled,
  };
};

export const ClearQueueConfirmationsAlertDialog = memo(() => {
  useAssertSingleton('ClearQueueConfirmationsAlertDialog');
  const { t } = useTranslation();
  const clearQueue = useClearQueueDialog();

  return (
    <ConfirmationAlertDialog
      isOpen={clearQueue.isOpen}
      onClose={clearQueue.closeDialog}
      title={t('queue.clearTooltip')}
      acceptCallback={clearQueue.clearQueue}
      acceptButtonText={t('queue.clear')}
      useInert={false}
    >
      <Text>{t('queue.clearQueueAlertDialog')}</Text>
      <br />
      <Text>{t('queue.clearQueueAlertDialog2')}</Text>
    </ConfirmationAlertDialog>
  );
});

ClearQueueConfirmationsAlertDialog.displayName = 'ClearQueueConfirmationsAlertDialog';
