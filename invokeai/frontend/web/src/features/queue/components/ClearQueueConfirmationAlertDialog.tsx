import { ConfirmationAlertDialog, Text } from '@invoke-ai/ui-library';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { buildUseBoolean } from 'common/hooks/useBoolean';
import { useClearQueue } from 'features/queue/hooks/useClearQueue';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const [useClearQueueConfirmationAlertDialog] = buildUseBoolean(false);

const useClearQueueDialog = () => {
  const dialog = useClearQueueConfirmationAlertDialog();
  const clearQueue = useClearQueue();

  return {
    isOpen: dialog.isTrue,
    openDialog: dialog.setTrue,
    closeDialog: dialog.setFalse,
    trigger: clearQueue.trigger,
    isLoading: clearQueue.isLoading,
    isDisabled: clearQueue.isDisabled,
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
      acceptCallback={clearQueue.trigger}
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
