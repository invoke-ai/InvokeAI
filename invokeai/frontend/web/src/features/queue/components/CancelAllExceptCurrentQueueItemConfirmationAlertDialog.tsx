import { ConfirmationAlertDialog, Text } from '@invoke-ai/ui-library';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { buildUseBoolean } from 'common/hooks/useBoolean';
import { useCancelAllExceptCurrentQueueItem } from 'features/queue/hooks/useCancelAllExceptCurrentQueueItem';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const [useCancelAllExceptCurrentQueueItemConfirmationAlertDialog] = buildUseBoolean(false);

export const useCancelAllExceptCurrentQueueItemDialog = () => {
  const dialog = useCancelAllExceptCurrentQueueItemConfirmationAlertDialog();
  const { cancelAllExceptCurrentQueueItem, isLoading, isDisabled, queueStatus } = useCancelAllExceptCurrentQueueItem();

  return {
    cancelAllExceptCurrentQueueItem,
    isOpen: dialog.isTrue,
    openDialog: dialog.setTrue,
    closeDialog: dialog.setFalse,
    isLoading,
    queueStatus,
    isDisabled,
  };
};

export const CancelAllExceptCurrentQueueItemConfirmationAlertDialog = memo(() => {
  useAssertSingleton('CancelAllExceptCurrentQueueItemConfirmationAlertDialog');
  const { t } = useTranslation();
  const cancelAllExceptCurrentQueueItem = useCancelAllExceptCurrentQueueItemDialog();

  return (
    <ConfirmationAlertDialog
      isOpen={cancelAllExceptCurrentQueueItem.isOpen}
      onClose={cancelAllExceptCurrentQueueItem.closeDialog}
      title={t('queue.cancelAllExceptCurrentTooltip')}
      acceptCallback={cancelAllExceptCurrentQueueItem.cancelAllExceptCurrentQueueItem}
      acceptButtonText={t('queue.confirm')}
      useInert={false}
    >
      <Text>{t('queue.cancelAllExceptCurrentQueueItemAlertDialog')}</Text>
      <br />
      <Text>{t('queue.cancelAllExceptCurrentQueueItemAlertDialog2')}</Text>
    </ConfirmationAlertDialog>
  );
});

CancelAllExceptCurrentQueueItemConfirmationAlertDialog.displayName =
  'CancelAllExceptCurrentQueueItemConfirmationAlertDialog';
