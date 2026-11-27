import { ConfirmationAlertDialog, Text } from '@invoke-ai/ui-library';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { buildUseBoolean } from 'common/hooks/useBoolean';
import { useDeleteAllExceptCurrentQueueItem } from 'features/queue/hooks/useDeleteAllExceptCurrentQueueItem';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const [useDeleteAllExceptCurrentQueueItemConfirmationAlertDialog] = buildUseBoolean(false);

export const useDeleteAllExceptCurrentQueueItemDialog = () => {
  const dialog = useDeleteAllExceptCurrentQueueItemConfirmationAlertDialog();
  const deleteAllExceptCurrentQueueItem = useDeleteAllExceptCurrentQueueItem();

  return {
    trigger: deleteAllExceptCurrentQueueItem.trigger,
    isOpen: dialog.isTrue,
    openDialog: dialog.setTrue,
    closeDialog: dialog.setFalse,
    isLoading: deleteAllExceptCurrentQueueItem.isLoading,
    isDisabled: deleteAllExceptCurrentQueueItem.isDisabled,
  };
};

export const DeleteAllExceptCurrentQueueItemConfirmationAlertDialog = memo(() => {
  useAssertSingleton('DeleteAllExceptCurrentQueueItemConfirmationAlertDialog');
  const { t } = useTranslation();
  const deleteAllExceptCurrentQueueItem = useDeleteAllExceptCurrentQueueItemDialog();

  return (
    <ConfirmationAlertDialog
      isOpen={deleteAllExceptCurrentQueueItem.isOpen}
      onClose={deleteAllExceptCurrentQueueItem.closeDialog}
      title={t('queue.cancelAllExceptCurrentTooltip')}
      acceptCallback={deleteAllExceptCurrentQueueItem.trigger}
      acceptButtonText={t('queue.confirm')}
      useInert={false}
    >
      <Text>{t('queue.cancelAllExceptCurrentQueueItemAlertDialog')}</Text>
      <br />
      <Text>{t('queue.cancelAllExceptCurrentQueueItemAlertDialog2')}</Text>
    </ConfirmationAlertDialog>
  );
});

DeleteAllExceptCurrentQueueItemConfirmationAlertDialog.displayName =
  'DeleteAllExceptCurrentQueueItemConfirmationAlertDialog';
