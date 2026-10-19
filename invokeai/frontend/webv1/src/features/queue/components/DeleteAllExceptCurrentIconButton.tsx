import { IconButton } from '@invoke-ai/ui-library';
import { useDeleteAllExceptCurrentQueueItemDialog } from 'features/queue/components/DeleteAllExceptCurrentQueueItemConfirmationAlertDialog';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiXCircle } from 'react-icons/pi';

export const DeleteAllExceptCurrentIconButton = memo(() => {
  const { t } = useTranslation();
  const deleteAllExceptCurrent = useDeleteAllExceptCurrentQueueItemDialog();

  return (
    <IconButton
      size="lg"
      isDisabled={deleteAllExceptCurrent.isDisabled}
      isLoading={deleteAllExceptCurrent.isLoading}
      aria-label={t('queue.clear')}
      tooltip={t('queue.cancelAllExceptCurrentTooltip')}
      icon={<PiXCircle />}
      colorScheme="error"
      onClick={deleteAllExceptCurrent.openDialog}
    />
  );
});

DeleteAllExceptCurrentIconButton.displayName = 'DeleteAllExceptCurrentIconButton';
