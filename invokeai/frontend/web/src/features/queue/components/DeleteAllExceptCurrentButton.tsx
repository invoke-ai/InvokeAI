import type { ButtonProps } from '@invoke-ai/ui-library';
import { Button } from '@invoke-ai/ui-library';
import { useDeleteAllExceptCurrentQueueItemDialog } from 'features/queue/components/DeleteAllExceptCurrentQueueItemConfirmationAlertDialog';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiXCircle } from 'react-icons/pi';

type Props = ButtonProps;

export const DeleteAllExceptCurrentButton = memo((props: Props) => {
  const { t } = useTranslation();
  const deleteAllExceptCurrent = useDeleteAllExceptCurrentQueueItemDialog();

  return (
    <>
      <Button
        onClick={deleteAllExceptCurrent.openDialog}
        isLoading={deleteAllExceptCurrent.isLoading}
        isDisabled={deleteAllExceptCurrent.isDisabled}
        tooltip={t('queue.cancelAllExceptCurrentTooltip')}
        leftIcon={<PiXCircle />}
        colorScheme="error"
        data-testid={t('queue.clear')}
        {...props}
      >
        {t('queue.clear')}
      </Button>
    </>
  );
});

DeleteAllExceptCurrentButton.displayName = 'DeleteAllExceptCurrentButton';
