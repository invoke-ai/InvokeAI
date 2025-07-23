import type { ButtonProps } from '@invoke-ai/ui-library';
import { Button } from '@invoke-ai/ui-library';
import { useCancelAllExceptCurrentQueueItemDialog } from 'features/queue/components/CancelAllExceptCurrentQueueItemConfirmationAlertDialog';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiXCircle } from 'react-icons/pi';

export const CancelAllExceptCurrentButton = memo((props: ButtonProps) => {
  const { t } = useTranslation();
  const api = useCancelAllExceptCurrentQueueItemDialog();

  return (
    <Button
      isDisabled={api.isDisabled}
      isLoading={api.isLoading}
      tooltip={t('queue.cancelAllExceptCurrentTooltip')}
      leftIcon={<PiXCircle />}
      colorScheme="error"
      onClick={api.openDialog}
      {...props}
    >
      {t('queue.cancelAllExceptCurrentTooltip')}
    </Button>
  );
});

CancelAllExceptCurrentButton.displayName = 'CancelAllExceptCurrentButton';
