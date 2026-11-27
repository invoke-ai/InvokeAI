import { IconButton } from '@invoke-ai/ui-library';
import { useCancelAllExceptCurrentQueueItemDialog } from 'features/queue/components/CancelAllExceptCurrentQueueItemConfirmationAlertDialog';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiXCircle } from 'react-icons/pi';

export const CancelAllExceptCurrentIconButton = memo(() => {
  const { t } = useTranslation();
  const api = useCancelAllExceptCurrentQueueItemDialog();

  return (
    <IconButton
      size="lg"
      isDisabled={api.isDisabled}
      isLoading={api.isLoading}
      aria-label={t('queue.clear')}
      tooltip={t('queue.cancelAllExceptCurrentTooltip')}
      icon={<PiXCircle />}
      colorScheme="error"
      onClick={api.openDialog}
    />
  );
});

CancelAllExceptCurrentIconButton.displayName = 'CancelAllExceptCurrentIconButton';
