import { IconButton, useShiftModifier } from '@invoke-ai/ui-library';
import { useCancelAllExceptCurrentQueueItemDialog } from 'features/queue/components/CancelAllExceptCurrentQueueItemConfirmationAlertDialog';
import { useCancelCurrentQueueItem } from 'features/queue/hooks/useCancelCurrentQueueItem';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiXBold, PiXCircle } from 'react-icons/pi';

export const ClearQueueIconButton = memo(() => {
  const shift = useShiftModifier();

  if (!shift) {
    return <CancelCurrentIconButton />;
  }

  return <CancelAllExceptCurrentIconButton />;
});

ClearQueueIconButton.displayName = 'ClearQueueIconButton';

const CancelCurrentIconButton = memo(() => {
  const { t } = useTranslation();
  const cancelCurrentQueueItem = useCancelCurrentQueueItem();

  return (
    <IconButton
      size="lg"
      isDisabled={cancelCurrentQueueItem.isDisabled}
      isLoading={cancelCurrentQueueItem.isLoading}
      aria-label={t('queue.cancel')}
      tooltip={t('queue.cancelTooltip')}
      icon={<PiXBold />}
      colorScheme="error"
      onClick={cancelCurrentQueueItem.cancelQueueItem}
    />
  );
});

CancelCurrentIconButton.displayName = 'CancelCurrentIconButton';

const CancelAllExceptCurrentIconButton = memo(() => {
  const { t } = useTranslation();
  const cancelAllExceptCurrent = useCancelAllExceptCurrentQueueItemDialog();

  return (
    <IconButton
      size="lg"
      isDisabled={cancelAllExceptCurrent.isDisabled}
      isLoading={cancelAllExceptCurrent.isLoading}
      aria-label={t('queue.clear')}
      tooltip={t('queue.cancelAllExceptCurrentTooltip')}
      icon={<PiXCircle />}
      colorScheme="error"
      onClick={cancelAllExceptCurrent.openDialog}
    />
  );
});

CancelAllExceptCurrentIconButton.displayName = 'CancelAllExceptCurrentIconButton';
