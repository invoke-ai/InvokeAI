import { IconButton } from '@invoke-ai/ui-library';
import { useCancelCurrentQueueItem } from 'features/queue/hooks/useCancelCurrentQueueItem';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiXBold } from 'react-icons/pi';

export const CancelCurrentQueueItemIconButton = memo(() => {
  const { t } = useTranslation();
  const cancelCurrentQueueItem = useCancelCurrentQueueItem();

  const cancelCurrentQueueItemWithToast = useCallback(() => {
    cancelCurrentQueueItem.trigger({ withToast: true });
  }, [cancelCurrentQueueItem]);

  return (
    <IconButton
      size="lg"
      onClick={cancelCurrentQueueItemWithToast}
      isDisabled={cancelCurrentQueueItem.isDisabled}
      isLoading={cancelCurrentQueueItem.isLoading}
      aria-label={t('queue.cancel')}
      tooltip={t('queue.cancelTooltip')}
      icon={<PiXBold />}
      colorScheme="error"
    />
  );
});

CancelCurrentQueueItemIconButton.displayName = 'CancelCurrentQueueItemIconButton';
