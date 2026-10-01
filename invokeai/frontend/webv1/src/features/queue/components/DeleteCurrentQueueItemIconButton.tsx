import { IconButton } from '@invoke-ai/ui-library';
import { useDeleteCurrentQueueItem } from 'features/queue/hooks/useDeleteCurrentQueueItem';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiXBold } from 'react-icons/pi';

export const DeleteCurrentQueueItemIconButton = memo(() => {
  const { t } = useTranslation();
  const deleteCurrentQueueItem = useDeleteCurrentQueueItem();

  return (
    <IconButton
      size="lg"
      onClick={deleteCurrentQueueItem.trigger}
      isDisabled={deleteCurrentQueueItem.isDisabled}
      isLoading={deleteCurrentQueueItem.isLoading}
      aria-label={t('queue.cancel')}
      tooltip={t('queue.cancelTooltip')}
      icon={<PiXBold />}
      colorScheme="error"
    />
  );
});

DeleteCurrentQueueItemIconButton.displayName = 'DeleteCurrentQueueItemIconButton';
