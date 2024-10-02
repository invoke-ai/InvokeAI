import { IconButton } from '@invoke-ai/ui-library';
import { useCancelCurrentQueueItem } from 'features/queue/hooks/useCancelCurrentQueueItem';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiXBold } from 'react-icons/pi';

const CancelCurrentQueueItemIconButton = () => {
  const { t } = useTranslation();
  const { cancelQueueItem, isLoading, isDisabled } = useCancelCurrentQueueItem();

  return (
    <IconButton
      isDisabled={isDisabled}
      isLoading={isLoading}
      aria-label={t('queue.cancel')}
      tooltip={t('queue.cancelTooltip')}
      icon={<PiXBold />}
      onClick={cancelQueueItem}
      colorScheme="error"
    />
  );
};

export default memo(CancelCurrentQueueItemIconButton);
