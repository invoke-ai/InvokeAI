import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { FaTimes } from 'react-icons/fa';
import { useCancelCurrentQueueItem } from '../hooks/useCancelCurrentQueueItem';
import QueueButton from './common/QueueButton';

type Props = {
  asIconButton?: boolean;
};

const CancelCurrentQueueItemButton = ({ asIconButton }: Props) => {
  const { t } = useTranslation();
  const { cancelQueueItem, isLoading, currentQueueItemId } =
    useCancelCurrentQueueItem();

  return (
    <QueueButton
      isDisabled={!currentQueueItemId}
      isLoading={isLoading}
      asIconButton={asIconButton}
      label={t('queue.cancel')}
      tooltip={t('queue.cancelTooltip')}
      icon={<FaTimes />}
      onClick={cancelQueueItem}
      colorScheme="error"
    />
  );
};

export default memo(CancelCurrentQueueItemButton);
