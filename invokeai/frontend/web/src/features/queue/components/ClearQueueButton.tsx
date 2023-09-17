import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { FaTrash } from 'react-icons/fa';
import { useClearQueue } from '../hooks/useClearQueue';
import QueueButton from './common/QueueButton';

type Props = {
  asIconButton?: boolean;
};

const ClearQueueButton = ({ asIconButton }: Props) => {
  const { t } = useTranslation();
  const { clearQueue, isLoading, queueStatus } = useClearQueue();

  return (
    <QueueButton
      isDisabled={!queueStatus?.queue.total}
      isLoading={isLoading}
      asIconButton={asIconButton}
      label={t('queue.clear')}
      tooltip={t('queue.clearTooltip')}
      icon={<FaTrash />}
      onClick={clearQueue}
      colorScheme="error"
    />
  );
};

export default memo(ClearQueueButton);
