import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { BsStars } from 'react-icons/bs';
import { usePruneQueue } from '../hooks/usePruneQueue';
import QueueButton from './common/QueueButton';

type Props = {
  asIconButton?: boolean;
};

const PruneQueueButton = ({ asIconButton }: Props) => {
  const { t } = useTranslation();
  const { pruneQueue, isLoading, finishedCount, isDisabled } = usePruneQueue();

  return (
    <QueueButton
      isDisabled={isDisabled}
      isLoading={isLoading}
      asIconButton={asIconButton}
      label={t('queue.prune')}
      tooltip={t('queue.pruneTooltip', { item_count: finishedCount })}
      icon={<BsStars />}
      onClick={pruneQueue}
      colorScheme="blue"
    />
  );
};

export default memo(PruneQueueButton);
