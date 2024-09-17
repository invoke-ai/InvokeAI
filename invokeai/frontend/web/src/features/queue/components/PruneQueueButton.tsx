import { usePruneQueue } from 'features/queue/hooks/usePruneQueue';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiBroomBold } from 'react-icons/pi';

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
      icon={<PiBroomBold />}
      onClick={pruneQueue}
      colorScheme="invokeBlue"
    />
  );
};

export default memo(PruneQueueButton);
