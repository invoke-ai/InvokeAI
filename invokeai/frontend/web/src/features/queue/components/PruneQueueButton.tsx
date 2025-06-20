import { useFinishedCount, usePruneQueue } from 'features/queue/hooks/usePruneQueue';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiBroomBold } from 'react-icons/pi';

import QueueButton from './common/QueueButton';

type Props = {
  asIconButton?: boolean;
};

const PruneQueueButton = ({ asIconButton }: Props) => {
  const { t } = useTranslation();
  const pruneQueue = usePruneQueue();
  const finishedCount = useFinishedCount();

  return (
    <QueueButton
      onClick={pruneQueue.trigger}
      isDisabled={pruneQueue.isDisabled}
      isLoading={pruneQueue.isLoading}
      asIconButton={asIconButton}
      label={t('queue.prune')}
      tooltip={t('queue.pruneTooltip', { item_count: finishedCount })}
      icon={<PiBroomBold />}
      colorScheme="invokeBlue"
    />
  );
};

export default memo(PruneQueueButton);
