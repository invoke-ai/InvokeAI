import { Stack } from '@chakra-ui/react';
import { getPersonalQueueActivity } from '@features/queue/core/types';
import { Scrollable } from '@platform/ui/Scrollable';
import { StatusWidgetChip } from '@platform/ui/StatusWidgetChip';
import { ListOrderedIcon } from 'lucide-react';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';

import type { QueueFilterId } from './queueFilters';

import { CurrentBatchSection } from './NowNextSection';
import { useQueueCounts } from './queueDataStore';
import { QueueFilterTabs } from './QueueFilterTabs';
import { QueueStats } from './QueueStats';
import { usePendingQueueItemReveal } from './queueUiStore';
import { RecentSection } from './RecentSection';

export const QueueWidgetView = ({
  presentation,
  region,
}: {
  presentation?: 'compact' | 'expanded' | 'tooltip';
  region: 'bottom' | 'center' | 'dialog' | 'floating' | 'left' | 'popover' | 'right';
}) => {
  const { t } = useTranslation();
  const counts = useQueueCounts();
  const activity = getPersonalQueueActivity(counts);

  if (region === 'bottom' && presentation !== 'expanded') {
    const isGenerating = activity.inProgress > 0;

    return (
      <StatusWidgetChip icon={ListOrderedIcon} tone={isGenerating ? 'accent' : undefined}>
        {isGenerating
          ? t('widgets.queue.generating', { count: activity.inProgress })
          : t('widgets.queue.queued', { count: activity.pending })}
      </StatusWidgetChip>
    );
  }

  return <QueueContent />;
};

const QueueContent = () => {
  const { t } = useTranslation();
  const [filter, setFilter] = useState<QueueFilterId>('all');
  const revealRequest = usePendingQueueItemReveal();
  const [handledRevealRequestId, setHandledRevealRequestId] = useState<number | null>(null);

  // Adjust the active tab during render so it cannot hide a requested item.
  if (revealRequest !== null && revealRequest.requestId !== handledRevealRequestId) {
    setHandledRevealRequestId(revealRequest.requestId);
    setFilter('all');
  }

  // The stats and filter stay put while the item lists scroll on their own.
  return (
    <Stack gap="3" h="full" minH="0" pt="3">
      <Stack flexShrink={0} gap="3" px="3">
        <QueueStats />
        <QueueFilterTabs value={filter} onChange={setFilter} />
      </Stack>
      <Scrollable flex="1" label={t('widgets.queue.items')} minH="0" overflowX="hidden">
        <Stack gap="3" pb="3" px="3">
          <CurrentBatchSection revealRequest={revealRequest} />
          <RecentSection filter={filter} revealRequest={revealRequest} />
        </Stack>
      </Scrollable>
    </Stack>
  );
};
