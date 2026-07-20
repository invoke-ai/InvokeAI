import { Stack } from '@chakra-ui/react';
import { getPersonalQueueActivity } from '@features/queue/core/types';
import { StatusWidgetChip } from '@platform/ui/StatusWidgetChip';
import { ListOrderedIcon } from 'lucide-react';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';

import type { QueueFilterId } from './queueFilters';

import { CurrentBatchSection } from './NowNextSection';
import { useQueueCounts } from './queueDataStore';
import { QueueFilterTabs } from './QueueFilterTabs';
import { QueueStats } from './QueueStats';
import { RecentSection } from './RecentSection';

/**
 * Queue console: server-wide stats, status filters, a live NOW & NEXT card, and
 * the RECENT history. Header (title/Pause/menu) and the MODEL CACHE footer are
 * provided through the manifest's chrome slots; this view is just the scrolling
 * body. In a collapsed bottom dock it degrades to a single status chip.
 */
export const QueueWidgetView = ({
  presentation,
  region,
}: {
  presentation?: 'compact' | 'expanded' | 'tooltip';
  region: 'bottom' | 'center' | 'dialog' | 'left' | 'popover' | 'right';
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
  const [filter, setFilter] = useState<QueueFilterId>('all');

  return (
    <Stack gap="3" p="3">
      <QueueStats />
      <QueueFilterTabs value={filter} onChange={setFilter} />
      <CurrentBatchSection />
      <RecentSection filter={filter} />
    </Stack>
  );
};
