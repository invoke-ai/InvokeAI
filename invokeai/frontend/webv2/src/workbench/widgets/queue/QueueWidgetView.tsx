import type { WidgetViewProps } from '@workbench/types';

import { Stack } from '@chakra-ui/react';
import { StatusWidgetChip } from '@workbench/widget-frame';
import { ListOrderedIcon } from 'lucide-react';
import { useState } from 'react';

import type { QueueFilterId } from './queueFilters';

import { CurrentBatchSection } from './NowNextSection';
import { QueueFilterTabs } from './QueueFilterTabs';
import { useScopedQueueCounts } from './queueScope';
import { QueueStats } from './QueueStats';
import { RecentSection } from './RecentSection';

/**
 * Queue console: server-wide stats, status filters, a live NOW & NEXT card, and
 * the RECENT history. Header (title/Pause/menu) and the MODEL CACHE footer are
 * provided through the manifest's chrome slots; this view is just the scrolling
 * body. In a collapsed bottom dock it degrades to a single status chip.
 */
export const QueueWidgetView = ({ presentation, region }: WidgetViewProps) => {
  const counts = useScopedQueueCounts();

  if (region === 'bottom' && presentation !== 'expanded') {
    const isGenerating = counts.in_progress > 0;

    return (
      <StatusWidgetChip icon={ListOrderedIcon} tone={isGenerating ? 'accent' : undefined}>
        {isGenerating ? `${counts.in_progress} generating` : `${counts.pending} queued`}
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
