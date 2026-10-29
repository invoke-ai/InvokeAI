import type { QueueActivity } from '@features/queue/core/queueActivity';

import { Stack } from '@chakra-ui/react';
import { Scrollable } from '@platform/ui/Scrollable';
import { StatusWidgetChip } from '@platform/ui/StatusWidgetChip';
import { CloudOffIcon, ListOrderedIcon, PauseIcon, type LucideIcon } from 'lucide-react';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';

import type { QueueFilterId } from './queueFilters';

import { CurrentBatchSection } from './NowNextSection';
import { QueueFilterTabs } from './QueueFilterTabs';
import { usePendingQueueItemReveal } from './queueUiStore';
import { RecentSection } from './RecentSection';
import { useQueueWidgetActivity } from './useQueueWidgetActivity';

export const QueueWidgetView = ({
  presentation,
  region,
}: {
  presentation?: 'compact' | 'expanded' | 'tooltip';
  region: 'bottom' | 'center' | 'dialog' | 'floating' | 'left' | 'popover' | 'right';
}) => {
  if (region === 'bottom' && presentation !== 'expanded') {
    return <QueueStatusChip />;
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

  // The counted filters stay put while the item lists scroll on their own.
  return (
    <Stack gap="3" h="full" minH="0" pt="3">
      <Stack flexShrink={0} gap="3" px="3">
        <QueueFilterTabs value={filter} onChange={setFilter} />
      </Stack>
      <Scrollable flex="1" label={t('widgets.queue.items')} minH="0" overflowX="hidden">
        <Stack gap="3" pb="3" px="3">
          <CurrentBatchSection filter={filter} revealRequest={revealRequest} />
          <RecentSection filter={filter} revealRequest={revealRequest} />
        </Stack>
      </Scrollable>
    </Stack>
  );
};

const chipIcons = {
  idle: ListOrderedIcon,
  'loading-model': ListOrderedIcon,
  offline: CloudOffIcon,
  paused: PauseIcon,
  running: ListOrderedIcon,
} satisfies Record<QueueActivity, LucideIcon>;

const QueueStatusChip = () => {
  const { t } = useTranslation();
  const { activity, inProgress, loadingModel, pending } = useQueueWidgetActivity();
  const openCount = inProgress + pending;
  const label =
    activity === 'offline'
      ? t('widgets.queueStatus.offline', { count: openCount })
      : activity === 'paused'
        ? t('widgets.queueStatus.paused', { count: openCount })
        : activity === 'loading-model'
          ? t('widgets.queueStatus.loadingModel', { model: loadingModel })
          : inProgress > 0
            ? t('widgets.queue.generating', { count: inProgress })
            : t('widgets.queue.queued', { count: pending });

  return (
    <StatusWidgetChip
      icon={chipIcons[activity]}
      tone={
        activity === 'offline'
          ? 'warning'
          : activity === 'running' || activity === 'loading-model'
            ? 'accent'
            : undefined
      }
    >
      {label}
    </StatusWidgetChip>
  );
};
