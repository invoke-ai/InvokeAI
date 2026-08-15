import type { WidgetViewProps } from '@workbench/widgetContracts';

import { Stack, Text } from '@chakra-ui/react';
import { getQueueSummary } from '@features/queue/contracts';
import { useIsProcessorPaused, useQueueItemProgress } from '@features/queue/react';
import { StatusWidgetChip } from '@workbench/widget-frame';
import { useActiveProjectSelector } from '@workbench/WorkbenchContext';
import { ListOrderedIcon, PauseIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

import { getQueueStatusChip, getQueueStatusProgress } from './queueStatusModel';

export const QueueStatusWidgetView = ({ presentation }: WidgetViewProps) => {
  const { t } = useTranslation();
  const queueItems = useActiveProjectSelector((project) => project.queue.items);
  const isPaused = useIsProcessorPaused();
  const summary = getQueueSummary(queueItems);
  const progress = useQueueItemProgress(summary.runningQueueItemId ?? '');
  const chip = getQueueStatusChip(summary, isPaused);
  const percent = typeof progress?.percentage === 'number' ? Math.round(progress.percentage * 100) : null;
  const baseLabel =
    chip.labelKey === 'idle'
      ? t('widgets.queueStatus.idle')
      : t(`widgets.queueStatus.${chip.labelKey}`, { count: chip.count });
  // The count answers "how much is left"; the percent answers "how far into the
  // current one", which is the half the chip was missing.
  const label = chip.tone === 'running' && percent !== null ? `${baseLabel} · ${percent}%` : baseLabel;

  if (presentation === 'tooltip') {
    return (
      <Stack gap="2">
        <Text fontSize="xs" fontWeight="700">
          {t('widgets.labels.queueStatus')}
        </Text>
        <Text color="fg.subtle" fontSize="2xs">
          {label}
        </Text>
      </Stack>
    );
  }

  return (
    <StatusWidgetChip
      icon={chip.tone === 'paused' ? PauseIcon : ListOrderedIcon}
      progress={getQueueStatusProgress(chip, progress?.percentage)}
    >
      {label}
    </StatusWidgetChip>
  );
};
