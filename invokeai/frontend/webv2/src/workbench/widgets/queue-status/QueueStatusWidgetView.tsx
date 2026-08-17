import type { WidgetViewProps } from '@workbench/widgetContracts';

import { Stack, Text } from '@chakra-ui/react';
import { getDeterminateProgressPercent, getQueueSummary } from '@features/queue/contracts';
import { useIsProcessorPaused } from '@features/queue/react';
import { useActiveQueueProgress } from '@workbench/queue-integration/useActiveQueueProgress';
import { StatusWidgetChip } from '@workbench/widget-frame';
import { ListOrderedIcon, PauseIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

import { getQueueStatusChip, getQueueStatusProgress } from './queueStatusModel';

export const QueueStatusWidgetView = ({ presentation }: WidgetViewProps) => {
  const { t } = useTranslation();
  const { progress, queueItems } = useActiveQueueProgress();
  const isPaused = useIsProcessorPaused();
  // The chip's remaining/total counts must hold steady as progress advances
  // through a running batch's sub-images, so they come from a summary built
  // without progress; only the percent label tracks live progress.
  const summary = getQueueSummary(queueItems);
  const chip = getQueueStatusChip(summary, isPaused);
  const percent = getDeterminateProgressPercent(progress?.percentage);
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
