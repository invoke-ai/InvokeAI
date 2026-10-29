import type { WidgetViewProps } from '@workbench/widgetContracts';

import { Stack, Text } from '@chakra-ui/react';
import { useModelLoads } from '@features/models';
import { getDeterminateProgressPercent, getQueueActivity, getQueueSummary } from '@features/queue/contracts';
import { useIsProcessorPaused } from '@features/queue/react';
import { useActiveQueueProgress } from '@workbench/queue-integration/useActiveQueueProgress';
import { StatusWidgetChip } from '@workbench/widget-frame';
import { useWorkbenchSelector } from '@workbench/WorkbenchContext';
import { CloudOffIcon, ListOrderedIcon, PauseIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

import { getQueueStatusChip, getQueueStatusProgress } from './queueStatusModel';

export const QueueStatusWidgetView = ({ presentation }: WidgetViewProps) => {
  const { t } = useTranslation();
  const { progress, queueItems } = useActiveQueueProgress();
  const isPaused = useIsProcessorPaused();
  const isConnected = useWorkbenchSelector((snapshot) => snapshot.backendConnection.status === 'connected');
  const modelLoads = useModelLoads();
  // Derive counts without progress so batch sub-image updates change percent only.
  const summary = getQueueSummary(queueItems);
  const chip = getQueueStatusChip(
    summary,
    getQueueActivity({ hasOpenWork: summary.total > 0, isConnected, isLoadingModels: modelLoads.length > 0, isPaused })
  );
  const percent = getDeterminateProgressPercent(progress?.percentage);
  const label =
    chip.labelKey === 'idle'
      ? t('widgets.queueStatus.idle')
      : t(`widgets.queueStatus.${chip.labelKey}`, { count: chip.count, model: modelLoads[0]?.label });
  const showPercent = chip.labelKey === 'queued' && percent !== null;
  const detail = chip.labelKey === 'queued' ? progress?.message.trim() : undefined;

  if (presentation === 'tooltip') {
    return (
      <Stack gap="2">
        <Text fontSize="md" fontWeight="700">
          {t('widgets.labels.queueStatus')}
        </Text>
        <Text color="fg.subtle" fontSize="xs">
          {showPercent ? `${label} · ${percent}%` : label}
        </Text>
        {detail ? (
          <Text color="fg.subtle" fontSize="xs">
            {detail}
          </Text>
        ) : null}
      </Stack>
    );
  }

  return (
    <StatusWidgetChip
      hiddenLabel={t('widgets.labels.queueStatus')}
      icon={chip.tone === 'paused' ? PauseIcon : chip.tone === 'offline' ? CloudOffIcon : ListOrderedIcon}
      progress={getQueueStatusProgress(chip, progress?.percentage)}
      tone={chip.tone === 'offline' ? 'warning' : undefined}
    >
      {label}
      {/* Reserved, tabular digits: the chip must not reflow the status bar as
          the percent ticks from 9 to 10 to 100. */}
      {showPercent ? (
        <Text as="span" display="inline-block" fontVariantNumeric="tabular-nums" minW="7ch" textAlign="end">
          {` · ${percent}%`}
        </Text>
      ) : null}
    </StatusWidgetChip>
  );
};
