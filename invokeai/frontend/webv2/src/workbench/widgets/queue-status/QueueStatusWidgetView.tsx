import type { WidgetViewProps } from '@workbench/widgetContracts';

import { Stack, Text } from '@chakra-ui/react';
import { getDeterminateProgressPercent } from '@features/queue/contracts';
import { useQueueChromeActivity } from '@workbench/queue-integration/useActiveQueueProgress';
import { StatusWidgetChip } from '@workbench/widget-frame';
import { CloudOffIcon, ListOrderedIcon, PauseIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

import { getQueueStatusChip, getQueueStatusProgress } from './queueStatusModel';

export const QueueStatusWidgetView = ({ presentation }: WidgetViewProps) => {
  const { t } = useTranslation();
  const { activity, elsewhere, modelLoads, progress, remaining, summary } = useQueueChromeActivity();
  const chip = getQueueStatusChip({ remaining }, activity);
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
        {elsewhere > 0 ? (
          <Text color="fg.subtle" fontSize="xs">
            {t(summary.total === 0 ? 'topbar.queue.queuedElsewhere' : 'topbar.queue.alsoQueuedElsewhere', {
              count: elsewhere,
            })}
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
