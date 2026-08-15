import type { ModelLoadInfo } from '@features/models';
import type { QueueItem } from '@features/queue/contracts';
import type { QueueItemProgress } from '@features/queue/react';
import type { ReactNode } from 'react';

import { Badge, chakra, HStack, Icon, Menu, Portal, Progress, Stack, Text } from '@chakra-ui/react';
import { useModelLoads } from '@features/models';
import {
  getDeterminateProgressFraction,
  getQueueItemExpectedImageCount,
  getQueueItemSnapshotPositivePrompt,
} from '@features/queue/contracts';
import { QueueMenuItems, useQueueMenuActions } from '@features/queue/menu';
import { useIsProcessorPaused } from '@features/queue/react';
import { Button, IconButton } from '@platform/ui/Button';
import { Group } from '@platform/ui/Group';
import { MenuContent } from '@platform/ui/Menu';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { Tooltip } from '@platform/ui/Tooltip';
import { getDestinationLabel, getSourceLabel } from '@workbench/invocation';
import { useActiveQueueProgress } from '@workbench/queue-integration/useActiveQueueProgress';
import { useOpenWorkbenchWidget } from '@workbench/useOpenWorkbenchWidget';
import { useWorkbenchSelector } from '@workbench/WorkbenchContext';
import { ChevronDownIcon, ListOrderedIcon, PauseIcon, XIcon } from 'lucide-react';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const TOOLTIP_CONTENT_PROPS = { maxW: '22rem', p: '0' };

type QueueTone = 'idle' | 'running' | 'paused';

const tonePalettes: Record<QueueTone, string | undefined> = {
  idle: undefined,
  paused: 'orange',
  running: 'accent',
};

export const QueueCluster = () => {
  const { t } = useTranslation();
  const backendConnectionStatus = useWorkbenchSelector((snapshot) => snapshot.backendConnection.status);
  const modelLoads = useModelLoads();
  const openWorkbenchWidget = useOpenWorkbenchWidget();
  const actions = useQueueMenuActions({ includeOpenQueue: true });
  const cancelCurrent = actions[0];

  const { progress: runningProgress, queueItems, summary } = useActiveQueueProgress();
  const { current, remaining, total } = summary;
  const runningItem = summary.runningQueueItemId
    ? queueItems.find((item) => item.id === summary.runningQueueItemId)
    : undefined;

  const isConnected = backendConnectionStatus === 'connected';
  const isPaused = useIsProcessorPaused();
  const hasOpenWork = total > 0;
  const tone: QueueTone = isPaused && hasOpenWork ? 'paused' : hasOpenWork && isConnected ? 'running' : 'idle';
  const progressValue = getDeterminateProgressFraction(runningProgress?.percentage);
  const isCancellable = !cancelCurrent?.disabled;

  const handleOpenQueue = useCallback(() => openWorkbenchWidget('queue'), [openWorkbenchWidget]);

  return (
    <Menu.Root>
      <Group attached colorPalette={tonePalettes[tone]} flexShrink={0}>
        <QueueButtonTooltip
          current={current}
          item={runningItem}
          modelLoads={modelLoads}
          progress={runningProgress}
          total={total}
        >
          <Button
            aria-label={t('topbar.queue.remaining', { count: remaining })}
            fontVariantNumeric="tabular-nums"
            overflow="hidden"
            position="relative"
            size="xs"
            px={2}
            variant={tone === 'idle' ? 'outline' : 'subtle'}
            onClick={handleOpenQueue}
          >
            {/* No background fill here any more: the top bar's progress rail is
                twenty pixels below this button and says the same thing at
                viewport width, where a 35%-opacity wash behind a number never
                really did. The count and the tooltip are this control's job. */}
            <Icon as={isPaused ? PauseIcon : ListOrderedIcon} position="relative" zIndex="1" />
            <chakra.span fontVariantNumeric="tabular-nums" position="relative" textAlign="center" zIndex="1">
              {remaining}
            </chakra.span>
          </Button>
        </QueueButtonTooltip>

        {isCancellable && cancelCurrent ? (
          <Tooltip content={cancelCurrent.label} showArrow>
            <IconButton
              aria-label={cancelCurrent.label}
              color="fg.error"
              size="xs"
              variant="outline"
              onClick={cancelCurrent.onClick}
            >
              <Icon as={XIcon} />
            </IconButton>
          </Tooltip>
        ) : null}

        <Menu.Trigger asChild>
          <IconButton
            aria-label={t('topbar.queue.actions')}
            color="fg.subtle"
            size="xs"
            minW="0"
            w="6"
            variant="outline"
          >
            <Icon as={ChevronDownIcon} />
          </IconButton>
        </Menu.Trigger>
      </Group>

      <QueueProgressAnnouncement current={current} progress={progressValue} total={total} />

      <Portal>
        <Menu.Positioner>
          <MenuContent>
            <QueueMenuItems actions={actions} />
          </MenuContent>
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );
};

const QueueButtonTooltip = ({
  children,
  current,
  item,
  modelLoads,
  progress,
  total,
}: {
  children: ReactNode;
  current: number;
  item?: QueueItem;
  modelLoads: ModelLoadInfo[];
  progress: QueueItemProgress | null;
  total: number;
}) => {
  const content = useMemo(
    () => <QueueTooltip current={current} item={item} modelLoads={modelLoads} progress={progress} total={total} />,
    [current, item, modelLoads, progress, total]
  );

  return (
    <Tooltip content={content} contentProps={TOOLTIP_CONTENT_PROPS} openDelay={200} showArrow>
      {children}
    </Tooltip>
  );
};

// Quantize screen-reader announcements so progress events do not flood the live region.
const QueueProgressAnnouncement = ({
  current,
  progress,
  total,
}: {
  current: number;
  progress: number | null;
  total: number;
}) => {
  const { t } = useTranslation();
  const decile = progress === null ? null : Math.round(progress * 10) * 10;
  const message =
    total === 0
      ? t('topbar.queue.idle')
      : decile === null
        ? t('topbar.queue.imageProgress', { current, total })
        : t('topbar.queue.imageProgressPercent', { current, percent: decile, total });

  return (
    <chakra.span aria-live="polite" srOnly>
      {message}
    </chakra.span>
  );
};

const QueueTooltip = ({
  current,
  item,
  modelLoads,
  progress,
  total,
}: {
  current: number;
  item?: QueueItem;
  modelLoads: ModelLoadInfo[];
  progress: QueueItemProgress | null;
  total: number;
}) => {
  const { t } = useTranslation();
  if (modelLoads.length > 0 && total === 0) {
    return <ModelLoadsTooltipContent modelLoads={modelLoads} />;
  }

  if (total === 0) {
    return (
      <Stack gap="1" p="3">
        <Text fontSize="xs" fontWeight="800" fontVariantNumeric="tabular-nums">
          {t('topbar.queue.idle')}
        </Text>
        <Text color="fg.subtle" fontSize="2xs">
          {t('topbar.queue.noActiveBatches')}
        </Text>
      </Stack>
    );
  }

  if (!item) {
    return (
      <Stack gap="2" p="3">
        <Text fontSize="xs" fontWeight="800" fontVariantNumeric="tabular-nums">
          {modelLoads.length ? t('topbar.queue.loadingModels') : t('topbar.queue.waitingToStart')}
        </Text>
        <Text color="fg.subtle" fontSize="2xs">
          {t('topbar.queue.queuedImages', { count: total })}
        </Text>
        {modelLoads.length ? <ModelLoadList modelLoads={modelLoads} /> : null}
      </Stack>
    );
  }

  const prompt = getQueueItemSnapshotPositivePrompt(item).trim();
  const expectedCount = getQueueItemExpectedImageCount(item);
  const activeItemIndex = Math.min(
    expectedCount,
    Math.max(1, progress?.activeItemIndex ?? (progress?.completedItemCount ?? 0) + 1)
  );
  const activeBackendItemId = item.backendItemIds?.[activeItemIndex - 1];
  const progressLabel = modelLoads.length
    ? t('topbar.queue.loadingModelsCount', { count: modelLoads.length })
    : progress?.message?.trim() || t('topbar.queue.waitingForProgress');

  return (
    <Stack gap="2" p="3" minW="18rem">
      <HStack justify="space-between" gap="3">
        <Text fontSize="xs" fontWeight="800">
          {t('topbar.queue.loadingImage', { current, total })}
        </Text>
        <Badge colorPalette="blue" fontSize="2xs">
          {item.status}
        </Badge>
      </HStack>
      <Stack gap="1">
        <Text color="fg.subtle" fontSize="2xs">
          {progressLabel}
        </Text>
        {modelLoads.length ? <ModelLoadList modelLoads={modelLoads} /> : null}
        <Progress.Root
          aria-label={t('topbar.queue.currentImageProgress')}
          max={1}
          size="xs"
          value={getDeterminateProgressFraction(progress?.percentage)}
        >
          <Progress.Track>
            <Progress.Range />
          </Progress.Track>
        </Progress.Root>
      </Stack>
      <Stack gap="1" color="fg.subtle" fontSize="2xs">
        <Text fontVariantNumeric="tabular-nums">
          {t('topbar.queue.batchImage', { current: activeItemIndex, total: expectedCount })}
          {activeBackendItemId !== undefined ? ` · ${t('topbar.queue.backendItem', { id: activeBackendItemId })}` : ''}
        </Text>
        <MiddleTruncate fontFamily="mono" text={t('topbar.queue.queueItem', { id: item.id })} />
        {item.backendBatchId ? (
          <MiddleTruncate fontFamily="mono" text={t('topbar.queue.backendBatch', { id: item.backendBatchId })} />
        ) : null}
        <Text truncate>
          {t('topbar.queue.route', {
            destination: getDestinationLabel(item.snapshot.destination),
            graph: item.snapshot.graph.label || item.snapshot.graph.id,
            source: getSourceLabel(item.snapshot.sourceId),
          })}
        </Text>
      </Stack>
      {prompt ? (
        <Text color="fg.muted" fontSize="2xs" lineClamp={2}>
          {prompt}
        </Text>
      ) : null}
    </Stack>
  );
};

const ModelLoadsTooltipContent = ({ modelLoads }: { modelLoads: ModelLoadInfo[] }) => {
  const { t } = useTranslation();

  return (
    <Stack gap="2" p="3" minW="18rem">
      <Text fontSize="xs" fontWeight="800">
        {t('topbar.queue.loadingModelsCount', { count: modelLoads.length })}
      </Text>
      <ModelLoadList modelLoads={modelLoads} />
    </Stack>
  );
};

const ModelLoadList = ({ modelLoads }: { modelLoads: ModelLoadInfo[] }) => {
  const { t } = useTranslation();

  return (
    <Stack gap="0.5" color="fg.muted" fontSize="2xs">
      {modelLoads.slice(0, 3).map((modelLoad, index) => (
        <MiddleTruncate key={`${modelLoad.label}:${index}`} text={modelLoad.label} />
      ))}
      {modelLoads.length > 3 ? <Text>{t('topbar.queue.more', { count: modelLoads.length - 3 })}</Text> : null}
    </Stack>
  );
};
