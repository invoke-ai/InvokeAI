import type { ModelLoadInfo } from '@features/models';
import type { QueueActivity, QueueItem } from '@features/queue/contracts';
import type { QueueItemProgress } from '@features/queue/react';

import { Badge, chakra, HStack, Icon, Menu, Portal, Progress, Stack, Text } from '@chakra-ui/react';
import { useModelLoads } from '@features/models';
import {
  getDeterminateProgressFraction,
  getQueueActivity,
  getQueueItemExpectedImageCount,
  getQueueItemSnapshotPositivePrompt,
} from '@features/queue/contracts';
import { getQueueAction, QueueMenuItems, useQueueActions } from '@features/queue/menu';
import { useIsProcessorPaused } from '@features/queue/react';
import { useMountEffect } from '@platform/react/useMountEffect';
import { Button, IconButton } from '@platform/ui/Button';
import { Group } from '@platform/ui/Group';
import { MenuContent } from '@platform/ui/Menu';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { Tooltip } from '@platform/ui/Tooltip';
import { getDestinationLabel, getSourceLabel } from '@workbench/invocation';
import { useActiveQueueProgress } from '@workbench/queue-integration/useActiveQueueProgress';
import { useOpenWorkbenchWidget } from '@workbench/useOpenWorkbenchWidget';
import { useWorkbenchSelector } from '@workbench/WorkbenchContext';
import { CloudOffIcon, ListOrderedIcon, ListXIcon, PauseIcon, XIcon } from 'lucide-react';
import { useCallback, useMemo, useState, type MouseEvent, type PointerEvent, type ReactNode } from 'react';
import { useTranslation } from 'react-i18next';

import { TopbarMenuChevron } from './TopbarMenuChevron';

const TOOLTIP_CONTENT_PROPS = { maxW: '22rem', p: '0' };

const activityPalettes: Record<QueueActivity, string | undefined> = {
  idle: undefined,
  'loading-model': 'accent',
  offline: 'orange',
  paused: 'orange',
  running: 'accent',
};

const activityIcons = {
  idle: ListOrderedIcon,
  'loading-model': ListOrderedIcon,
  offline: CloudOffIcon,
  paused: PauseIcon,
  running: ListOrderedIcon,
} satisfies Record<QueueActivity, typeof ListOrderedIcon>;

/** Local queue item statuses, worded as the queue widget words backend ones. */
const itemStatusLabelKeys: Record<QueueItem['status'], string> = {
  cancelled: 'common.status.canceled',
  completed: 'common.status.completed',
  failed: 'common.status.failed',
  pending: 'common.status.pending',
  running: 'common.generating',
};

export const QueueCluster = () => {
  const { t } = useTranslation();
  const backendConnectionStatus = useWorkbenchSelector((snapshot) => snapshot.backendConnection.status);
  const modelLoads = useModelLoads();
  const openWorkbenchWidget = useOpenWorkbenchWidget();
  const actions = useQueueActions({ includeOpenQueue: true });
  const cancelCurrent = getQueueAction(actions, 'cancel-current');
  const cancelAll = getQueueAction(actions, 'cancel-all');
  const [isCancelHovered, setIsCancelHovered] = useState(false);
  const [isShiftHeld, setIsShiftHeld] = useState(false);

  const { progress: runningProgress, queueItems, summary } = useActiveQueueProgress();
  const { current, remaining, total } = summary;
  const runningItem = summary.runningQueueItemId
    ? queueItems.find((item) => item.id === summary.runningQueueItemId)
    : undefined;

  const isConnected = backendConnectionStatus === 'connected';
  const isPaused = useIsProcessorPaused();
  const activity = getQueueActivity({
    hasOpenWork: total > 0,
    isConnected,
    isLoadingModels: modelLoads.length > 0,
    isPaused,
  });
  const progressValue = getDeterminateProgressFraction(runningProgress?.percentage);
  const isCancellable = !cancelCurrent?.disabled;
  const [lastCancellable, setLastCancellable] = useState(isCancellable);

  // The "x" unmounts while hovered when the running job finishes, so its
  // pointer-leave never fires; reset the gesture state with it.
  if (isCancellable !== lastCancellable) {
    setLastCancellable(isCancellable);

    if (!isCancellable) {
      setIsCancelHovered(false);
      setIsShiftHeld(false);
    }
  }

  const isCancelAllArmed = isCancelHovered && isShiftHeld && cancelAll !== undefined && !cancelAll.disabled;

  const handleOpenQueue = useCallback(() => openWorkbenchWidget('queue'), [openWorkbenchWidget]);
  const handleCancelPointerEnter = useCallback((event: PointerEvent<HTMLButtonElement>) => {
    setIsCancelHovered(true);
    setIsShiftHeld(event.shiftKey);
  }, []);
  const handleCancelPointerLeave = useCallback(() => {
    setIsCancelHovered(false);
    setIsShiftHeld(false);
  }, []);
  // Shift-click reaches every item, so it confirms first; a plain click stops only the running one and does not.
  const handleCancelClick = useCallback(
    (event: MouseEvent<HTMLButtonElement>) => {
      if (event.shiftKey && cancelAll && !cancelAll.disabled) {
        void cancelAll.trigger();
        return;
      }

      void cancelCurrent?.trigger();
    },
    [cancelAll, cancelCurrent]
  );

  return (
    <Menu.Root>
      <Group attached colorPalette={activityPalettes[activity]} flexShrink={0}>
        <QueueButtonTooltip
          activity={activity}
          current={current}
          item={runningItem}
          modelLoads={modelLoads}
          progress={runningProgress}
          remaining={remaining}
          total={total}
        >
          <Button
            aria-label={t(activity === 'offline' ? 'topbar.queue.remainingOffline' : 'topbar.queue.remaining', {
              count: remaining,
            })}
            fontVariantNumeric="tabular-nums"
            overflow="hidden"
            position="relative"
            size="lg"
            px={2}
            variant={activity === 'idle' ? 'outline' : 'subtle'}
            onClick={handleOpenQueue}
          >
            <Icon as={activityIcons[activity]} position="relative" zIndex="1" />
            <chakra.span fontVariantNumeric="tabular-nums" position="relative" textAlign="center" zIndex="1">
              {remaining}
            </chakra.span>
          </Button>
        </QueueButtonTooltip>

        {isCancellable && cancelCurrent ? (
          <Tooltip
            content={
              isCancelAllArmed && cancelAll
                ? cancelAll.label
                : `${cancelCurrent.label} · ${t('topbar.queue.shiftCancelsAll')}`
            }
            showArrow
          >
            <IconButton
              aria-label={isCancelAllArmed && cancelAll ? cancelAll.label : cancelCurrent.label}
              color="fg.error"
              size="lg"
              variant="outline"
              onClick={handleCancelClick}
              onPointerEnter={handleCancelPointerEnter}
              onPointerLeave={handleCancelPointerLeave}
            >
              <Icon as={isCancelAllArmed ? ListXIcon : XIcon} />
            </IconButton>
          </Tooltip>
        ) : null}
        {/* Shift over the hovered button swaps it to cancel-all live; pointer
            events alone cannot see a bare shift press, so watch the keyboard
            only while hovered. */}
        {isCancelHovered ? <ShiftKeyWatcher onChange={setIsShiftHeld} /> : null}

        <Menu.Trigger asChild>
          <IconButton
            aria-label={t('topbar.queue.actions')}
            className="group"
            size="lg"
            minW="0"
            w="6"
            variant="outline"
          >
            <TopbarMenuChevron fillTrigger />
          </IconButton>
        </Menu.Trigger>
      </Group>

      <QueueProgressAnnouncement
        activity={activity}
        current={current}
        progress={progressValue}
        remaining={remaining}
        total={total}
      />

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

const ShiftKeyWatcher = ({ onChange }: { onChange: (isShiftHeld: boolean) => void }) => {
  useMountEffect(() => {
    const readShift = (event: KeyboardEvent) => onChange(event.shiftKey);
    const clearShift = () => onChange(false);

    window.addEventListener('keydown', readShift);
    window.addEventListener('keyup', readShift);
    window.addEventListener('blur', clearShift);
    return () => {
      window.removeEventListener('keydown', readShift);
      window.removeEventListener('keyup', readShift);
      window.removeEventListener('blur', clearShift);
    };
  });

  return null;
};

const QueueButtonTooltip = ({
  activity,
  children,
  current,
  item,
  modelLoads,
  progress,
  remaining,
  total,
}: {
  activity: QueueActivity;
  children: ReactNode;
  current: number;
  item?: QueueItem;
  modelLoads: ModelLoadInfo[];
  progress: QueueItemProgress | null;
  remaining: number;
  total: number;
}) => {
  const content = useMemo(
    () => (
      <QueueTooltip
        activity={activity}
        current={current}
        item={item}
        modelLoads={modelLoads}
        progress={progress}
        remaining={remaining}
        total={total}
      />
    ),
    [activity, current, item, modelLoads, progress, remaining, total]
  );

  return (
    <Tooltip content={content} contentProps={TOOLTIP_CONTENT_PROPS} openDelay={200} showArrow>
      {children}
    </Tooltip>
  );
};

// Quantize screen-reader announcements so progress events do not flood the live region.
const QueueProgressAnnouncement = ({
  activity,
  current,
  progress,
  remaining,
  total,
}: {
  activity: QueueActivity;
  current: number;
  progress: number | null;
  remaining: number;
  total: number;
}) => {
  const { t } = useTranslation();
  const decile = progress === null ? null : Math.round(progress * 10) * 10;
  const message =
    activity === 'idle'
      ? t('topbar.queue.idle')
      : activity === 'offline'
        ? t('topbar.queue.offlineWaiting', { count: remaining })
        : activity === 'paused'
          ? t('topbar.queue.pausedWaiting', { count: remaining })
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
  activity,
  current,
  item,
  modelLoads,
  progress,
  remaining,
  total,
}: {
  activity: QueueActivity;
  current: number;
  item?: QueueItem;
  modelLoads: ModelLoadInfo[];
  progress: QueueItemProgress | null;
  remaining: number;
  total: number;
}) => {
  const { t } = useTranslation();
  if (modelLoads.length > 0 && total === 0) {
    return <ModelLoadsTooltipContent modelLoads={modelLoads} />;
  }

  if (activity === 'offline' || activity === 'paused') {
    return (
      <Stack gap="1" p="3">
        <Text fontSize="md" fontWeight="800">
          {activity === 'offline' ? t('topbar.queue.reconnecting') : t('widgets.queue.processorPaused')}
        </Text>
        <Text color="fg.subtle" fontSize="xs">
          {activity === 'offline'
            ? t('topbar.queue.offlineDescription', { count: remaining })
            : t('topbar.queue.pausedDescription', { count: remaining })}
        </Text>
      </Stack>
    );
  }

  if (total === 0) {
    return (
      <Stack gap="1" p="3">
        <Text fontSize="md" fontWeight="800" fontVariantNumeric="tabular-nums">
          {t('topbar.queue.idle')}
        </Text>
        <Text color="fg.subtle" fontSize="xs">
          {t('topbar.queue.noActiveBatches')}
        </Text>
      </Stack>
    );
  }

  if (!item) {
    return (
      <Stack gap="2" p="3">
        <Text fontSize="md" fontWeight="800" fontVariantNumeric="tabular-nums">
          {modelLoads[0]
            ? t('topbar.queue.loadingModel', { model: modelLoads[0].label })
            : t('topbar.queue.waitingToStart')}
        </Text>
        <Text color="fg.subtle" fontSize="xs">
          {t('topbar.queue.queuedImages', { count: total })}
        </Text>
        {modelLoads.length > 1 ? <ModelLoadList modelLoads={modelLoads} /> : null}
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
  const progressLabel = modelLoads[0]
    ? t('topbar.queue.loadingModel', { model: modelLoads[0].label })
    : progress?.message?.trim() || t('topbar.queue.waitingForProgress');

  return (
    <Stack gap="2" p="3" minW="18rem">
      <HStack justify="space-between" gap="3">
        <Text fontSize="md" fontWeight="800">
          {t('topbar.queue.generatingImage', { current, total })}
        </Text>
        <Badge colorPalette="blue" fontSize="xs">
          {t(itemStatusLabelKeys[item.status])}
        </Badge>
      </HStack>
      <Stack gap="1">
        <Text color="fg.subtle" fontSize="xs">
          {progressLabel}
        </Text>
        {modelLoads.length > 1 ? <ModelLoadList modelLoads={modelLoads} /> : null}
        <Progress.Root
          aria-label={t('topbar.queue.currentImageProgress')}
          max={1}
          value={getDeterminateProgressFraction(progress?.percentage)}
        >
          <Progress.Track>
            <Progress.Range />
          </Progress.Track>
        </Progress.Root>
      </Stack>
      <Stack gap="1" color="fg.subtle" fontSize="xs">
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
        <Text color="fg.muted" fontSize="xs" lineClamp={2}>
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
      <Text fontSize="md" fontWeight="800">
        {t('topbar.queue.loadingModelsCount', { count: modelLoads.length })}
      </Text>
      <ModelLoadList modelLoads={modelLoads} />
    </Stack>
  );
};

const ModelLoadList = ({ modelLoads }: { modelLoads: ModelLoadInfo[] }) => {
  const { t } = useTranslation();

  return (
    <Stack gap="0.5" color="fg.muted" fontSize="xs">
      {modelLoads.slice(0, 3).map((modelLoad, index) => (
        <MiddleTruncate key={`${modelLoad.label}:${index}`} text={modelLoad.label} />
      ))}
      {modelLoads.length > 3 ? <Text>{t('topbar.queue.more', { count: modelLoads.length - 3 })}</Text> : null}
    </Stack>
  );
};
