/* eslint-disable react/react-compiler */
import type { QueueItem } from '@workbench/types';

import { Badge, HStack, Icon, Progress, Stack, Text } from '@chakra-ui/react';
import { useModelLoads, type ModelLoadInfo } from '@workbench/backend/modelLoadStore';
import { useQueueItemProgress, type QueueItemProgress } from '@workbench/backend/progressStore';
import { Button, Tooltip } from '@workbench/components/ui';
import { getDestinationLabel, getSourceLabel } from '@workbench/invocation';
import { getQueueItemSourceWidgetValues } from '@workbench/queueSnapshot';
import { getQueueItemExpectedImageCount, getQueueProgressBarState, getQueueSummary } from '@workbench/queueSummary';
import { useOpenWorkbenchWidget } from '@workbench/useOpenWorkbenchWidget';
import { useActiveProjectSelector, useWorkbenchSelector } from '@workbench/WorkbenchContext';
import { ListOrderedIcon } from 'lucide-react';
import { useCallback, useMemo } from 'react';

const TOOLTIP_CONTENT_PROPS = { maxW: '22rem', p: '0' };

export const QueueInfo = () => {
  const queueItems = useActiveProjectSelector((project) => project.queue.items);
  const backendConnectionStatus = useWorkbenchSelector((snapshot) => snapshot.state.backendConnection.status);
  const modelLoads = useModelLoads();
  const openWorkbenchWidget = useOpenWorkbenchWidget();
  const baseSummary = getQueueSummary(queueItems);
  const runningProgress = useQueueItemProgress(baseSummary.runningQueueItemId ?? '');
  const summary = getQueueSummary(queueItems, runningProgress);
  const runningItem = summary.runningQueueItemId
    ? queueItems.find((item) => item.id === summary.runningQueueItemId)
    : undefined;

  const progressState = getQueueProgressBarState({
    isConnected: backendConnectionStatus === 'connected',
    isRunning: Boolean(runningItem),
    loadingModelsCount: modelLoads.length,
    progress: runningProgress,
  });

  const tooltipContent = useMemo(
    () => (
      <QueueInfoTooltip
        item={runningItem}
        modelLoads={modelLoads}
        progress={runningProgress}
        total={summary.total}
        current={summary.current}
      />
    ),
    [modelLoads, runningItem, runningProgress, summary]
  );

  const handleOpenQueue = useCallback(() => openWorkbenchWidget('queue'), [openWorkbenchWidget]);

  return (
    <Tooltip content={tooltipContent} contentProps={TOOLTIP_CONTENT_PROPS} openDelay={200} showArrow>
      <Button
        aria-label={`Queue progress ${summary.current} of ${summary.total}`}
        variant="outline"
        h="9"
        borderColor="border.subtle"
        fontSize="xs"
        fontWeight="700"
        gap="1"
        overflow="hidden"
        position="relative"
        px="2"
        onClick={handleOpenQueue}
      >
        <Icon as={ListOrderedIcon} boxSize="4" flexShrink="0" />
        {summary.current}/{summary.total}
        <QueueBadgeProgressBar state={progressState} />
      </Button>
    </Tooltip>
  );
};

const QueueBadgeProgressBar = ({ state }: { state: ReturnType<typeof getQueueProgressBarState> }) => {
  const isIdle = state.kind === 'idle';

  return (
    <Progress.Root
      aria-label={isIdle ? 'Invoke progress idle' : 'Invoke progress'}
      bottom="0"
      colorPalette="accent"
      h={1}
      insetInline={0}
      max={1}
      pointerEvents="none"
      position="absolute"
      value={state.value}
      visibility={state.kind !== 'idle' ? 'visible' : 'hidden'}
    >
      <Progress.Track bg={isIdle ? 'border.subtle' : 'bg.emphasized'} h="full">
        <Progress.Range bg="accent.solid" />
      </Progress.Track>
    </Progress.Root>
  );
};

const QueueInfoTooltip = ({
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
  if (modelLoads.length > 0 && total === 0) {
    return <ModelLoadsTooltipContent modelLoads={modelLoads} />;
  }

  if (total === 0) {
    return (
      <Stack gap="1" p="3">
        <Text fontSize="xs" fontWeight="800">
          Queue idle
        </Text>
        <Text color="fg.subtle" fontSize="2xs">
          No active or pending generation batches.
        </Text>
      </Stack>
    );
  }

  if (!item) {
    return (
      <Stack gap="2" p="3">
        <Text fontSize="xs" fontWeight="800">
          {modelLoads.length ? 'Loading models' : 'Waiting to start'}
        </Text>
        <Text color="fg.subtle" fontSize="2xs">
          {total} image{total === 1 ? '' : 's'} queued.
        </Text>
        {modelLoads.length ? <ModelLoadList modelLoads={modelLoads} /> : null}
      </Stack>
    );
  }

  const sourceValues = getQueueItemSourceWidgetValues(item);
  const prompt = typeof sourceValues.positivePrompt === 'string' ? sourceValues.positivePrompt.trim() : '';
  const expectedCount = getQueueItemExpectedImageCount(item);
  const activeItemIndex = Math.min(expectedCount, Math.max(1, progress?.activeItemIndex ?? 1));
  const activeBackendItemId = item.backendItemIds?.[activeItemIndex - 1];
  const progressLabel = modelLoads.length
    ? `Loading ${modelLoads.length} model${modelLoads.length === 1 ? '' : 's'}.`
    : progress?.message?.trim() || 'Backend accepted batch; waiting for progress event.';

  return (
    <Stack gap="2" p="3" minW="18rem">
      <HStack justify="space-between" gap="3">
        <Text fontSize="xs" fontWeight="800">
          Loading image {current}/{total}
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
        {progress?.percentage !== null && progress?.percentage !== undefined ? (
          <Progress.Root aria-label="Current image progress" max={1} size="xs" value={progress.percentage}>
            <Progress.Track>
              <Progress.Range />
            </Progress.Track>
          </Progress.Root>
        ) : null}
      </Stack>
      <Stack gap="1" color="fg.subtle" fontSize="2xs">
        <Text>
          Batch image {activeItemIndex}/{expectedCount}
          {activeBackendItemId !== undefined ? ` · backend item ${activeBackendItemId}` : ''}
        </Text>
        <Text fontFamily="mono" truncate>
          Queue item {item.id}
        </Text>
        {item.backendBatchId ? (
          <Text fontFamily="mono" truncate>
            Backend batch {item.backendBatchId}
          </Text>
        ) : null}
        <Text truncate>
          {getSourceLabel(item.snapshot.sourceId)} to {getDestinationLabel(item.snapshot.destination)} ·{' '}
          {item.snapshot.graph.label || item.snapshot.graph.id}
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

const ModelLoadsTooltipContent = ({ modelLoads }: { modelLoads: ModelLoadInfo[] }) => (
  <Stack gap="2" p="3" minW="18rem">
    <Text fontSize="xs" fontWeight="800">
      Loading {modelLoads.length} model{modelLoads.length === 1 ? '' : 's'}
    </Text>
    <ModelLoadList modelLoads={modelLoads} />
  </Stack>
);

const ModelLoadList = ({ modelLoads }: { modelLoads: ModelLoadInfo[] }) => (
  <Stack gap="0.5" color="fg.muted" fontSize="2xs">
    {modelLoads.slice(0, 3).map((modelLoad, index) => (
      <Text key={`${modelLoad.label}:${index}`} truncate>
        {modelLoad.label}
      </Text>
    ))}
    {modelLoads.length > 3 ? <Text>{modelLoads.length - 3} more...</Text> : null}
  </Stack>
);
