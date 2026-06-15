import type { QueueItem } from '@workbench/types';

import {
  Badge,
  Flex,
  Group,
  HStack,
  Icon,
  Menu,
  NumberInput,
  Portal,
  Progress,
  Separator,
  Stack,
  Text,
} from '@chakra-ui/react';
import { Link } from '@tanstack/react-router';
import { DEFAULT_THEME_ID, THEMES_BY_ID } from '@theme/themes';
import { AccountMenu } from '@workbench/auth/components/AccountMenu';
import { useModelLoads, type ModelLoadInfo } from '@workbench/backend/modelLoadStore';
import { useQueueItemProgress, type QueueItemProgress } from '@workbench/backend/progressStore';
import { getDestinationLabel, getSourceLabel } from '@workbench/invocation';
import { getQueueItemExpectedImageCount, getQueueProgressBarState, getQueueSummary } from '@workbench/queueSummary';
import { useWorkbenchPreferences } from '@workbench/settings/store';
import { useOpenWorkbenchWidget } from '@workbench/useOpenWorkbenchWidget';
import { useActiveProjectSelector, useWorkbenchDispatch, useWorkbenchSelector } from '@workbench/WorkbenchContext';
import { ChevronDownIcon, ListOrderedIcon, PauseIcon, PlayIcon, XIcon } from 'lucide-react';

import { InvokeControl } from './InvokeControl';
import { LayoutPresetMenu } from './LayoutPresetMenu';
import { ProjectTabs } from './ProjectTabs';
import { Button, IconButton } from './ui/Button';
import { Tooltip } from './ui/Tooltip';

/** Workbench top bar: brand, global Invoke command cluster, project tabs, layout + account controls. */
export const TopBar = () => (
  <Flex
    align="center"
    as="header"
    bg="bg.subtle"
    borderBottomWidth="1px"
    borderColor="border.subtle"
    flexShrink={0}
    gap="2"
    h="12"
    pe="1.5"
    w="full"
  >
    <BrandMark />
    <InvokeControl />
    <BatchCountField />
    <QueueInfo />
    <QueueActions />
    <Separator orientation="vertical" h={5} />
    <ProjectTabs />
    <LayoutPresetMenu />
    <AccountMenu />
  </Flex>
);

/** Compact Invoke logo used as a link to home screen. */
const BrandMark = () => {
  const { themeId } = useWorkbenchPreferences();
  const theme = THEMES_BY_ID[themeId] ?? THEMES_BY_ID[DEFAULT_THEME_ID];

  return (
    <Link
      to="/"
      style={{
        flexShrink: 0,
        height: '100%',
        aspectRatio: '1/1',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}
    >
      <svg aria-hidden="true" fill="none" height="20" viewBox="0 0 44 44" width="20">
        <path
          d="M29.1951 10.6667H42V2H2V10.6667H14.8049L29.1951 33.3333H42V42H2V33.3333H14.8049"
          stroke={theme.colors.brand.solid}
          strokeWidth="2.8"
        />
      </svg>
    </Link>
  );
};

const getBatchCount = (values: Record<string, unknown>): number => {
  const batchCount = values.batchCount;

  return typeof batchCount === 'number' && Number.isFinite(batchCount) ? batchCount : 1;
};

const BatchCountField = () => {
  const generateValues = useActiveProjectSelector((project) => project.widgetStates.generate.values);
  const dispatch = useWorkbenchDispatch();
  const batchCount = getBatchCount(generateValues);

  return (
    <NumberInput.Root
      allowMouseWheel
      flexShrink={0}
      max={64}
      min={1}
      size="sm"
      value={String(batchCount)}
      w="14"
      onValueChange={({ valueAsNumber }) => {
        if (Number.isFinite(valueAsNumber)) {
          dispatch({ batchCount: valueAsNumber, type: 'setGenerateBatchCount' });
        }
      }}
    >
      <NumberInput.Control />
      <NumberInput.Input paddingStart="4" aria-label="Batch count" />
    </NumberInput.Root>
  );
};

const QueueInfo = () => {
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
  const onClick = () => {
    openWorkbenchWidget('queue');
  };

  return (
    <Tooltip
      content={
        <QueueInfoTooltip
          item={runningItem}
          modelLoads={modelLoads}
          progress={runningProgress}
          total={summary.total}
          current={summary.current}
        />
      }
      contentProps={{ maxW: '22rem', p: '0' }}
      openDelay={200}
      showArrow
    >
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
        onClick={onClick}
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

  const generateValues = item.snapshot.widgetStates.generate.values;
  const prompt = typeof generateValues.positivePrompt === 'string' ? generateValues.positivePrompt.trim() : '';
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

/**
 * Queue status + cancel cluster placeholder.
 *
 * Mirrors the spec's queue progress / cancel affordance. Real queue wiring,
 * snapshotting, and cancellation arrive with the Invocation Controller phases.
 */
const QueueActions = () => {
  const queueEndActions = [
    {
      label: 'Cancel Current Item',
      icon: XIcon,
      onClick: () => void 0,
    },
    {
      label: 'Cancel All Items',
      icon: XIcon,
      onClick: () => void 0,
    },
    {
      label: 'Cancel all except current item',
      icon: XIcon,
      onClick: () => void 0,
    },
  ];

  const queueProcessorActions = [
    {
      label: 'Resume Processor',
      icon: PlayIcon,
      onClick: () => void 0,
    },
    {
      label: 'Pause Processor',
      icon: PauseIcon,
      onClick: () => void 0,
    },
    {
      label: 'Open Queue',
      icon: ListOrderedIcon,
      onClick: () => void 0,
    },
  ];

  return (
    <Menu.Root>
      <Group attached>
        <Tooltip content="Cancel current item" showArrow>
          <IconButton variant="outline" size="sm" roundedEnd="none" borderColor="border.subtle">
            <XIcon />
          </IconButton>
        </Tooltip>
        <Menu.Trigger>
          <IconButton
            variant="outline"
            size="sm"
            roundedStart="none"
            borderStartWidth="0"
            aspectRatio="unset"
            minW="0"
            w="6"
            borderColor="border.subtle"
          >
            <ChevronDownIcon />
          </IconButton>
        </Menu.Trigger>
      </Group>
      <Portal>
        <Menu.Positioner>
          <Menu.Content>
            <Menu.ItemGroup>
              {queueEndActions.map((action, index) => (
                <Menu.Item
                  key={index}
                  onClick={action.onClick}
                  value={action.label}
                  color="fg.error"
                  _hover={{ bg: 'bg.error', color: 'fg.error' }}
                >
                  <Icon as={action.icon} boxSize="3" />
                  <span>{action.label}</span>
                </Menu.Item>
              ))}
            </Menu.ItemGroup>
            <Menu.Separator />
            <Menu.ItemGroup>
              {queueProcessorActions.map((action, index) => (
                <Menu.Item key={index} onClick={action.onClick} value={action.label}>
                  <Icon as={action.icon} boxSize="3" />
                  <span>{action.label}</span>
                </Menu.Item>
              ))}
            </Menu.ItemGroup>
          </Menu.Content>
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );
};
