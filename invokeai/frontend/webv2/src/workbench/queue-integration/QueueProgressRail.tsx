import type { SystemStyleObject } from '@chakra-ui/react';
import type { CSSProperties } from 'react';

import { Box } from '@chakra-ui/react';
import { useModelLoads } from '@features/models';
import {
  getProgressRailModel,
  getProgressRailSegmentValue,
  getQueueSummary,
  selectProjectProgressItemIds,
} from '@features/queue/contracts';
import { useActiveProgressItemIds, useItemProgress } from '@features/queue/react';
import { useActiveProjectSelector, useWorkbenchSelector } from '@workbench/WorkbenchContext';
import { useMemo } from 'react';

/**
 * Share progress fills across hosts; callers own placement. Divide width among sessions, hide offline/idle, and
 * leave announcements to the queue's throttled live region.
 */
export const QueueProgressRail = ({ css }: { css: SystemStyleObject }) => {
  const queueItems = useActiveProjectSelector((project) => project.queue.items);
  const isConnected = useWorkbenchSelector((snapshot) => snapshot.backendConnection.status === 'connected');
  const isLoadingModels = useModelLoads().length > 0;
  const activeItemIds = useActiveProgressItemIds();

  const sessionItemIds = useMemo(
    () => selectProjectProgressItemIds(queueItems, activeItemIds),
    [activeItemIds, queueItems]
  );

  const model = getProgressRailModel({
    hasOpenWork: getQueueSummary(queueItems).total > 0,
    isConnected,
    sessionItemIds,
  });

  if (model.kind === 'hidden') {
    return null;
  }

  return (
    <Box aria-hidden="true" css={css} data-queue-progress-rail="">
      {model.kind === 'pending' ? (
        <RailSegment isLoadingModels={isLoadingModels} itemId={null} />
      ) : (
        model.itemIds.map((itemId) => <RailSegment isLoadingModels={isLoadingModels} itemId={itemId} key={itemId} />)
      )}
    </Box>
  );
};

const SEGMENT_SX: SystemStyleObject = {
  flex: '1 1 0',
  minWidth: 0,
  overflow: 'hidden',
  position: 'relative',
};

const DETERMINATE_SX: SystemStyleObject = {
  bg: 'accent.solid',
  height: 'full',
  transition: 'width var(--wb-motion-duration-fast) linear',
};

const INDETERMINATE_SX: SystemStyleObject = {
  '--animate-from-x': '-45%',
  '--animate-to-x': '100%',
  animation: 'position 1.1s ease infinite',
  backgroundImage: 'linear-gradient(to right, transparent, {colors.accent.solid}, transparent)',
  insetBlock: 0,
  minWidth: '45%',
  position: 'absolute',
  // Reduced motion uses a static fill; use the raw selector because _reduceMotion is typed only in token values.
  ':root[data-reduce-motion=true] &': {
    animation: 'none',
    backgroundImage: 'none',
    bg: 'accent.solid/40',
    insetInline: 0,
  },
};

/**
 * One session's fill. Each segment subscribes to its own item so a step event
 * on one GPU does not re-render the other GPUs' segments — or the top bar.
 */
const RailSegment = ({ isLoadingModels, itemId }: { isLoadingModels: boolean; itemId: number | null }) => {
  const progress = useItemProgress(itemId);
  const value = getProgressRailSegmentValue({ isLoadingModels, percentage: progress?.percentage });
  const fillStyle = useMemo<CSSProperties>(() => ({ width: value === null ? undefined : `${value * 100}%` }), [value]);

  return (
    <Box css={SEGMENT_SX}>
      {value === null ? <Box css={INDETERMINATE_SX} /> : <Box css={DETERMINATE_SX} style={fillStyle} />}
    </Box>
  );
};
