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
 * The application's primary generation indicator: a hairline that lights up
 * along the whole bottom edge of the top bar while work is in flight.
 *
 * Everything else that shows progress — the Invoke icon ring, the queue
 * readout, the preview overlay, the gallery placeholders — is small, and most
 * of it only exists if the user happens to have that widget on screen. A rail
 * spanning the viewport is detectable without being looked at, and it is
 * present in every layout, which is what the legacy bar under the Invoke button
 * had and this app lost.
 *
 * It sits *over* the top bar's 1px divider rather than below it, so appearing
 * and disappearing costs no reflow and reads as the existing line lighting up
 * rather than as a new element shoving the workspace down.
 *
 * Sessions divide the width instead of stacking: a 2px rail cannot stack to
 * four GPUs without either growing into the workspace or shrinking below
 * visibility, and side-by-side segments filling in parallel say "four things are
 * running" at a glance.
 *
 * Not exposed to assistive tech — the queue cluster already owns a throttled
 * live region for this, and a second announcer would only double the chatter.
 */

const RAIL_SX: SystemStyleObject = {
  bottom: '-1px',
  display: 'flex',
  gap: '1px',
  height: '2px',
  insetInline: 0,
  pointerEvents: 'none',
  position: 'absolute',
  zIndex: 3,
};

export const TopbarProgressRail = () => {
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
    <Box aria-hidden="true" css={RAIL_SX}>
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

// Reuses Chakra's built-in `position` keyframe, the same one its own
// indeterminate Progress range animates on.
const INDETERMINATE_SX: SystemStyleObject = {
  '--animate-from-x': '-45%',
  '--animate-to-x': '100%',
  animation: 'position 1.1s ease infinite',
  backgroundImage: 'linear-gradient(to right, transparent, {colors.accent.solid}, transparent)',
  insetBlock: 0,
  minWidth: '45%',
  position: 'absolute',
  // With motion off there is no sweep to fall back on, so the segment becomes a
  // quiet static fill: still "something is running", minus the movement. Spelled
  // as a raw selector because the `_reduceMotion` condition this repo defines is
  // only typed inside token values, not in style objects.
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
