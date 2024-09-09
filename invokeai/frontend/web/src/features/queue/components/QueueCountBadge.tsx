import { Badge, Portal } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { $isLeftPanelOpen, TABS_WITH_LEFT_PANEL } from 'features/ui/store/uiSlice';
import type { RefObject } from 'react';
import { memo, useEffect, useState } from 'react';
import { useGetQueueStatusQuery } from 'services/api/endpoints/queue';

type Props = {
  targetRef: RefObject<HTMLDivElement>;
};

const selectActiveTabShouldShowBadge = createSelector(selectActiveTab, (activeTab) =>
  TABS_WITH_LEFT_PANEL.includes(activeTab)
);

export const QueueCountBadge = memo(({ targetRef }: Props) => {
  const [badgePos, setBadgePos] = useState<{ x: string; y: string } | null>(null);
  const activeTabShouldShowBadge = useAppSelector(selectActiveTabShouldShowBadge);
  const isParametersPanelOpen = useStore($isLeftPanelOpen);
  const { queueSize } = useGetQueueStatusQuery(undefined, {
    selectFromResult: (res) => ({
      queueSize: res.data ? res.data.queue.pending + res.data.queue.in_progress : 0,
    }),
  });

  useEffect(() => {
    if (!targetRef.current) {
      return;
    }

    const target = targetRef.current;
    const parent = target.parentElement;

    if (!parent) {
      return;
    }

    const cb = () => {
      if (!$isLeftPanelOpen.get()) {
        return;
      }
      const { x, y } = target.getBoundingClientRect();
      setBadgePos({ x: `${x - 7}px`, y: `${y - 5}px` });
    };

    const resizeObserver = new ResizeObserver(cb);
    resizeObserver.observe(parent);
    cb();

    return () => {
      resizeObserver.disconnect();
    };
  }, [targetRef]);

  if (queueSize === 0) {
    return null;
  }
  if (!badgePos) {
    return null;
  }
  if (!isParametersPanelOpen) {
    return null;
  }
  if (!activeTabShouldShowBadge) {
    return null;
  }

  return (
    <Portal>
      <Badge
        pos="absolute"
        insetInlineStart={badgePos.x}
        insetBlockStart={badgePos.y}
        colorScheme="invokeYellow"
        zIndex="docked"
        shadow="dark-lg"
        userSelect="none"
      >
        {queueSize}
      </Badge>
    </Portal>
  );
});

QueueCountBadge.displayName = 'QueueCountBadge';
