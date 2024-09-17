import { Badge, Portal } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { $isLeftPanelOpen } from 'features/ui/store/uiSlice';
import type { RefObject } from 'react';
import { memo, useEffect, useState } from 'react';
import { useGetQueueStatusQuery } from 'services/api/endpoints/queue';

type Props = {
  targetRef: RefObject<HTMLDivElement>;
};

export const QueueCountBadge = memo(({ targetRef }: Props) => {
  const [badgePos, setBadgePos] = useState<{ x: string; y: string } | null>(null);
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
