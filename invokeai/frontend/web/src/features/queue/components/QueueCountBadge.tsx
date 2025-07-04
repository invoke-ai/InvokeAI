import { Badge, Portal } from '@invoke-ai/ui-library';
import type { RefObject } from 'react';
import { memo, useEffect, useState } from 'react';
import { useGetQueueStatusQuery } from 'services/api/endpoints/queue';

type Props = {
  targetRef: RefObject<HTMLDivElement>;
};

export const QueueCountBadge = memo(({ targetRef }: Props) => {
  const [badgePos, setBadgePos] = useState<{ x: string; y: string } | null>(null);
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
      const { x, y } = target.getBoundingClientRect();
      if (x === 0 || y === 0) {
        // If the target is not visible, do not show the badge
        setBadgePos(null);
        return;
      }
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
