import { Badge, Portal } from '@invoke-ai/ui-library';
import type { RefObject } from 'react';
import { memo, useEffect, useMemo, useState } from 'react';
import { useGetQueueStatusQuery } from 'services/api/endpoints/queue';
import type { components } from 'services/api/schema';

type Props = {
  targetRef: RefObject<HTMLDivElement>;
};

type SessionQueueStatus = components['schemas']['SessionQueueStatus'];

/**
 * Calculates the appropriate badge text based on queue status.
 * Returns null if badge should be hidden.
 *
 * In multiuser mode, the backend already scopes counts to the current user for non-admins,
 * so pending + in_progress reflects the user's own queue items.
 */
const getBadgeText = (queueData: SessionQueueStatus | undefined): string | null => {
  if (!queueData) {
    return null;
  }

  const totalPending = queueData.pending + queueData.in_progress;

  if (totalPending === 0) {
    return null;
  }

  return totalPending.toString();
};

export const QueueCountBadge = memo(({ targetRef }: Props) => {
  const [badgePos, setBadgePos] = useState<{ x: string; y: string } | null>(null);
  const { queueData } = useGetQueueStatusQuery(undefined, {
    selectFromResult: (res) => ({
      queueData: res.data?.queue,
    }),
  });

  const badgeText = useMemo(() => getBadgeText(queueData), [queueData]);

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
      // If the parent element is not visible, we do not want to show the badge. This can be tricky to reliably
      // determine. The best way I've found is to check the bounding rect of the target and its parent.
      const badgeElRect = target.getBoundingClientRect();
      const parentElRect = parent.getBoundingClientRect();
      if (
        badgeElRect.x === 0 ||
        badgeElRect.y === 0 ||
        badgeElRect.width === 0 ||
        badgeElRect.height === 0 ||
        parentElRect.x === 0 ||
        parentElRect.y === 0 ||
        parentElRect.width === 0 ||
        parentElRect.height === 0
      ) {
        setBadgePos(null);
        return;
      }
      setBadgePos({ x: `${badgeElRect.x - 7}px`, y: `${badgeElRect.y - 5}px` });
    };

    const resizeObserver = new ResizeObserver(cb);
    resizeObserver.observe(parent);
    cb();

    return () => {
      resizeObserver.disconnect();
    };
  }, [targetRef]);

  if (!badgeText) {
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
        {badgeText}
      </Badge>
    </Portal>
  );
});

QueueCountBadge.displayName = 'QueueCountBadge';
