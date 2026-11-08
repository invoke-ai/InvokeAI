import { Badge, Portal } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { selectIsAuthenticated } from 'features/auth/store/authSlice';
import type { RefObject } from 'react';
import { memo, useEffect, useMemo, useState } from 'react';
import { useGetQueueStatusQuery } from 'services/api/endpoints/queue';
import type { components } from 'services/api/schema';

type Props = {
  targetRef: RefObject<HTMLDivElement | null>;
};

type SessionQueueStatus = components['schemas']['SessionQueueStatus'];

/**
 * Determines if per-user queue counts are available. The backend only populates
 * user_pending/user_in_progress for non-admin callers in multiuser mode.
 */
const hasUserCounts = (queueData: SessionQueueStatus): boolean => {
  return (
    queueData.user_pending !== undefined &&
    queueData.user_pending !== null &&
    queueData.user_in_progress !== undefined &&
    queueData.user_in_progress !== null
  );
};

/**
 * Calculates the appropriate badge text based on queue status and authentication state.
 * Returns null if badge should be hidden.
 *
 * The backend reports global aggregate counts (pending/in_progress) plus, for non-admin
 * users, their own counts (user_pending/user_in_progress). In multiuser mode this renders
 * as "X/Y" where X is the user's own pending jobs and Y is the global total. In single-user
 * mode (or for admins, where per-user counts are absent) it renders just the total.
 */
const getBadgeText = (queueData: SessionQueueStatus | undefined, isAuthenticated: boolean): string | null => {
  if (!queueData) {
    return null;
  }

  const totalPending = queueData.pending + queueData.in_progress;

  if (totalPending === 0) {
    return null;
  }

  if (isAuthenticated && hasUserCounts(queueData)) {
    const userPending = queueData.user_pending! + queueData.user_in_progress!;
    return `${userPending}/${totalPending}`;
  }

  return totalPending.toString();
};

export const QueueCountBadge = memo(({ targetRef }: Props) => {
  const [badgePos, setBadgePos] = useState<{ x: string; y: string } | null>(null);
  const isAuthenticated = useAppSelector(selectIsAuthenticated);
  const { queueData } = useGetQueueStatusQuery(undefined, {
    selectFromResult: (res) => ({
      queueData: res.data?.queue,
    }),
  });

  const badgeText = useMemo(() => getBadgeText(queueData, isAuthenticated), [queueData, isAuthenticated]);

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
