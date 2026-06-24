import { useAppSelector } from 'app/store/storeHooks';
import { selectCurrentUser } from 'features/auth/store/authSlice';
import { useCurrentQueueItemId } from 'features/queue/hooks/useCurrentQueueItemId';
import { useCallback, useMemo } from 'react';
import { useGetSetupStatusQuery } from 'services/api/endpoints/auth';
import { useGetCurrentQueueItemQuery } from 'services/api/endpoints/queue';

import { useCancelQueueItem } from './useCancelQueueItem';

export const useCancelCurrentQueueItem = () => {
  const currentQueueItemId = useCurrentQueueItemId();
  const { data: currentQueueItem } = useGetCurrentQueueItemQuery();
  const currentUser = useAppSelector(selectCurrentUser);
  const { data: setupStatus } = useGetSetupStatusQuery();
  const cancelQueueItem = useCancelQueueItem();

  // Check if current user can cancel the current item
  const canCancelCurrentItem = useMemo(() => {
    // In single-user mode, allow canceling current item without auth checks.
    if (setupStatus && !setupStatus.multiuser_enabled) {
      return true;
    }

    if (!currentUser || !currentQueueItem) {
      return false;
    }
    // Admin users can cancel all items
    if (currentUser.is_admin) {
      return true;
    }
    // Non-admin users can only cancel their own items
    return currentQueueItem.user_id === currentUser.user_id;
  }, [setupStatus, currentUser, currentQueueItem]);

  const trigger = useCallback(
    (options?: { withToast?: boolean }) => {
      if (currentQueueItemId === null) {
        return;
      }
      cancelQueueItem.trigger(currentQueueItemId, options);
    },
    [currentQueueItemId, cancelQueueItem]
  );

  return {
    trigger,
    isLoading: cancelQueueItem.isLoading,
    isDisabled: cancelQueueItem.isDisabled || currentQueueItemId === null || !canCancelCurrentItem,
  };
};
