import { enqueueRequested } from 'app/store/actions';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useIsReadyToEnqueue } from 'common/hooks/useIsReadyToEnqueue';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { useCallback, useMemo } from 'react';
import { useEnqueueBatchMutation } from 'services/api/endpoints/queue';

export const useQueueFront = () => {
  const dispatch = useAppDispatch();
  const tabName = useAppSelector(activeTabNameSelector);
  const { isReady } = useIsReadyToEnqueue();
  const [_, { isLoading }] = useEnqueueBatchMutation({
    fixedCacheKey: 'enqueueBatch',
  });
  const prependEnabled = useFeatureStatus('prependQueue').isFeatureEnabled;

  const isDisabled = useMemo(() => {
    return !isReady || !prependEnabled;
  }, [isReady, prependEnabled]);

  const queueFront = useCallback(() => {
    if (isDisabled) {
      return;
    }
    dispatch(enqueueRequested({ tabName, prepend: true }));
  }, [dispatch, isDisabled, tabName]);

  return { queueFront, isLoading, isDisabled };
};
