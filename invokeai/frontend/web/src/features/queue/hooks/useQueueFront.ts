import { enqueueRequested } from 'app/store/actions';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useIsReadyToEnqueue } from 'common/hooks/useIsReadyToEnqueue';
import { clampSymmetrySteps } from 'features/parameters/store/generationSlice';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { useCallback, useMemo } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useEnqueueBatchMutation } from 'services/api/endpoints/queue';

export const useQueueFront = () => {
  const dispatch = useAppDispatch();
  const tabName = useAppSelector(activeTabNameSelector);
  const { isReady } = useIsReadyToEnqueue();
  const [_, { isLoading }] = useEnqueueBatchMutation({
    fixedCacheKey: 'enqueueBatch',
  });
  const isDisabled = useMemo(() => !isReady, [isReady]);
  const queueFront = useCallback(() => {
    if (isDisabled) {
      return;
    }
    dispatch(clampSymmetrySteps());
    dispatch(enqueueRequested({ tabName, prepend: true }));
  }, [dispatch, isDisabled, tabName]);

  useHotkeys(
    ['ctrl+shift+enter', 'meta+shift+enter'],
    queueFront,
    {
      enabled: () => !isDisabled && !isLoading,
      preventDefault: true,
      enableOnFormTags: ['input', 'textarea', 'select'],
    },
    [isLoading, isDisabled, tabName]
  );

  return { queueFront, isLoading, isDisabled };
};
