import { useStore } from '@nanostores/react';
import { enqueueRequested } from 'app/store/actions';
import { $true } from 'app/store/nanostores/util';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useCanvasManagerSafe } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { $templates } from 'features/nodes/store/nodesSlice';
import {
  buildSelectIsReadyToEnqueueCanvasTab,
  buildSelectIsReadyToEnqueueUpscaleTab,
  buildSelectIsReadyToEnqueueWorkflowsTab,
} from 'features/queue/store/readiness';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { useCallback, useMemo } from 'react';
import { enqueueMutationFixedCacheKeyOptions, useEnqueueBatchMutation } from 'services/api/endpoints/queue';
import { $isConnected } from 'services/events/stores';

export const useInvoke = () => {
  const dispatch = useAppDispatch();
  const tabName = useAppSelector(selectActiveTab);
  const isConnected = useStore($isConnected);
  const canvasManager = useCanvasManagerSafe();
  const canvasIsFiltering = useStore(canvasManager?.stateApi.$isFiltering ?? $true);
  const canvasIsTransforming = useStore(canvasManager?.stateApi.$isTransforming ?? $true);
  const canvasIsRasterizing = useStore(canvasManager?.stateApi.$isRasterizing ?? $true);
  const canvasIsSelectingObject = useStore(canvasManager?.stateApi.$isSegmenting ?? $true);
  const canvasIsCompositing = useStore(canvasManager?.compositor.$isBusy ?? $true);
  const templates = useStore($templates);

  const selectIsReady = useMemo(() => {
    if (tabName === 'canvas') {
      return buildSelectIsReadyToEnqueueCanvasTab({
        isConnected,
        canvasIsFiltering,
        canvasIsTransforming,
        canvasIsRasterizing,
        canvasIsSelectingObject,
        canvasIsCompositing,
      });
    }
    if (tabName === 'upscaling') {
      return buildSelectIsReadyToEnqueueUpscaleTab({ isConnected });
    }
    if (tabName === 'workflows') {
      return buildSelectIsReadyToEnqueueWorkflowsTab({ isConnected, templates });
    }
    return () => false;
  }, [
    tabName,
    isConnected,
    canvasIsFiltering,
    canvasIsTransforming,
    canvasIsRasterizing,
    canvasIsSelectingObject,
    canvasIsCompositing,
    templates,
  ]);

  const isReady = useAppSelector(selectIsReady);

  const [_, { isLoading }] = useEnqueueBatchMutation(enqueueMutationFixedCacheKeyOptions);
  const queueBack = useCallback(() => {
    if (!isReady) {
      return;
    }
    dispatch(enqueueRequested({ tabName, prepend: false }));
  }, [dispatch, isReady, tabName]);
  const queueFront = useCallback(() => {
    if (!isReady) {
      return;
    }
    dispatch(enqueueRequested({ tabName, prepend: true }));
  }, [dispatch, isReady, tabName]);

  return { queueBack, queueFront, isLoading, isDisabled: !isReady };
};
