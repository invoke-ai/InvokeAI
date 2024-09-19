import { useAppDispatch } from 'app/store/storeHooks';
import { addScope, removeScope, setScopes } from 'common/hooks/interactionScopes';
import { useClearQueue } from 'features/queue/components/ClearQueueConfirmationAlertDialog';
import { useCancelCurrentQueueItem } from 'features/queue/hooks/useCancelCurrentQueueItem';
import { useInvoke } from 'features/queue/hooks/useInvoke';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { setActiveTab } from 'features/ui/store/uiSlice';

export const useGlobalHotkeys = () => {
  const dispatch = useAppDispatch();
  const isModelManagerEnabled = useFeatureStatus('modelManager');
  const queue = useInvoke();

  useRegisteredHotkeys({
    id: 'invoke',
    category: 'app',
    callback: queue.queueBack,
    options: {
      enabled: !queue.isDisabled && !queue.isLoading,
      preventDefault: true,
      enableOnFormTags: ['input', 'textarea', 'select'],
    },
    dependencies: [queue],
  });

  useRegisteredHotkeys({
    id: 'invokeFront',
    category: 'app',
    callback: queue.queueFront,
    options: {
      enabled: !queue.isDisabled && !queue.isLoading,
      preventDefault: true,
      enableOnFormTags: ['input', 'textarea', 'select'],
    },
    dependencies: [queue],
  });

  const {
    cancelQueueItem,
    isDisabled: isDisabledCancelQueueItem,
    isLoading: isLoadingCancelQueueItem,
  } = useCancelCurrentQueueItem();

  useRegisteredHotkeys({
    id: 'cancelQueueItem',
    category: 'app',
    callback: cancelQueueItem,
    options: {
      enabled: !isDisabledCancelQueueItem && !isLoadingCancelQueueItem,
      preventDefault: true,
    },
    dependencies: [cancelQueueItem, isDisabledCancelQueueItem, isLoadingCancelQueueItem],
  });

  const { clearQueue, isDisabled: isDisabledClearQueue, isLoading: isLoadingClearQueue } = useClearQueue();

  useRegisteredHotkeys({
    id: 'clearQueue',
    category: 'app',
    callback: clearQueue,
    options: {
      enabled: !isDisabledClearQueue && !isLoadingClearQueue,
      preventDefault: true,
    },
    dependencies: [clearQueue, isDisabledClearQueue, isLoadingClearQueue],
  });

  useRegisteredHotkeys({
    id: 'selectCanvasTab',
    category: 'app',
    callback: () => {
      dispatch(setActiveTab('canvas'));
      addScope('canvas');
      removeScope('workflows');
    },
    dependencies: [dispatch],
  });

  useRegisteredHotkeys({
    id: 'selectUpscalingTab',
    category: 'app',
    callback: () => {
      dispatch(setActiveTab('upscaling'));
      removeScope('canvas');
      removeScope('workflows');
    },
    dependencies: [dispatch],
  });

  useRegisteredHotkeys({
    id: 'selectWorkflowsTab',
    category: 'app',
    callback: () => {
      dispatch(setActiveTab('workflows'));
      removeScope('canvas');
      addScope('workflows');
    },
    dependencies: [dispatch],
  });

  useRegisteredHotkeys({
    id: 'selectModelsTab',
    category: 'app',
    callback: () => {
      dispatch(setActiveTab('models'));
      setScopes([]);
    },
    options: {
      enabled: isModelManagerEnabled,
    },
    dependencies: [dispatch, isModelManagerEnabled],
  });

  useRegisteredHotkeys({
    id: 'selectQueueTab',
    category: 'app',
    callback: () => {
      dispatch(setActiveTab('queue'));
      setScopes([]);
    },
    dependencies: [dispatch, isModelManagerEnabled],
  });
};
