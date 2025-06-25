import { useAppDispatch } from 'app/store/storeHooks';
import { useClearQueue } from 'features/queue/hooks/useClearQueue';
import { useDeleteCurrentQueueItem } from 'features/queue/hooks/useDeleteCurrentQueueItem';
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
    callback: queue.enqueueBack,
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
    callback: queue.enqueueFront,
    options: {
      enabled: !queue.isDisabled && !queue.isLoading,
      preventDefault: true,
      enableOnFormTags: ['input', 'textarea', 'select'],
    },
    dependencies: [queue],
  });

  const deleteCurrentQueueItem = useDeleteCurrentQueueItem();

  useRegisteredHotkeys({
    id: 'cancelQueueItem',
    category: 'app',
    callback: deleteCurrentQueueItem.trigger,
    options: {
      enabled: !deleteCurrentQueueItem.isDisabled && !deleteCurrentQueueItem.isLoading,
      preventDefault: true,
    },
    dependencies: [deleteCurrentQueueItem],
  });

  const clearQueue = useClearQueue();

  useRegisteredHotkeys({
    id: 'clearQueue',
    category: 'app',
    callback: clearQueue.trigger,
    options: {
      enabled: !clearQueue.isDisabled && !clearQueue.isLoading,
      preventDefault: true,
    },
    dependencies: [clearQueue],
  });

  useRegisteredHotkeys({
    id: 'selectGenerateTab',
    category: 'app',
    callback: () => {
      dispatch(setActiveTab('generate'));
    },
    dependencies: [dispatch],
  });

  useRegisteredHotkeys({
    id: 'selectCanvasTab',
    category: 'app',
    callback: () => {
      dispatch(setActiveTab('canvas'));
    },
    dependencies: [dispatch],
  });

  useRegisteredHotkeys({
    id: 'selectUpscalingTab',
    category: 'app',
    callback: () => {
      dispatch(setActiveTab('upscaling'));
    },
    dependencies: [dispatch],
  });

  useRegisteredHotkeys({
    id: 'selectWorkflowsTab',
    category: 'app',
    callback: () => {
      dispatch(setActiveTab('workflows'));
    },
    dependencies: [dispatch],
  });

  useRegisteredHotkeys({
    id: 'selectModelsTab',
    category: 'app',
    callback: () => {
      dispatch(setActiveTab('models'));
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
    },
    dependencies: [dispatch, isModelManagerEnabled],
  });

  // TODO: implement delete - needs to handle gallery focus, which has changed w/ dockview
  // useRegisteredHotkeys({
  //   id: 'deleteSelection',
  //   category: 'gallery',
  //   callback: () => {
  //     if (!selection.length) {
  //       return;
  //     }
  //     deleteImageModal.delete(selection);
  //   },
  //   options: {
  //     enabled: (isGalleryFocused || isImageViewerFocused) && isDeleteEnabledByTab && !isWorkflowsFocused,
  //   },
  //   dependencies: [isWorkflowsFocused, isDeleteEnabledByTab, selection, isWorkflowsFocused],
  // });
};
