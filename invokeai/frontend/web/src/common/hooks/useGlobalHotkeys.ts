import { useAppStore } from 'app/store/storeHooks';
import { useDeleteImageModalApi } from 'features/deleteImageModal/store/state';
import { selectSelection } from 'features/gallery/store/gallerySelectors';
import { useCancelCurrentQueueItem } from 'features/queue/hooks/useCancelCurrentQueueItem';
import { useClearQueue } from 'features/queue/hooks/useClearQueue';
import { useInvoke } from 'features/queue/hooks/useInvoke';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { navigationApi } from 'features/ui/layouts/navigation-api';

import { getFocusedRegion } from './focus';

export const useGlobalHotkeys = () => {
  const { dispatch, getState } = useAppStore();
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

  const cancelCurrentQueueItem = useCancelCurrentQueueItem();

  useRegisteredHotkeys({
    id: 'cancelQueueItem',
    category: 'app',
    callback: () => {
      cancelCurrentQueueItem.trigger();
    },
    options: {
      enabled: !cancelCurrentQueueItem.isDisabled && !cancelCurrentQueueItem.isLoading,
      preventDefault: true,
    },
    dependencies: [cancelCurrentQueueItem],
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
      navigationApi.switchToTab('generate');
    },
    dependencies: [dispatch],
  });

  useRegisteredHotkeys({
    id: 'selectCanvasTab',
    category: 'app',
    callback: () => {
      navigationApi.switchToTab('canvas');
    },
    dependencies: [dispatch],
  });

  useRegisteredHotkeys({
    id: 'selectUpscalingTab',
    category: 'app',
    callback: () => {
      navigationApi.switchToTab('upscaling');
    },
    dependencies: [dispatch],
  });

  useRegisteredHotkeys({
    id: 'selectWorkflowsTab',
    category: 'app',
    callback: () => {
      navigationApi.switchToTab('workflows');
    },
    dependencies: [dispatch],
  });

  useRegisteredHotkeys({
    id: 'selectModelsTab',
    category: 'app',
    callback: () => {
      navigationApi.switchToTab('models');
    },
    dependencies: [dispatch],
  });

  useRegisteredHotkeys({
    id: 'selectQueueTab',
    category: 'app',
    callback: () => {
      navigationApi.switchToTab('queue');
    },
    dependencies: [dispatch],
  });

  const deleteImageModalApi = useDeleteImageModalApi();

  useRegisteredHotkeys({
    id: 'deleteSelection',
    category: 'gallery',
    callback: () => {
      const focusedRegion = getFocusedRegion();
      if (focusedRegion !== 'gallery' && focusedRegion !== 'viewer') {
        return;
      }
      const selection = selectSelection(getState());
      if (!selection.length) {
        return;
      }
      deleteImageModalApi.delete(selection);
    },
    dependencies: [getState, deleteImageModalApi],
  });

  useRegisteredHotkeys({
    id: 'toggleViewer',
    category: 'viewer',
    callback: () => {
      navigationApi.toggleViewerPanel();
    },
    dependencies: [],
  });
};
