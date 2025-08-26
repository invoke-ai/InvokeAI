import { useAppStore } from 'app/store/storeHooks';
import { useDeleteImageModalApi } from 'features/deleteImageModal/store/state';
import { useDeleteVideoModalApi } from 'features/deleteVideoModal/store/state';
import { selectSelection } from 'features/gallery/store/gallerySelectors';
import { useClearQueue } from 'features/queue/hooks/useClearQueue';
import { useDeleteCurrentQueueItem } from 'features/queue/hooks/useDeleteCurrentQueueItem';
import { useInvoke } from 'features/queue/hooks/useInvoke';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { navigationApi } from 'features/ui/layouts/navigation-api';

import { getFocusedRegion } from './focus';

export const useGlobalHotkeys = () => {
  const { dispatch, getState } = useAppStore();
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
    options: {
      enabled: isModelManagerEnabled,
    },
    dependencies: [dispatch, isModelManagerEnabled],
  });

  useRegisteredHotkeys({
    id: 'selectQueueTab',
    category: 'app',
    callback: () => {
      navigationApi.switchToTab('queue');
    },
    dependencies: [dispatch, isModelManagerEnabled],
  });

  const deleteImageModalApi = useDeleteImageModalApi();
  const deleteVideoModalApi = useDeleteVideoModalApi();

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
      if (selection.every(({ type }) => type === 'image')) {
        deleteImageModalApi.delete(selection.map((s) => s.id));
      } else if (selection.every(({ type }) => type === 'video')) {
        deleteVideoModalApi.delete(selection.map((s) => s.id));
      } else {
        // no-op, we expect selections to always be only images or only video
      }
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
