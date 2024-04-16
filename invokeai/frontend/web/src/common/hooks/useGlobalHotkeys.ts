import { useAppDispatch } from 'app/store/storeHooks';
import { useCancelCurrentQueueItem } from 'features/queue/hooks/useCancelCurrentQueueItem';
import { useClearQueue } from 'features/queue/hooks/useClearQueue';
import { useQueueBack } from 'features/queue/hooks/useQueueBack';
import { useQueueFront } from 'features/queue/hooks/useQueueFront';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { setActiveTab } from 'features/ui/store/uiSlice';
import { useHotkeys } from 'react-hotkeys-hook';

export const useGlobalHotkeys = () => {
  const dispatch = useAppDispatch();
  const isModelManagerEnabled = useFeatureStatus('modelManager');
  const { queueBack, isDisabled: isDisabledQueueBack, isLoading: isLoadingQueueBack } = useQueueBack();

  useHotkeys(
    ['ctrl+enter', 'meta+enter'],
    queueBack,
    {
      enabled: () => !isDisabledQueueBack && !isLoadingQueueBack,
      preventDefault: true,
      enableOnFormTags: ['input', 'textarea', 'select'],
    },
    [queueBack, isDisabledQueueBack, isLoadingQueueBack]
  );

  const { queueFront, isDisabled: isDisabledQueueFront, isLoading: isLoadingQueueFront } = useQueueFront();

  useHotkeys(
    ['ctrl+shift+enter', 'meta+shift+enter'],
    queueFront,
    {
      enabled: () => !isDisabledQueueFront && !isLoadingQueueFront,
      preventDefault: true,
      enableOnFormTags: ['input', 'textarea', 'select'],
    },
    [queueFront, isDisabledQueueFront, isLoadingQueueFront]
  );

  const {
    cancelQueueItem,
    isDisabled: isDisabledCancelQueueItem,
    isLoading: isLoadingCancelQueueItem,
  } = useCancelCurrentQueueItem();

  useHotkeys(
    ['shift+x'],
    cancelQueueItem,
    {
      enabled: () => !isDisabledCancelQueueItem && !isLoadingCancelQueueItem,
      preventDefault: true,
    },
    [cancelQueueItem, isDisabledCancelQueueItem, isLoadingCancelQueueItem]
  );

  const { clearQueue, isDisabled: isDisabledClearQueue, isLoading: isLoadingClearQueue } = useClearQueue();

  useHotkeys(
    ['ctrl+shift+x', 'meta+shift+x'],
    clearQueue,
    {
      enabled: () => !isDisabledClearQueue && !isLoadingClearQueue,
      preventDefault: true,
    },
    [clearQueue, isDisabledClearQueue, isLoadingClearQueue]
  );

  useHotkeys(
    '1',
    () => {
      dispatch(setActiveTab('txt2img'));
    },
    [dispatch]
  );

  useHotkeys(
    '2',
    () => {
      dispatch(setActiveTab('img2img'));
    },
    [dispatch]
  );

  useHotkeys(
    '3',
    () => {
      dispatch(setActiveTab('unifiedCanvas'));
    },
    [dispatch]
  );

  useHotkeys(
    '4',
    () => {
      dispatch(setActiveTab('nodes'));
    },
    [dispatch]
  );

  useHotkeys(
    '5',
    () => {
      if (isModelManagerEnabled) {
        dispatch(setActiveTab('modelManager'));
      }
    },
    [dispatch, isModelManagerEnabled]
  );

  useHotkeys(
    isModelManagerEnabled ? '6' : '5',
    () => {
      dispatch(setActiveTab('queue'));
    },
    [dispatch, isModelManagerEnabled]
  );
};
