import { useAppDispatch } from 'app/store/storeHooks';
import { useQueueBack } from 'features/queue/hooks/useQueueBack';
import { useQueueFront } from 'features/queue/hooks/useQueueFront';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { setActiveTab } from 'features/ui/store/uiSlice';
import { useHotkeys } from 'react-hotkeys-hook';

export const useGlobalHotkeys = () => {
  const dispatch = useAppDispatch();
  const isModelManagerEnabled =
    useFeatureStatus('modelManager').isFeatureEnabled;
  const {
    queueBack,
    isDisabled: isDisabledQueueBack,
    isLoading: isLoadingQueueBack,
  } = useQueueBack();

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

  const {
    queueFront,
    isDisabled: isDisabledQueueFront,
    isLoading: isLoadingQueueFront,
  } = useQueueFront();

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
