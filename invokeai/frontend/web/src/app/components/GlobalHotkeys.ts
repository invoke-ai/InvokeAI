import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useQueueBack } from 'features/queue/hooks/useQueueBack';
import { useQueueFront } from 'features/queue/hooks/useQueueFront';
import {
  ctrlKeyPressed,
  metaKeyPressed,
  shiftKeyPressed,
} from 'features/ui/store/hotkeysSlice';
import { setActiveTab } from 'features/ui/store/uiSlice';
import React, { memo } from 'react';
import { isHotkeyPressed, useHotkeys } from 'react-hotkeys-hook';

const globalHotkeysSelector = createMemoizedSelector(
  [stateSelector],
  ({ hotkeys }) => {
    const { shift, ctrl, meta } = hotkeys;
    return { shift, ctrl, meta };
  }
);

// TODO: Does not catch keypresses while focused in an input. Maybe there is a way?

/**
 * Logical component. Handles app-level global hotkeys.
 * @returns null
 */
const GlobalHotkeys: React.FC = () => {
  const dispatch = useAppDispatch();
  const { shift, ctrl, meta } = useAppSelector(globalHotkeysSelector);
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
    '*',
    () => {
      if (isHotkeyPressed('shift')) {
        !shift && dispatch(shiftKeyPressed(true));
      } else {
        shift && dispatch(shiftKeyPressed(false));
      }
      if (isHotkeyPressed('ctrl')) {
        !ctrl && dispatch(ctrlKeyPressed(true));
      } else {
        ctrl && dispatch(ctrlKeyPressed(false));
      }
      if (isHotkeyPressed('meta')) {
        !meta && dispatch(metaKeyPressed(true));
      } else {
        meta && dispatch(metaKeyPressed(false));
      }
    },
    { keyup: true, keydown: true },
    [shift, ctrl, meta]
  );

  useHotkeys('1', () => {
    dispatch(setActiveTab('txt2img'));
  });

  useHotkeys('2', () => {
    dispatch(setActiveTab('img2img'));
  });

  useHotkeys('3', () => {
    dispatch(setActiveTab('unifiedCanvas'));
  });

  useHotkeys('4', () => {
    dispatch(setActiveTab('nodes'));
  });

  useHotkeys('5', () => {
    dispatch(setActiveTab('modelManager'));
  });

  return null;
};

export default memo(GlobalHotkeys);
