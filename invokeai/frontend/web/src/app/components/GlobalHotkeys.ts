import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  ctrlKeyPressed,
  metaKeyPressed,
  shiftKeyPressed,
} from 'features/ui/store/hotkeysSlice';
import { setActiveTab } from 'features/ui/store/uiSlice';
import { isEqual } from 'lodash-es';
import React, { memo } from 'react';
import { isHotkeyPressed, useHotkeys } from 'react-hotkeys-hook';

const globalHotkeysSelector = createSelector(
  [stateSelector],
  ({ hotkeys }) => {
    const { shift, ctrl, meta } = hotkeys;
    return { shift, ctrl, meta };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
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
