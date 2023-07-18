import { createSelector } from '@reduxjs/toolkit';
import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { requestCanvasRescale } from 'features/canvas/store/thunks/requestCanvasScale';
import { shiftKeyPressed } from 'features/ui/store/hotkeysSlice';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import {
  setActiveTab,
  toggleGalleryPanel,
  toggleParametersPanel,
  togglePinGalleryPanel,
  togglePinParametersPanel,
} from 'features/ui/store/uiSlice';
import { isEqual } from 'lodash-es';
import React, { memo } from 'react';
import { isHotkeyPressed, useHotkeys } from 'react-hotkeys-hook';

const globalHotkeysSelector = createSelector(
  (state: RootState) => state.hotkeys,
  (hotkeys) => {
    const { shift } = hotkeys;
    return { shift };
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
  const { shift } = useAppSelector(globalHotkeysSelector);
  const activeTabName = useAppSelector(activeTabNameSelector);

  useHotkeys(
    '*',
    () => {
      if (isHotkeyPressed('shift')) {
        !shift && dispatch(shiftKeyPressed(true));
      } else {
        shift && dispatch(shiftKeyPressed(false));
      }
    },
    { keyup: true, keydown: true },
    [shift]
  );

  useHotkeys('o', () => {
    dispatch(toggleParametersPanel());
    if (activeTabName === 'unifiedCanvas') {
      dispatch(requestCanvasRescale());
    }
  });

  useHotkeys(['shift+o'], () => {
    dispatch(togglePinParametersPanel());
  });

  useHotkeys('g', () => {
    dispatch(toggleGalleryPanel());
    if (activeTabName === 'unifiedCanvas') {
      dispatch(requestCanvasRescale());
    }
  });

  useHotkeys(['shift+g'], () => {
    dispatch(togglePinGalleryPanel());
  });

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

  return null;
};

export default memo(GlobalHotkeys);
