import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { requestCanvasRescale } from 'features/canvas/store/thunks/requestCanvasScale';
import {
  ctrlKeyPressed,
  metaKeyPressed,
  shiftKeyPressed,
} from 'features/ui/store/hotkeysSlice';
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
  [stateSelector],
  ({ hotkeys, ui }) => {
    const { shift, ctrl, meta } = hotkeys;
    const { shouldPinParametersPanel, shouldPinGallery } = ui;
    return { shift, ctrl, meta, shouldPinGallery, shouldPinParametersPanel };
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
  const { shift, ctrl, meta, shouldPinParametersPanel, shouldPinGallery } =
    useAppSelector(globalHotkeysSelector);
  const activeTabName = useAppSelector(activeTabNameSelector);

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

  useHotkeys('o', () => {
    dispatch(toggleParametersPanel());
    if (activeTabName === 'unifiedCanvas' && shouldPinParametersPanel) {
      dispatch(requestCanvasRescale());
    }
  });

  useHotkeys(['shift+o'], () => {
    dispatch(togglePinParametersPanel());
    if (activeTabName === 'unifiedCanvas') {
      dispatch(requestCanvasRescale());
    }
  });

  useHotkeys('g', () => {
    dispatch(toggleGalleryPanel());
    if (activeTabName === 'unifiedCanvas' && shouldPinGallery) {
      dispatch(requestCanvasRescale());
    }
  });

  useHotkeys(['shift+g'], () => {
    dispatch(togglePinGalleryPanel());
    if (activeTabName === 'unifiedCanvas') {
      dispatch(requestCanvasRescale());
    }
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
