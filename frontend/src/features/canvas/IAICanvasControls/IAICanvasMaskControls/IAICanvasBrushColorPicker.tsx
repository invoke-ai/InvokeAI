import React from 'react';
import { RgbaColor } from 'react-colorful';
import { FaPalette } from 'react-icons/fa';
import { useAppDispatch, useAppSelector } from 'app/store';
import IAIColorPicker from 'common/components/IAIColorPicker';
import IAIIconButton from 'common/components/IAIIconButton';
import IAIPopover from 'common/components/IAIPopover';
import {
  areHotkeysEnabledSelector,
  currentCanvasSelector,
  GenericCanvasState,
  setMaskColor,
} from 'features/canvas/canvasSlice';

import _ from 'lodash';
import { createSelector } from '@reduxjs/toolkit';
import { activeTabNameSelector } from 'features/options/optionsSelectors';
import { useHotkeys } from 'react-hotkeys-hook';

const selector = createSelector(
  [currentCanvasSelector, activeTabNameSelector, areHotkeysEnabledSelector],
  (currentCanvas: GenericCanvasState, activeTabName, areHotkeysEnabled) => {
    const { brushColor } = currentCanvas;

    return {
      brushColor,
      activeTabName,
      areHotkeysEnabled,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
    },
  }
);

export default function IAICanvasBrushColorPicker() {
  const dispatch = useAppDispatch();
  const { brushColor, activeTabName, areHotkeysEnabled } =
    useAppSelector(selector);

  const handleChangeBrushColor = (newColor: RgbaColor) => {
    dispatch(setMaskColor(newColor));
  };

  // Decrease brush opacity
  useHotkeys(
    'shift+[',
    (e: KeyboardEvent) => {
      e.preventDefault();
      handleChangeBrushColor({
        ...brushColor,
        a: Math.max(brushColor.a - 0.05, 0),
      });
    },
    {
      enabled: areHotkeysEnabled,
    },
    [activeTabName, brushColor.a]
  );

  // Increase brush opacity
  useHotkeys(
    'shift+]',
    (e: KeyboardEvent) => {
      e.preventDefault();
      handleChangeBrushColor({
        ...brushColor,
        a: Math.min(brushColor.a + 0.05, 1),
      });
    },
    {
      enabled: areHotkeysEnabled,
    },
    [activeTabName, brushColor.a]
  );

  return (
    <IAIColorPicker color={brushColor} onChange={handleChangeBrushColor} />
  );
}
