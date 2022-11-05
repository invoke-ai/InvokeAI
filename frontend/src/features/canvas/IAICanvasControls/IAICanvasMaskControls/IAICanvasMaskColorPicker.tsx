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

const maskColorPickerSelector = createSelector(
  [currentCanvasSelector, activeTabNameSelector, areHotkeysEnabledSelector],
  (currentCanvas: GenericCanvasState, activeTabName, areHotkeysEnabled) => {
    const { shouldShowMask, maskColor } = currentCanvas;

    return {
      shouldShowMask,
      maskColor,
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

export default function IAICanvasMaskColorPicker() {
  const { shouldShowMask, maskColor, activeTabName, areHotkeysEnabled } =
    useAppSelector(maskColorPickerSelector);
  const dispatch = useAppDispatch();
  const handleChangeMaskColor = (newColor: RgbaColor) => {
    dispatch(setMaskColor(newColor));
  };

  // Hotkeys
  // Decrease mask opacity
  useHotkeys(
    'shift+[',
    (e: KeyboardEvent) => {
      e.preventDefault();
      handleChangeMaskColor({
        ...maskColor,
        a: Math.max(maskColor.a - 0.05, 0),
      });
    },
    {
      enabled: areHotkeysEnabled,
    },
    [activeTabName, shouldShowMask, maskColor.a]
  );

  // Increase mask opacity
  useHotkeys(
    'shift+]',
    (e: KeyboardEvent) => {
      e.preventDefault();
      handleChangeMaskColor({
        ...maskColor,
        a: Math.min(maskColor.a + 0.05, 1),
      });
    },
    {
      enabled: areHotkeysEnabled,
    },
    [activeTabName, shouldShowMask, maskColor.a]
  );

  return (
    <IAIPopover
      trigger="hover"
      styleClass="inpainting-color-picker"
      triggerComponent={
        <IAIIconButton
          aria-label="Mask Color"
          icon={<FaPalette />}
          isDisabled={!shouldShowMask}
          cursor={'pointer'}
        />
      }
    >
      <IAIColorPicker color={maskColor} onChange={handleChangeMaskColor} />
    </IAIPopover>
  );
}
