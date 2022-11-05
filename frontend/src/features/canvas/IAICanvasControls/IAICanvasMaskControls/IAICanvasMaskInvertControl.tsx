import { createSelector } from '@reduxjs/toolkit';
import React from 'react';
import { MdInvertColors, MdInvertColorsOff } from 'react-icons/md';
import { RootState, useAppDispatch, useAppSelector } from 'app/store';
import IAIIconButton from 'common/components/IAIIconButton';
import {
  areHotkeysEnabledSelector,
  currentCanvasSelector,
  GenericCanvasState,
  setShouldInvertMask,
} from 'features/canvas/canvasSlice';

import _ from 'lodash';
import { activeTabNameSelector } from 'features/options/optionsSelectors';
import { useHotkeys } from 'react-hotkeys-hook';

const canvasMaskInvertSelector = createSelector(
  [currentCanvasSelector, activeTabNameSelector, areHotkeysEnabledSelector],
  (currentCanvas: GenericCanvasState, activeTabName, areHotkeysEnabled) => {
    const { shouldShowMask, shouldInvertMask } = currentCanvas;

    return {
      shouldInvertMask,
      shouldShowMask,
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

export default function IAICanvasMaskInvertControl() {
  const { shouldInvertMask, shouldShowMask, activeTabName, areHotkeysEnabled } =
    useAppSelector(canvasMaskInvertSelector);
  const dispatch = useAppDispatch();

  const handleToggleShouldInvertMask = () =>
    dispatch(setShouldInvertMask(!shouldInvertMask));

  // Invert mask
  useHotkeys(
    'shift+m',
    (e: KeyboardEvent) => {
      e.preventDefault();
      handleToggleShouldInvertMask();
    },
    {
      enabled: areHotkeysEnabled,
    },
    [activeTabName, shouldInvertMask, shouldShowMask]
  );

  return (
    <IAIIconButton
      tooltip="Invert Mask Display (Shift+M)"
      aria-label="Invert Mask Display (Shift+M)"
      data-selected={shouldInvertMask}
      icon={
        shouldInvertMask ? (
          <MdInvertColors size={22} />
        ) : (
          <MdInvertColorsOff size={22} />
        )
      }
      onClick={handleToggleShouldInvertMask}
      isDisabled={!shouldShowMask}
    />
  );
}
