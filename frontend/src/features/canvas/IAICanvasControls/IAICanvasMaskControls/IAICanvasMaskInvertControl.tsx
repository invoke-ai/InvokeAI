import { createSelector } from '@reduxjs/toolkit';
import { MdInvertColors, MdInvertColorsOff } from 'react-icons/md';
import { useAppDispatch, useAppSelector } from 'app/store';
import IAIIconButton from 'common/components/IAIIconButton';
import {
  currentCanvasSelector,
  GenericCanvasState,
  setShouldInvertMask,
} from 'features/canvas/canvasSlice';

import _ from 'lodash';
import { activeTabNameSelector } from 'features/options/optionsSelectors';
import { useHotkeys } from 'react-hotkeys-hook';

const canvasMaskInvertSelector = createSelector(
  [currentCanvasSelector, activeTabNameSelector],
  (currentCanvas: GenericCanvasState, activeTabName) => {
    const { isMaskEnabled, shouldInvertMask } = currentCanvas;

    return {
      shouldInvertMask,
      isMaskEnabled,
      activeTabName,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
    },
  }
);

export default function IAICanvasMaskInvertControl() {
  const { shouldInvertMask, isMaskEnabled, activeTabName } = useAppSelector(
    canvasMaskInvertSelector
  );
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
      enabled: true,
    },
    [activeTabName, shouldInvertMask, isMaskEnabled]
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
      isDisabled={!isMaskEnabled}
    />
  );
}
