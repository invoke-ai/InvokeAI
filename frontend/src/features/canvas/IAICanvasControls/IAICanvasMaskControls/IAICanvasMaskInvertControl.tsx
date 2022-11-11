import { createSelector } from '@reduxjs/toolkit';
import { MdInvertColors, MdInvertColorsOff } from 'react-icons/md';
import { useAppDispatch, useAppSelector } from 'app/store';
import IAIIconButton from 'common/components/IAIIconButton';
import {
  currentCanvasSelector,
  GenericCanvasState,
  setShouldPreserveMaskedArea,
} from 'features/canvas/canvasSlice';

import _ from 'lodash';
import { activeTabNameSelector } from 'features/options/optionsSelectors';
import { useHotkeys } from 'react-hotkeys-hook';

const canvasMaskInvertSelector = createSelector(
  [currentCanvasSelector, activeTabNameSelector],
  (currentCanvas: GenericCanvasState, activeTabName) => {
    const { isMaskEnabled, shouldPreserveMaskedArea } = currentCanvas;

    return {
      shouldPreserveMaskedArea,
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
  const { shouldPreserveMaskedArea, isMaskEnabled, activeTabName } = useAppSelector(
    canvasMaskInvertSelector
  );
  const dispatch = useAppDispatch();

  const handleToggleShouldInvertMask = () =>
    dispatch(setShouldPreserveMaskedArea(!shouldPreserveMaskedArea));

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
    [activeTabName, shouldPreserveMaskedArea, isMaskEnabled]
  );

  return (
    <IAIIconButton
      tooltip="Invert Mask Display (Shift+M)"
      aria-label="Invert Mask Display (Shift+M)"
      data-selected={shouldPreserveMaskedArea}
      icon={
        shouldPreserveMaskedArea ? (
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
