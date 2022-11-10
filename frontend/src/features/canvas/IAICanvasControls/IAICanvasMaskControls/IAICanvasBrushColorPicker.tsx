import { RgbaColor } from 'react-colorful';
import { useAppDispatch, useAppSelector } from 'app/store';
import IAIColorPicker from 'common/components/IAIColorPicker';
import {
  currentCanvasSelector,
  GenericCanvasState,
  setMaskColor,
} from 'features/canvas/canvasSlice';

import _ from 'lodash';
import { createSelector } from '@reduxjs/toolkit';
import { activeTabNameSelector } from 'features/options/optionsSelectors';
import { useHotkeys } from 'react-hotkeys-hook';

const selector = createSelector(
  [currentCanvasSelector, activeTabNameSelector],
  (currentCanvas: GenericCanvasState, activeTabName) => {
    const { brushColor } = currentCanvas;

    return {
      brushColor,
      activeTabName,
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
  const { brushColor, activeTabName } = useAppSelector(selector);

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
      enabled: true,
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
      enabled: true,
    },
    [activeTabName, brushColor.a]
  );

  return (
    <IAIColorPicker color={brushColor} onChange={handleChangeBrushColor} />
  );
}
