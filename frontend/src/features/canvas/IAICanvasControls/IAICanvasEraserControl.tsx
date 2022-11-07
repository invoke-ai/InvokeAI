import { createSelector } from '@reduxjs/toolkit';
import { useHotkeys } from 'react-hotkeys-hook';
import { FaEraser } from 'react-icons/fa';
import { useAppDispatch, useAppSelector } from 'app/store';
import IAIIconButton from 'common/components/IAIIconButton';
import {
  areHotkeysEnabledSelector,
  currentCanvasSelector,
  GenericCanvasState,
  setTool,
} from 'features/canvas/canvasSlice';

import _ from 'lodash';
import { activeTabNameSelector } from 'features/options/optionsSelectors';

const eraserSelector = createSelector(
  [currentCanvasSelector, activeTabNameSelector, areHotkeysEnabledSelector],
  (currentCanvas: GenericCanvasState, activeTabName, areHotkeysEnabled) => {
    const { tool, shouldShowMask } = currentCanvas;

    return {
      tool,
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

export default function IAICanvasEraserControl() {
  const { tool, shouldShowMask, activeTabName, areHotkeysEnabled } =
    useAppSelector(eraserSelector);
  const dispatch = useAppDispatch();

  const handleSelectEraserTool = () => dispatch(setTool('maskEraser'));

  // Hotkeys
  // Set tool to maskEraser
  useHotkeys(
    'e',
    (e: KeyboardEvent) => {
      e.preventDefault();
      handleSelectEraserTool();
    },
    {
      enabled: areHotkeysEnabled,
    },
    [activeTabName, shouldShowMask]
  );

  return (
    <IAIIconButton
      aria-label={
        activeTabName === 'inpainting' ? 'Eraser (E)' : 'Erase Mask (E)'
      }
      tooltip={activeTabName === 'inpainting' ? 'Eraser (E)' : 'Erase Mask (E)'}
      icon={<FaEraser />}
      onClick={handleSelectEraserTool}
      data-selected={tool === 'maskEraser'}
      isDisabled={!shouldShowMask}
    />
  );
}
