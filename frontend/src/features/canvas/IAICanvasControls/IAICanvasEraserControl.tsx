import { createSelector } from '@reduxjs/toolkit';
import { useHotkeys } from 'react-hotkeys-hook';
import { FaEraser } from 'react-icons/fa';
import { useAppDispatch, useAppSelector } from 'app/store';
import IAIIconButton from 'common/components/IAIIconButton';
import { currentCanvasSelector, setTool } from 'features/canvas/canvasSlice';

import _ from 'lodash';
import { activeTabNameSelector } from 'features/options/optionsSelectors';

const eraserSelector = createSelector(
  [currentCanvasSelector, activeTabNameSelector],
  (currentCanvas, activeTabName) => {
    const { tool, isMaskEnabled } = currentCanvas;

    return {
      tool,
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

export default function IAICanvasEraserControl() {
  const { tool, isMaskEnabled, activeTabName } = useAppSelector(eraserSelector);
  const dispatch = useAppDispatch();

  const handleSelectEraserTool = () => dispatch(setTool('eraser'));

  // Hotkeys
  // Set tool to maskEraser
  useHotkeys(
    'e',
    (e: KeyboardEvent) => {
      e.preventDefault();
      handleSelectEraserTool();
    },
    {
      enabled: true,
    },
    [activeTabName]
  );

  return (
    <IAIIconButton
      aria-label={
        activeTabName === 'inpainting' ? 'Eraser (E)' : 'Erase Mask (E)'
      }
      tooltip={activeTabName === 'inpainting' ? 'Eraser (E)' : 'Erase Mask (E)'}
      icon={<FaEraser />}
      onClick={handleSelectEraserTool}
      data-selected={tool === 'eraser'}
      isDisabled={!isMaskEnabled}
    />
  );
}
