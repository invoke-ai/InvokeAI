import { createSelector } from '@reduxjs/toolkit';
import { useHotkeys } from 'react-hotkeys-hook';
import { useAppDispatch, useAppSelector } from 'app/store';
import IAIIconButton from 'common/components/IAIIconButton';
import { currentCanvasSelector, setTool } from 'features/canvas/canvasSlice';

import _ from 'lodash';
import { activeTabNameSelector } from 'features/options/optionsSelectors';
import { BsEraser } from 'react-icons/bs';

const imageEraserSelector = createSelector(
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

export default function IAICanvasImageEraserControl() {
  const { tool, isMaskEnabled, activeTabName } =
    useAppSelector(imageEraserSelector);
  const dispatch = useAppDispatch();

  const handleSelectEraserTool = () => dispatch(setTool('eraser'));

  // Hotkeys
  useHotkeys(
    'shift+e',
    (e: KeyboardEvent) => {
      e.preventDefault();
      handleSelectEraserTool();
    },
    {
      enabled: true,
    },
    [activeTabName, isMaskEnabled]
  );

  return (
    <IAIIconButton
      aria-label="Erase Canvas (Shift+E)"
      tooltip="Erase Canvas (Shift+E)"
      icon={<BsEraser />}
      fontSize={18}
      onClick={handleSelectEraserTool}
      data-selected={tool === 'eraser'}
      isDisabled={!isMaskEnabled}
    />
  );
}
