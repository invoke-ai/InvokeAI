import { createSelector } from '@reduxjs/toolkit';
import { useHotkeys } from 'react-hotkeys-hook';
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
import { BsEraser } from 'react-icons/bs';

const imageEraserSelector = createSelector(
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

export default function IAICanvasImageEraserControl() {
  const { tool, shouldShowMask, activeTabName, areHotkeysEnabled } =
    useAppSelector(imageEraserSelector);
  const dispatch = useAppDispatch();

  const handleSelectEraserTool = () => dispatch(setTool('imageEraser'));

  // Hotkeys
  useHotkeys(
    'shift+e',
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
      aria-label="Erase Canvas (Shift+E)"
      tooltip="Erase Canvas (Shift+E)"
      icon={<BsEraser />}
      fontSize={18}
      onClick={handleSelectEraserTool}
      data-selected={tool === 'imageEraser'}
      isDisabled={!shouldShowMask}
    />
  );
}
