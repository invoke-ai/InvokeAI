import { createSelector } from '@reduxjs/toolkit';
import { useHotkeys } from 'react-hotkeys-hook';
import { FaRedo } from 'react-icons/fa';
import { RootState, useAppDispatch, useAppSelector } from 'app/store';
import IAIIconButton from 'common/components/IAIIconButton';
import { activeTabNameSelector } from 'features/options/optionsSelectors';
import {
  areHotkeysEnabledSelector,
  currentCanvasSelector,
  GenericCanvasState,
  OutpaintingCanvasState,
  redo,
  redoOutpaintingAction,
} from 'features/canvas/canvasSlice';

import _ from 'lodash';

const canvasRedoSelector = createSelector(
  [
    (state: RootState) => state.canvas.outpainting,
    currentCanvasSelector,
    activeTabNameSelector,
    areHotkeysEnabledSelector,
  ],
  (
    outpainting: OutpaintingCanvasState,
    currentCanvas: GenericCanvasState,
    activeTabName,
    areHotkeysEnabled
  ) => {
    const { futureLines, shouldShowMask, tool } = currentCanvas;

    return {
      canRedo:
        tool === 'imageEraser'
          ? outpainting.futureObjects.length > 0
          : futureLines.length > 0,
      shouldShowMask,
      activeTabName,
      areHotkeysEnabled,
      tool,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
    },
  }
);

export default function IAICanvasRedoControl() {
  const dispatch = useAppDispatch();
  const { canRedo, shouldShowMask, activeTabName, tool, areHotkeysEnabled } =
    useAppSelector(canvasRedoSelector);

  const handleRedo = () => {
    if (tool === 'imageEraser') {
      dispatch(redoOutpaintingAction());
    } else {
      dispatch(redo());
    }
  };

  // Hotkeys

  // Redo
  useHotkeys(
    'cmd+shift+z, control+shift+z, control+y, cmd+y',
    (e: KeyboardEvent) => {
      e.preventDefault();
      handleRedo();
    },
    {
      enabled: areHotkeysEnabled && canRedo,
    },
    [activeTabName, shouldShowMask, canRedo, tool]
  );

  return (
    <IAIIconButton
      aria-label="Redo"
      tooltip="Redo"
      icon={<FaRedo />}
      onClick={handleRedo}
      isDisabled={!canRedo || !shouldShowMask}
    />
  );
}
